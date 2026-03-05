from __future__ import annotations

import json
import logging
from typing import Any

from . import planner
from .config import get_config
from .db import (
    _get_state_version,
    bump_event_version,
    bump_version,
    count_uncompacted_messages,
    create_summary_and_mark_messages,
    db_transaction,
    get_event_version,
    get_fact_by_id,
    get_fact_rows_for_ids,
    get_messages_for_compaction,
    get_vector_watermark,
    get_version,
    insert_event,
    insert_fact,
    insert_message,
    insert_outbox,
    insert_planner_run,
    insert_step_trace,
    list_pending_consolidation_proposals,
    persist_result,
)
from .executor import execute_step
from .interrupt import InterruptChannel
from .llm import llm_call
from .models import ContextSnapshot, FingerprintDiff, ValidatedPlanStep
from .obsidian import write_decision_note, write_event_note, write_weekly_review
from .retrieval import build_context_snapshot, fingerprint_diff, refresh_context_fingerprint
from .working_memory import working_memory

logger = logging.getLogger(__name__)

VERSION_DRIFT_THRESHOLD: int | None = None  # Deprecated: use Config.VERSION_DRIFT_THRESHOLD only in threshold mode.
MAX_REPLANS_PER_EVENT = 3
SAFE_REPLAN_FALLBACK_MESSAGE = (
    "I hit repeated context changes while working on that, so I'm stopping here instead of looping."
)
_PINNED_FACT_KINDS = {"user_model", "preference"}


def extract_goal(raw: dict[str, Any]) -> str:
    goal = raw.get("goal")
    if isinstance(goal, str) and goal.strip():
        return goal.strip()
    text = raw.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return json.dumps(raw, ensure_ascii=False, sort_keys=True)


def extract_facts_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for item in result.get("facts", []):
        if isinstance(item, str) and item.strip():
            facts.append({"content": item.strip(), "importance": "contextual", "tier": "active", "meta": {}})
        elif isinstance(item, dict):
            content = str(item.get("content", "")).strip()
            if content:
                facts.append(
                    {
                        "content": content,
                        "importance": item.get("importance"),
                        "tier": item.get("tier"),
                        "meta": item.get("meta", {}),
                    }
                )
    return facts


def _maybe_compact_messages() -> None:
    if count_uncompacted_messages() <= 50:
        return
    rows = get_messages_for_compaction(limit=30)
    if len(rows) < 30:
        return
    transcript = "\n".join(f"{row['role']}: {row['content']}" for row in rows)
    summary = llm_call(
        "Summarize these older assistant messages for future planning. Preserve durable facts and open commitments.",
        transcript,
    )
    create_summary_and_mark_messages([row["id"] for row in rows], summary)


def _normalize_importance(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"core", "contextual", "transient"}:
        return normalized
    return "contextual"


def _normalize_tier_change(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"cold", "archived"}:
        return normalized
    return None


def _normalize_source_fact_ids(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("Proposal source ids must be a non-empty list.")
    normalized_ids = [int(item) for item in value]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Proposal source ids must be unique.")
    return normalized_ids


def _proposal_referenced_fact_ids(proposal: dict[str, Any]) -> list[int]:
    proposal_type = str(proposal.get("type") or "").strip()
    if proposal_type == "merge":
        try:
            return _normalize_source_fact_ids(proposal.get("source_fact_ids"))
        except (TypeError, ValueError):
            return []
    if proposal_type == "tier_change":
        try:
            fact_id = int(proposal.get("fact_id"))
        except (TypeError, ValueError):
            return []
        return [fact_id] if fact_id > 0 else []
    return []


def _is_pinned_fact_kind(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    kind = str(meta.get("kind") or "").strip().lower()
    return kind in _PINNED_FACT_KINDS


def _reject_proposal(
    conn: Any,
    proposal_id: int,
    *,
    reason: str | None = None,
    proposal: dict[str, Any] | None = None,
    affected_fact_id: int | None = None,
) -> None:
    # consolidation_proposals.status has no DB CHECK constraint; 'rejected' is already part of runtime lifecycle.
    if reason and isinstance(proposal, dict):
        updated_proposal = dict(proposal)
        meta = updated_proposal.get("meta")
        updated_meta = dict(meta) if isinstance(meta, dict) else {}
        updated_meta["reason"] = reason
        updated_meta["rejection_reason"] = reason
        if affected_fact_id is not None:
            updated_meta["affected_fact_id"] = int(affected_fact_id)
        updated_proposal["meta"] = updated_meta
        conn.execute(
            """
            UPDATE consolidation_proposals
            SET status = 'rejected', proposal_json = ?
            WHERE id = ?
            """,
            (json.dumps(updated_proposal, ensure_ascii=False), proposal_id),
        )
        return
    conn.execute(
        "UPDATE consolidation_proposals SET status = 'rejected' WHERE id = ?",
        (proposal_id,),
    )


def _merge_source_meta(meta: dict[str, Any] | None, *, merged_fact_id: int, proposal_id: int, version: int) -> dict[str, Any]:
    updated = dict(meta or {})
    updated["merged_into_fact_id"] = merged_fact_id
    updated["merged_by_proposal_id"] = proposal_id
    updated["merged_at_version"] = version
    return updated


def _archive_source_meta(meta: dict[str, Any] | None, *, proposal_id: int, version: int) -> dict[str, Any]:
    updated = dict(meta or {})
    updated["archived_by_proposal_id"] = proposal_id
    updated["archived_at_version"] = version
    return updated


def _apply_merge_proposal(conn: Any, proposal_record: dict[str, Any], proposal: dict[str, Any], version: int) -> bool:
    try:
        normalized_ids = _normalize_source_fact_ids(proposal.get("source_fact_ids"))
    except (TypeError, ValueError):
        _reject_proposal(conn, proposal_record["id"])
        return False

    merged_content = str(proposal.get("merged_content") or "").strip()
    if not merged_content or str(proposal.get("source_tier_after") or "").strip().lower() != "cold":
        _reject_proposal(conn, proposal_record["id"])
        return False

    facts = get_fact_rows_for_ids(conn, normalized_ids)
    by_id = {fact.id: fact for fact in facts}
    if any(fid not in by_id for fid in normalized_ids):
        _reject_proposal(conn, proposal_record["id"])
        return False
    if any(by_id[fid].status != "active" or by_id[fid].tier != "active" for fid in normalized_ids):
        _reject_proposal(conn, proposal_record["id"])
        return False

    merged_fact_id = insert_fact(
        conn,
        {
            "content": merged_content,
            "importance": _normalize_importance(proposal.get("merged_importance")),
            "tier": "active",
            "meta": {
                "merged_from": normalized_ids,
                "merge_source_proposal_id": proposal_record["id"],
            },
        },
        version,
    )
    insert_outbox(conn, merged_fact_id)
    for fact_id in normalized_ids:
        source_fact = by_id[fact_id]
        conn.execute(
            """
            UPDATE facts
            SET tier = 'cold', meta_json = ?
            WHERE id = ?
            """,
            (
                json.dumps(
                    _merge_source_meta(
                        source_fact.meta,
                        merged_fact_id=merged_fact_id,
                        proposal_id=proposal_record["id"],
                        version=version,
                    ),
                    ensure_ascii=False,
                ),
                fact_id,
            ),
        )
    conn.execute(
        "UPDATE consolidation_proposals SET status = 'applied' WHERE id = ?",
        (proposal_record["id"],),
    )
    return True


def _apply_tier_change_proposal(conn: Any, proposal_record: dict[str, Any], proposal: dict[str, Any], version: int) -> bool:
    try:
        fact_id = int(proposal.get("fact_id"))
    except (TypeError, ValueError):
        _reject_proposal(conn, proposal_record["id"])
        return False
    new_tier = _normalize_tier_change(proposal.get("new_tier"))
    if fact_id <= 0 or new_tier is None:
        _reject_proposal(conn, proposal_record["id"])
        return False

    facts = get_fact_rows_for_ids(conn, [fact_id])
    if not facts:
        _reject_proposal(conn, proposal_record["id"])
        return False
    fact = facts[0]
    if fact.status != "active":
        _reject_proposal(conn, proposal_record["id"])
        return False
    if fact.tier == new_tier:
        _reject_proposal(conn, proposal_record["id"])
        return False

    if new_tier == "cold":
        conn.execute(
            "UPDATE facts SET tier = 'cold' WHERE id = ?",
            (fact_id,),
        )
    else:
        merged_into_fact_id = (fact.meta or {}).get("merged_into_fact_id")
        if fact.tier != "cold" or not merged_into_fact_id:
            _reject_proposal(conn, proposal_record["id"])
            return False
        conn.execute(
            """
            UPDATE facts
            SET tier = 'archived', status = 'superseded', version_superseded = ?, meta_json = ?
            WHERE id = ?
            """,
            (
                version,
                json.dumps(
                    _archive_source_meta(
                        fact.meta,
                        proposal_id=proposal_record["id"],
                        version=version,
                    ),
                    ensure_ascii=False,
                ),
                fact_id,
            ),
        )

    conn.execute(
        "UPDATE consolidation_proposals SET status = 'applied' WHERE id = ?",
        (proposal_record["id"],),
    )
    return True


def apply_pending_proposals(version: int) -> int:
    applied = 0
    for proposal_record in list_pending_consolidation_proposals():
        proposal = proposal_record.get("proposal")
        if not isinstance(proposal, dict):
            with db_transaction() as conn:
                _reject_proposal(conn, proposal_record["id"])
            continue
        with db_transaction() as conn:
            referenced_fact_ids = _proposal_referenced_fact_ids(proposal)
            referenced_facts = get_fact_rows_for_ids(conn, referenced_fact_ids) if referenced_fact_ids else []
            pinned_fact = next(
                (fact for fact in referenced_facts if _is_pinned_fact_kind(fact.meta)),
                None,
            )
            if pinned_fact is not None:
                _reject_proposal(
                    conn,
                    proposal_record["id"],
                    reason="pinned_fact_protected",
                    proposal=proposal,
                    affected_fact_id=pinned_fact.id,
                )
                logger.warning(
                    "Proposal %s rejected: affects pinned fact %s",
                    proposal_record["id"],
                    pinned_fact.id,
                )
                continue
            proposal_type = str(proposal.get("type") or "").strip()
            if proposal_type == "merge":
                applied += int(_apply_merge_proposal(conn, proposal_record, proposal, version))
                continue
            if proposal_type == "tier_change":
                applied += int(_apply_tier_change_proposal(conn, proposal_record, proposal, version))
                continue
            _reject_proposal(conn, proposal_record["id"])
    return applied


def _persist_planner_run(event_id: int, plan_run: Any) -> int:
    with db_transaction() as conn:
        return insert_planner_run(
            conn,
            event_id=event_id,
            goal=plan_run.goal,
            snapshot=plan_run.snapshot,
            system_prompt=plan_run.system_prompt,
            user_prompt=plan_run.user_prompt,
            planner_status=plan_run.planner_status,
            first_response=plan_run.first_response,
            repair_prompt=plan_run.repair_prompt,
            repair_response=plan_run.repair_response,
            final_steps=[planner.serialize_step(step) for step in plan_run.steps],
            error_text=plan_run.error,
        )


def _build_snapshot(
    current_version: int,
    event_version: int,
    query: str,
    *,
    apply_proposals: bool,
) -> ContextSnapshot:
    vector_watermark = get_vector_watermark()
    snapshot = build_context_snapshot(current_version, event_version, vector_watermark, query)
    if apply_proposals and apply_pending_proposals(current_version):
        vector_watermark = get_vector_watermark()
        snapshot = build_context_snapshot(current_version, event_version, vector_watermark, query)
    return snapshot


def _sort_fact_ids(fact_ids: list[str]) -> list[str]:
    return sorted(
        {str(fact_id) for fact_id in fact_ids},
        key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item),
    )


def _precondition_changed_fact_ids(
    step: ValidatedPlanStep,
    snapshot: ContextSnapshot,
) -> list[str]:
    changed_fact_ids: list[str] = []
    for raw_fact_id in step.precondition_fact_ids:
        try:
            fact_id = int(raw_fact_id)
        except (TypeError, ValueError):
            changed_fact_ids.append(str(raw_fact_id))
            continue
        fact = get_fact_by_id(fact_id)
        expected = snapshot.fingerprint.fact_versions.get(str(fact_id))
        if fact is None or expected is None:
            changed_fact_ids.append(str(fact_id))
            continue
        current_state = (fact.version_created, fact.status, fact.tier)
        if fact.status != "active" or current_state != expected:
            changed_fact_ids.append(str(fact_id))
    return _sort_fact_ids(changed_fact_ids)


def _fingerprint_diff_summary(diff: FingerprintDiff) -> dict[str, Any]:
    return {
        "changed_fact_ids": diff.changed_fact_ids,
        "removed_fact_ids": diff.removed_fact_ids,
        "task_changes": diff.task_changes,
        "message_changes": diff.message_changes,
    }


def _precondition_diff_summary(changed_fact_ids: list[str]) -> dict[str, Any]:
    return {
        "changed_fact_ids": _sort_fact_ids(changed_fact_ids),
        "removed_fact_ids": [],
        "task_changes": False,
        "message_changes": False,
    }


def _trace_replan(
    *,
    planner_run_id: int,
    step_index: int,
    step: ValidatedPlanStep,
    snapshot_state_version: int,
    current_state_version: int,
    revalidation_status: str,
    replan_reason: str,
    replan_count: int,
    replan_diff_summary: dict[str, Any],
) -> None:
    with db_transaction() as conn:
        insert_step_trace(
            conn,
            planner_run_id=planner_run_id,
            step_index=step_index,
            action=step.action,
            tool=step.tool,
            args=step.args_dict(),
            precondition_fact_ids=step.precondition_fact_ids,
            reasoning=step.reasoning,
            snapshot_state_version=snapshot_state_version,
            current_state_version=current_state_version,
            revalidation_status=revalidation_status,
            rejection_reason=replan_reason,
            replan_reason=replan_reason,
            replan_count=replan_count,
            replan_diff_summary=replan_diff_summary,
        )


def _replan_diff_signature(diff_summary: dict[str, Any]) -> str:
    return json.dumps(diff_summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_fallback_step() -> ValidatedPlanStep:
    return ValidatedPlanStep.model_validate(
        {
            "action": "respond",
            "tool": "respond",
            "args": {"message": SAFE_REPLAN_FALLBACK_MESSAGE},
            "precondition_fact_ids": [],
            "reasoning": "Forced fallback after repeated replans to guarantee event termination.",
        }
    )


async def ingest_event(raw: dict[str, Any], interrupt: InterruptChannel) -> list[str]:
    raw_text = str(raw.get("text") or "").strip()
    config = get_config()
    assistant_messages: list[str] = []
    decision_fact_ids: list[int] = []
    generated_reviews: list[dict[str, Any]] = []
    max_replans_per_event = max(1, int(config.max_replans_per_event or MAX_REPLANS_PER_EVENT))
    with db_transaction() as conn:
        event_id = insert_event(conn, raw)
        if raw_text:
            current_version = _get_state_version(conn)
            insert_message(conn, "user", raw_text, state_version=current_version)
        event_version = bump_event_version(conn)
        V = bump_version(conn)

    working_memory.update(raw, V)
    proposal_application_enabled = True
    snapshot = _build_snapshot(
        V,
        event_version,
        raw_text,
        apply_proposals=proposal_application_enabled,
    )
    goal = extract_goal(raw)
    plan_run = planner.plan(snapshot, goal)
    planner_run_id = _persist_planner_run(event_id, plan_run)
    steps = plan_run.steps
    replan_count = 0
    drift_checks_frozen = False
    last_diff_summary: dict[str, Any] | None = None
    last_fingerprint_diff_signature: str | None = None

    try:
        step_index = 0
        while step_index < len(steps):
            if await interrupt.check():
                return assistant_messages
            step = steps[step_index]
            cur_V = get_version()
            if not drift_checks_frozen:
                precondition_changes = _precondition_changed_fact_ids(step, snapshot)
                if precondition_changes:
                    replan_reason = "precondition_changed"
                    diff_summary = _precondition_diff_summary(precondition_changes)
                    last_diff_summary = diff_summary
                    if replan_count >= max_replans_per_event:
                        logger.warning(
                            "max_replans_per_event_reached %s",
                            json.dumps(
                                {
                                    "event_id": event_id,
                                    "replan_count": replan_count,
                                    "replan_reason": replan_reason,
                                    "replan_diff_summary": diff_summary,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        )
                        _trace_replan(
                            planner_run_id=planner_run_id,
                            step_index=step_index,
                            step=step,
                            snapshot_state_version=snapshot.state_version,
                            current_state_version=cur_V,
                            revalidation_status="replan_limit_reached",
                            replan_reason=replan_reason,
                            replan_count=replan_count,
                            replan_diff_summary=diff_summary,
                        )
                        steps = [_safe_fallback_step()]
                        step_index = 0
                        drift_checks_frozen = True
                        last_fingerprint_diff_signature = None
                        continue
                    replan_count += 1
                    _trace_replan(
                        planner_run_id=planner_run_id,
                        step_index=step_index,
                        step=step,
                        snapshot_state_version=snapshot.state_version,
                        current_state_version=cur_V,
                        revalidation_status="replanned",
                        replan_reason=replan_reason,
                        replan_count=replan_count,
                        replan_diff_summary=diff_summary,
                    )
                    current_event_version = get_event_version()
                    snapshot = _build_snapshot(
                        cur_V,
                        current_event_version,
                        raw_text,
                        apply_proposals=proposal_application_enabled,
                    )
                    plan_run = planner.plan(snapshot, goal)
                    planner_run_id = _persist_planner_run(event_id, plan_run)
                    steps = plan_run.steps
                    step_index = 0
                    last_fingerprint_diff_signature = None
                    continue

                if config.drift_policy == "threshold":
                    drift = cur_V - snapshot.state_version
                    if drift > config.VERSION_DRIFT_THRESHOLD:
                        replan_reason = "threshold_exceeded"
                        diff_summary = {
                            "changed_fact_ids": [],
                            "removed_fact_ids": [],
                            "task_changes": False,
                            "message_changes": False,
                            "drift": drift,
                            "threshold": config.VERSION_DRIFT_THRESHOLD,
                        }
                        last_diff_summary = diff_summary
                        if replan_count >= max_replans_per_event:
                            logger.warning(
                                "max_replans_per_event_reached %s",
                                json.dumps(
                                    {
                                        "event_id": event_id,
                                        "replan_count": replan_count,
                                        "replan_reason": replan_reason,
                                        "replan_diff_summary": diff_summary,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                            )
                            _trace_replan(
                                planner_run_id=planner_run_id,
                                step_index=step_index,
                                step=step,
                                snapshot_state_version=snapshot.state_version,
                                current_state_version=cur_V,
                                revalidation_status="replan_limit_reached",
                                replan_reason=replan_reason,
                                replan_count=replan_count,
                                replan_diff_summary=diff_summary,
                            )
                            steps = [_safe_fallback_step()]
                            step_index = 0
                            drift_checks_frozen = True
                            last_fingerprint_diff_signature = None
                            continue
                        replan_count += 1
                        _trace_replan(
                            planner_run_id=planner_run_id,
                            step_index=step_index,
                            step=step,
                            snapshot_state_version=snapshot.state_version,
                            current_state_version=cur_V,
                            revalidation_status="replanned",
                            replan_reason=replan_reason,
                            replan_count=replan_count,
                            replan_diff_summary=diff_summary,
                        )
                        current_event_version = get_event_version()
                        snapshot = _build_snapshot(
                            cur_V,
                            current_event_version,
                            raw_text,
                            apply_proposals=proposal_application_enabled,
                        )
                        plan_run = planner.plan(snapshot, goal)
                        planner_run_id = _persist_planner_run(event_id, plan_run)
                        steps = plan_run.steps
                        step_index = 0
                        last_fingerprint_diff_signature = None
                        continue
                else:
                    current_event_version = get_event_version()
                    if current_event_version != snapshot.event_version:
                        current_fingerprint = refresh_context_fingerprint(snapshot.fingerprint)
                        diff = fingerprint_diff(snapshot.fingerprint, current_fingerprint)
                        if not diff.is_empty:
                            replan_reason = "context_fingerprint_changed"
                            diff_summary = _fingerprint_diff_summary(diff)
                            diff_signature = _replan_diff_signature(diff_summary)
                            last_diff_summary = diff_summary
                            if last_fingerprint_diff_signature == diff_signature:
                                proposal_application_enabled = False
                                drift_checks_frozen = True
                                _trace_replan(
                                    planner_run_id=planner_run_id,
                                    step_index=step_index,
                                    step=step,
                                    snapshot_state_version=snapshot.state_version,
                                    current_state_version=cur_V,
                                    revalidation_status="loop_detected",
                                    replan_reason="replan_loop_detected",
                                    replan_count=replan_count,
                                    replan_diff_summary=diff_summary,
                                )
                                logger.warning(
                                    "replan_loop_detected %s",
                                    json.dumps(
                                        {
                                            "event_id": event_id,
                                            "replan_count": replan_count,
                                            "replan_diff_summary": diff_summary,
                                        },
                                        ensure_ascii=False,
                                        sort_keys=True,
                                    ),
                                )
                            else:
                                if replan_count >= max_replans_per_event:
                                    logger.warning(
                                        "max_replans_per_event_reached %s",
                                        json.dumps(
                                            {
                                                "event_id": event_id,
                                                "replan_count": replan_count,
                                                "replan_reason": replan_reason,
                                                "replan_diff_summary": diff_summary,
                                            },
                                            ensure_ascii=False,
                                            sort_keys=True,
                                        ),
                                    )
                                    _trace_replan(
                                        planner_run_id=planner_run_id,
                                        step_index=step_index,
                                        step=step,
                                        snapshot_state_version=snapshot.state_version,
                                        current_state_version=cur_V,
                                        revalidation_status="replan_limit_reached",
                                        replan_reason=replan_reason,
                                        replan_count=replan_count,
                                        replan_diff_summary=diff_summary,
                                    )
                                    steps = [_safe_fallback_step()]
                                    step_index = 0
                                    drift_checks_frozen = True
                                    last_fingerprint_diff_signature = None
                                    continue
                                replan_count += 1
                                _trace_replan(
                                    planner_run_id=planner_run_id,
                                    step_index=step_index,
                                    step=step,
                                    snapshot_state_version=snapshot.state_version,
                                    current_state_version=cur_V,
                                    revalidation_status="replanned",
                                    replan_reason=replan_reason,
                                    replan_count=replan_count,
                                    replan_diff_summary=diff_summary,
                                )
                                snapshot = _build_snapshot(
                                    cur_V,
                                    current_event_version,
                                    raw_text,
                                    apply_proposals=proposal_application_enabled,
                                )
                                plan_run = planner.plan(snapshot, goal)
                                planner_run_id = _persist_planner_run(event_id, plan_run)
                                steps = plan_run.steps
                                step_index = 0
                                last_fingerprint_diff_signature = diff_signature
                                continue
                        else:
                            snapshot.event_version = current_event_version
                            snapshot.fingerprint = current_fingerprint
                            last_fingerprint_diff_signature = None
            result = await execute_step(step)
            assistant_message = str(result.get("assistant_message") or "").strip()
            if assistant_message:
                assistant_messages.append(assistant_message)
            with db_transaction() as conn:
                persist_result(conn, result, event_id)
                created_fact_ids: list[int] = []
                for fact in extract_facts_from_result(result):
                    fact["source_event_id"] = event_id
                    fact_id = insert_fact(conn, fact, V)
                    insert_outbox(conn, fact_id)
                    created_fact_ids.append(fact_id)
                logged_result = dict(result)
                if created_fact_ids:
                    logged_result["persisted_fact_ids"] = created_fact_ids
                insert_step_trace(
                    conn,
                    planner_run_id=planner_run_id,
                    step_index=step_index,
                    action=step.action,
                    tool=step.tool,
                    args=step.args_dict(),
                    precondition_fact_ids=step.precondition_fact_ids,
                    reasoning=step.reasoning,
                    snapshot_state_version=snapshot.state_version,
                    current_state_version=cur_V,
                    revalidation_status="executed",
                    rejection_reason=None,
                    replan_reason=None,
                    replan_count=replan_count,
                    replan_diff_summary=last_diff_summary,
                    execution_status=str(result.get("status") or "").strip() or "ok",
                    result=logged_result,
                )
                bump_version(conn)
            decision_fact_ids.extend(created_fact_ids)
            generated_reviews.extend(result.get("generated_reviews", []))
            last_diff_summary = None
            last_fingerprint_diff_signature = None
            _maybe_compact_messages()
            if await interrupt.check():
                return assistant_messages
            step_index += 1
        return assistant_messages
    finally:
        for fact_id in decision_fact_ids:
            try:
                write_decision_note(fact_id)
            except Exception:
                logger.exception("Failed to write decision note for fact %s", fact_id)
        for review in generated_reviews:
            try:
                write_weekly_review(review)
            except Exception:
                logger.exception("Failed to write weekly review note for week %s", review.get("week_key"))
        try:
            write_event_note(event_id, assistant_messages)
        except Exception:
            logger.exception("Failed to write inbox note for event %s", event_id)
