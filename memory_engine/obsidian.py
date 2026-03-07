from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import get_config
from .db import (
    get_event_by_id,
    get_fact_record_by_id,
    list_planner_runs_for_event,
    list_step_traces_for_event,
)
from .epistemics import epistemic_label, normalize_confidence_score, normalize_verification_status


def _vault_root() -> Path:
    root = get_config().OBSIDIAN_VAULT_PATH
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_slug(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return cleaned[:60] or fallback


def _local_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _json_block(payload: Any) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def _text_block(payload: str) -> str:
    return "```text\n" + (payload or "") + "\n```"


def _ensure_vault_scaffold() -> None:
    root = _vault_root()
    for folder in ("Inbox", "Decision Log", "Review"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    home = root / "00 Home.md"
    if home.exists():
        return
    home.write_text(
        "\n".join(
            [
                "# Veronia Memory",
                "",
                "- Inbox: operational traces for each processed event.",
                "- Decision Log: durable decisions captured via `remember_fact` with `meta.kind=decision`.",
                "- Review: generated weekly drafts.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _event_note_relpath(event: dict[str, Any]) -> Path:
    created = datetime.fromtimestamp(event["ts"]).astimezone()
    return Path("Inbox") / str(created.year) / f"{created.date().isoformat()}-event-{int(event['id']):06d}.md"


def _decision_note_relpath(fact: dict[str, Any]) -> Path:
    created_at = fact.get("created_at")
    created = (
        datetime.fromtimestamp(created_at).astimezone()
        if isinstance(created_at, (float, int))
        else datetime.now().astimezone()
    )
    meta = fact.get("meta") if isinstance(fact.get("meta"), dict) else {}
    title = str(meta.get("title") or fact.get("content") or "").strip()
    slug = _safe_slug(title, f"decision-{int(fact['id']):06d}")
    filename = f"{created.date().isoformat()}-decision-{int(fact['id']):06d}-{slug}.md"
    return Path("Decision Log") / str(created.year) / filename


def _review_note_relpath(review: dict[str, Any]) -> Path:
    week_key = str(review.get("week_key") or "unknown-week").strip() or "unknown-week"
    year = week_key.split("-")[0]
    return Path("Review") / year / f"{week_key}.md"


def _wikilink(relpath: Path) -> str:
    return "[[" + relpath.with_suffix("").as_posix() + "]]"


def write_event_note(event_id: int, assistant_messages: list[str]) -> Path | None:
    _ensure_vault_scaffold()
    event = get_event_by_id(event_id)
    if event is None:
        return None

    relpath = _event_note_relpath(event)
    fullpath = _vault_root() / relpath
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    planner_runs = list_planner_runs_for_event(event_id)
    all_step_traces = list_step_traces_for_event(event_id)

    lines = [
        "---",
        "type: inbox",
        f"event_id: {event['id']}",
        f"created: {_local_iso(event['ts'])}",
        f"planner_runs: {len(planner_runs)}",
        "---",
        "",
        f"# Inbox Event {event['id']}",
        "",
        f"- Created: `{_local_iso(event['ts'])}`",
        f"- State version before ingest: `{event['state_version']}`",
        "",
        "## Raw Event",
        "",
        _json_block(event["raw_json"]),
        "",
    ]

    if assistant_messages:
        lines.extend(["## Assistant Output", ""])
        lines.extend([f"- {message}" for message in assistant_messages if message.strip()])
        lines.append("")

    for run in planner_runs:
        snapshot = run.get("snapshot") or {}
        step_traces = all_step_traces.get(run["id"], [])
        lines.extend(
            [
                f"## Planner Run {run['id']}",
                "",
                f"- Status: `{run['planner_status']}`",
                f"- Snapshot state version: `{run['snapshot_state_version']}`",
                f"- Vector watermark: `{run['vector_watermark']}`",
                f"- Tasks in snapshot: `{len(snapshot.get('tasks') or [])}`",
                f"- FTS results: `{len(snapshot.get('fts_results') or [])}`",
                f"- Vector results: `{len(snapshot.get('vector_results') or [])}`",
                f"- Delta facts: `{len(snapshot.get('delta_facts') or [])}`",
                f"- Recent messages: `{len(snapshot.get('recent_messages') or [])}`",
                "",
            ]
        )
        if run.get("error_text"):
            lines.extend([f"- Planner error: `{run['error_text']}`", ""])
        lines.extend(["### Snapshot", "", _json_block(snapshot), ""])
        lines.extend(["### First Response", "", _text_block(run.get("first_response") or ""), ""])
        if run.get("repair_prompt"):
            lines.extend(["### Repair Prompt", "", _text_block(run["repair_prompt"]), ""])
        if run.get("repair_response"):
            lines.extend(["### Repair Response", "", _text_block(run["repair_response"]), ""])
        lines.extend(["### Final Steps", "", _json_block(run.get("final_steps") or []), ""])
        lines.extend(["### Step Traces", ""])
        if not step_traces:
            lines.extend(["- No steps were executed for this run.", ""])
            continue
        for trace in step_traces:
            lines.extend(
                [
                    f"#### Step {trace['step_index'] + 1}: `{trace['tool'] or trace['action']}`",
                    "",
                    f"- Revalidation: `{trace['revalidation_status']}`",
                    f"- Execution status: `{trace['execution_status'] or 'n/a'}`",
                    f"- Snapshot version: `{trace['snapshot_state_version']}`",
                    f"- Current version: `{trace['current_state_version'] if trace['current_state_version'] is not None else 'n/a'}`",
                ]
            )
            if trace.get("rejection_reason"):
                lines.append(f"- Rejection reason: `{trace['rejection_reason']}`")
            if trace.get("reasoning"):
                lines.append(f"- Reasoning: {trace['reasoning']}")
            lines.extend(["", _json_block({"args": trace["args"], "result": trace["result"]}), ""])

    fullpath.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return fullpath


def write_decision_note(fact_id: int) -> Path | None:
    _ensure_vault_scaffold()
    fact = get_fact_record_by_id(fact_id)
    if fact is None:
        return None
    meta = fact.get("meta") if isinstance(fact.get("meta"), dict) else {}
    kind = str(meta.get("kind") or meta.get("type") or "").strip().lower()
    if kind != "decision":
        return None

    relpath = _decision_note_relpath(fact)
    fullpath = _vault_root() / relpath
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    title = str(meta.get("title") or fact.get("content") or f"Decision {fact_id}").strip()
    alternatives = meta.get("alternatives")
    if not isinstance(alternatives, list):
        alternatives = []
    tags = meta.get("tags")
    if not isinstance(tags, list):
        tags = []

    lines = [
        "---",
        "type: decision",
        f"fact_id: {fact['id']}",
        f"created: {_local_iso(fact['created_at']) if fact.get('created_at') else datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"status: {fact['status']}",
        "---",
        "",
        f"# {title}",
        "",
        "## Epistemic Status",
        "",
        f"- Confidence: `{normalize_confidence_score(fact.get('confidence_score')):.2f}`",
        f"- Verification: `{normalize_verification_status(fact.get('verification_status'))}`",
        f"- Label: `{epistemic_label(fact.get('verification_status'), fact.get('confidence_score'))}`",
        "",
        "## Decision",
        "",
        str(fact.get("content") or "").strip(),
        "",
    ]

    if meta.get("context"):
        lines.extend(["## Context", "", str(meta["context"]).strip(), ""])
    if meta.get("rationale"):
        lines.extend(["## Rationale", "", str(meta["rationale"]).strip(), ""])
    if alternatives:
        lines.extend(["## Alternatives", ""])
        lines.extend([f"- {str(item).strip()}" for item in alternatives if str(item).strip()])
        lines.append("")
    if tags:
        lines.extend(["## Tags", ""])
        lines.extend([f"- {str(item).strip()}" for item in tags if str(item).strip()])
        lines.append("")
    if fact.get("source_event_id"):
        event = get_event_by_id(int(fact["source_event_id"]))
        if event is not None:
            lines.extend(["## Source", "", f"- Event: {_wikilink(_event_note_relpath(event))}", ""])

    fullpath.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return fullpath


def write_weekly_review(review: dict[str, Any]) -> Path | None:
    _ensure_vault_scaffold()
    relpath = _review_note_relpath(review)
    fullpath = _vault_root() / relpath
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    summary = review.get("summary") if isinstance(review.get("summary"), dict) else {}
    lines = [
        "---",
        "type: weekly-review",
        f"week_key: {summary.get('week_key') or review.get('week_key') or 'unknown'}",
        f"period_start: {summary.get('period_start') or ''}",
        f"period_end: {summary.get('period_end') or ''}",
        "---",
        "",
        str(review.get("markdown") or "").strip(),
        "",
    ]
    fullpath.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return fullpath
