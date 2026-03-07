from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryStage:
    key: str
    title: str
    trigger: str
    stores: tuple[str, ...]
    outcome: str


@dataclass(frozen=True, slots=True)
class MemoryInvariant:
    key: str
    description: str


@dataclass(frozen=True, slots=True)
class MemoryLifecycleModel:
    objective: str
    retrieval_order: tuple[str, ...]
    stages: tuple[MemoryStage, ...]
    invariants: tuple[MemoryInvariant, ...]

    def stage_keys(self) -> tuple[str, ...]:
        return tuple(stage.key for stage in self.stages)

    def invariant_keys(self) -> tuple[str, ...]:
        return tuple(invariant.key for invariant in self.invariants)


DEFAULT_MEMORY_MODEL = MemoryLifecycleModel(
    objective=(
        "Approximate human-like memory by keeping recent interaction details, "
        "reinforcing what gets reused, and periodically compressing raw history "
        "into semantic summaries without silently losing durable facts, while "
        "tracking confidence separately from salience so the system does not self-confirm."
    ),
    retrieval_order=(
        "pinned_facts",
        "fts_results",
        "vector_results",
        "delta_facts",
        "working_memory_refs",
    ),
    stages=(
        MemoryStage(
            key="capture",
            title="Capture raw experience",
            trigger="Every user event and assistant result.",
            stores=("events", "messages", "working_memory"),
            outcome=(
                "Keep the raw interaction trace and any large transient tool output "
                "before deciding what deserves durable memory."
            ),
        ),
        MemoryStage(
            key="stabilize",
            title="Stabilize explicit memory",
            trigger="Explicit remember/save requests and structured results.",
            stores=("facts", "tasks", "embedding_outbox"),
            outcome=(
                "Persist high-fidelity active facts and tasks quickly so they are "
                "queryable before semantic indexing catches up."
            ),
        ),
        MemoryStage(
            key="rehearse",
            title="Reinforce by reuse",
            trigger="Whenever a fact is retrieved into context.",
            stores=("facts.last_accessed_at", "facts.access_count"),
            outcome=(
                "Update salience signals so repeated or recent memories stay easier "
                "to retrieve than one-off incidental details."
            ),
        ),
        MemoryStage(
            key="consolidate",
            title="Compress into semantic memory",
            trigger="Background consolidation over stale or redundant active facts.",
            stores=("consolidation_proposals", "facts"),
            outcome=(
                "Create generalized memories or tier changes while preserving source "
                "facts as cold audit trail rather than deleting detail outright."
            ),
        ),
        MemoryStage(
            key="reflect",
            title="Build periodic abstractions",
            trigger="Message compaction and weekly review generation.",
            stores=("summaries", "weekly_reviews"),
            outcome=(
                "Turn many raw exchanges into bounded summaries that describe patterns, "
                "decisions, and open loops across longer spans of time."
            ),
        ),
    ),
    invariants=(
        MemoryInvariant(
            key="explicit_memories_are_durable",
            description="Explicitly remembered facts must be persisted as facts, not only as chat text.",
        ),
        MemoryInvariant(
            key="pinned_identity_is_not_auto_compressed",
            description=(
                "User-model and preference memories are pinned and excluded from automatic consolidation."
            ),
        ),
        MemoryInvariant(
            key="compression_preserves_traceability",
            description=(
                "Compression may generalize memory, but it must preserve a path back to the original detail."
            ),
        ),
        MemoryInvariant(
            key="reuse_strengthens_salience",
            description="Retrieved memories must update salience markers such as access_count and last_accessed_at.",
        ),
        MemoryInvariant(
            key="confidence_requires_verification",
            description=(
                "Confidence and verification status may only change through explicit verification signals, "
                "not through repeated retrieval or consolidation alone."
            ),
        ),
        MemoryInvariant(
            key="reflection_is_additive",
            description=(
                "Summaries and reviews add higher-level memory artifacts; they do not silently erase the raw record."
            ),
        ),
    ),
)
