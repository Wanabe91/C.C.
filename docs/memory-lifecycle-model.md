# Memory Lifecycle Model

## Goal

The project should approximate human-like memory:

1. Keep recent experience in high fidelity.
2. Reinforce what gets reused.
3. Compress older detail into semantic summaries over time.
4. Preserve traceability instead of silently deleting durable detail.
5. Keep salience separate from confidence so repeated recall does not become self-verification.

## Lifecycle

### 1. Capture raw experience

- Stores: `events`, `messages`, `working_memory`
- Purpose: keep the immediate trace of what happened before deciding what deserves durable memory

### 2. Stabilize explicit memory

- Stores: `facts`, `tasks`, `embedding_outbox`
- Purpose: persist explicit "remember this" content and structured outcomes as active durable memory

### 3. Rehearse by reuse

- Stores: `facts.last_accessed_at`, `facts.access_count`
- Purpose: facts that are recalled more often become easier to keep in active circulation

### Epistemic layer

- Stores: `facts.confidence_score`, `facts.verification_status`, `facts.verification_count`, `facts.last_verified_at`, `facts.evidence_json`, `facts.contradiction_group_id`
- Purpose: track how well a memory is supported without confusing repeated retrieval with truth

### 4. Consolidate into semantic memory

- Stores: `consolidation_proposals`, `facts`
- Purpose: merge or cool stale/redundant facts into generalized memory without losing source detail

### 5. Reflect over longer spans

- Stores: `summaries`, `weekly_reviews`
- Purpose: compress many raw exchanges into bounded abstractions such as weekly patterns, decisions, and open loops

## Expected behavior over time

- During a conversation, new memories should appear quickly as active facts.
- Before vector indexing completes, the same memories must still be reachable through delta retrieval.
- Recalled memories should accumulate salience via access timestamps and counts.
- Recalled memories must not automatically gain confidence just because they were reused.
- Older, low-salience memories may be cooled or merged into semantic abstractions.
- Confidence should only move through verification such as user confirmation, external corroboration, or explicit consistency checks.
- Core identity and preference memories should remain pinned and resist automatic compression.
- Weekly or periodic artifacts should summarize patterns, not replace the raw facts or messages silently.

## Non-negotiable invariants

- Explicit memories are durable.
- Pinned identity memory is not auto-compressed.
- Compression preserves traceability.
- Reuse strengthens salience.
- Confidence requires verification.
- Reflection is additive.
