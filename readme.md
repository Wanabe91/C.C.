# Repository Review Map

This document describes the repository as it exists in code, with emphasis on execution flow, state changes, and module boundaries. It is meant to answer "how the current code works", not "how to run it".

## 1. What the repository is

The repository is a local personal-assistant shell around a persistent memory engine.

There are three practical surfaces:

1. `main.py`
   The human-facing CLI. In this repository it only supports text mode.
2. `memory_engine/`
   The runtime that owns planning, retrieval, persistence, indexing, consolidation, and note generation.
3. `Instruments/memory_viewer.py`
   A separate Streamlit inspection tool for SQLite and ChromaDB contents.

The top-level `modules/` package is small. Its main purpose is now to translate app config into an in-process `memory_engine.config.Config` object and keep the engine workers alive.

## 2. High-level runtime architecture

Primary interactive path:

```text
main.py
  -> PersonalAI
    -> modules.planner.Planner
      -> build Config dataclass from config.yaml
      -> set_active_config(...)
      -> assert tool registry integrity
      -> init SQLite schema / migrations
      -> start background workers
         -> memory_engine.indexer.run_indexer()
         -> memory_engine.consolidator.run_consolidator()
      -> memory_engine.loop.ingest_event({"text": user_input}, interrupt)
         -> persist event/message
         -> update transient working memory
         -> build retrieval snapshot + fingerprint
         -> optionally apply pending consolidation proposals
         -> memory_engine.planner.plan(...)
            -> memory_engine.llm.llm_call(...)
         -> execute validated steps
         -> persist assistant message / tasks / facts / reviews
         -> enqueue facts for embedding
         -> write Obsidian notes
```

Background maintenance path:

```text
New fact inserted
  -> embedding_outbox row created
  -> indexer claims outbox item
  -> embeddings.embed(...)
  -> Chroma collection upsert
  -> vector watermark recomputed

Candidate facts reviewed for tiering
  -> consolidator asks LLM for structured proposals
  -> proposals stored in SQLite as pending
  -> next ingest_event() may apply them synchronously
  -> merge sources moved to cold tier
  -> merged fact inserted and queued for embedding
  -> archived status only set for valid cold merge sources
```

Standalone engine path:

```text
memory_engine/main.py
  -> load_config_from_env()
  -> set_active_config(...)
  -> assert tool registry integrity
  -> init_db()
  -> start same indexer and consolidator workers
  -> read JSON events from stdin
  -> feed each event to ingest_event(...)
```

## 3. Entry points and control flow

### 3.1 `main.py`

`main.py` is the user-facing entry point.

Flow:

1. Calls `configure_huggingface_auth()` immediately so embedding model downloads can use `.env` credentials.
2. Loads `config.yaml`.
3. Builds `PersonalAI(config)`.
4. `PersonalAI` constructs `modules.planner.Planner`.
5. In text mode, loops on `input("You: ")`.
6. Each non-empty message is forwarded to `PersonalAI._on_text()`.
7. `_on_text()` awaits `self.planner.run(text)` and prints the returned assistant text.
8. Shutdown calls `Planner.close()` to stop background workers.

Important review note:
Voice and vision are stubbed out and intentionally raise runtime errors in this repository. Only text mode is live.

### 3.2 `modules/planner.py`

This file is the top-level adapter between app config and `memory_engine`.

Responsibilities:

- Normalize the LLM base URL.
- Derive default SQLite, Chroma, and Obsidian paths from the `memory` config.
- Build a `memory_engine.config.Config` dataclass.
- Call `set_active_config(...)` so the rest of the engine reads a single in-process config object.
- Assert that the tool registry is internally consistent.
- Initialize the database.
- Manage background worker lifetimes.
- Forward user input into `memory_engine.loop.ingest_event()`.

Important difference from the older design:
This module no longer exports config through environment variables. The main CLI path now uses `set_active_config(...)` directly. Environment-backed config still exists, but only for the standalone `memory_engine/main.py` entry point.

Runtime flow inside `Planner.run()`:

1. Ensure workers are running.
2. Call `ingest_event({"text": user_input}, interrupt)`.
3. Receive zero or more assistant messages.
4. Join them with newlines.

### 3.3 `memory_engine/main.py`

This is the engine-only entry point.

It does not read `config.yaml`. Instead it:

1. Loads config from environment via `load_config_from_env()`.
2. Sets that config active with `set_active_config(...)`.
3. Asserts registry integrity.
4. Calls `init_db()`.
5. Starts the same indexer and consolidator workers.
6. Reads events from stdin as JSON object, JSON array, or newline-delimited JSON.
7. Sends each event to `ingest_event(...)`.
8. Stops workers on exit.

This path is useful for testing or external orchestration, but it is not the startup path used by the normal CLI.

## 4. Core request lifecycle

The central runtime is `memory_engine.loop.ingest_event()`.

### Step 1: Normalize and persist the incoming event

`ingest_event(raw, interrupt)` begins by:

1. Extracting `raw_text = raw.get("text")`.
2. Opening a SQLite transaction with `db_transaction()`.
3. Inserting the raw event into `events`.
4. If text exists, inserting a `messages` row with role `user`.
5. Bumping `event_version` once for the new event.
6. Bumping the global state `version`.

There are now two logical version counters:

- `event_version`: advances once per ingested event
- `version`: advances on every durable write

That distinction is used later during replanning.

### Step 2: Update transient working memory

`working_memory.update(raw, V)` loads request-scoped transient state from the raw event.

What this stores:

- `constraints`: normalized list of dicts from `raw["constraints"]`
- transient `tasks`: normalized from `raw["tasks"]`

These are in-memory only. They are visible to the current planning pass, but they are not durable unless execution later creates persisted tasks.

### Step 3: Build the context snapshot

`_build_snapshot(...)` calls `retrieval.build_context_snapshot(...)`.

The snapshot combines:

- active persisted tasks from SQLite
- transient working-memory tasks
- transient constraints
- FTS matches from SQLite `facts_fts`
- semantic matches from ChromaDB, filtered by vector watermark
- delta facts newer than the vector watermark
- recent messages, plus the newest summary if one exists
- a context fingerprint used for drift detection

As a side effect, any fact surfaced through FTS, vector search, or `delta_facts` has its `last_accessed_at` and `access_count` updated in SQLite.

### Step 4: Apply pending consolidation proposals

Before planning, `_build_snapshot(..., apply_proposals=True)` may call `apply_pending_proposals(version)`.

For each pending proposal:

1. Parse the normalized proposal payload.
2. Re-load referenced facts inside a transaction.
3. Reject the proposal if facts are missing, inactive, inconsistent, or violate merge/tiering rules.
4. For a `merge`, insert one merged fact, queue it for embedding, and set source facts to `tier = 'cold'`.
5. For a `tier_change` to `cold`, update the fact tier only.
6. For a `tier_change` to `archived`, require the fact to already be a cold merge source, then set `tier = 'archived'`, `status = 'superseded'`, and `version_superseded`.
7. Mark the proposal as `applied`.

If any proposal is applied, the snapshot is rebuilt before planning continues.

### Step 5: Extract the goal and plan steps

`extract_goal(raw)` prefers:

1. `raw["goal"]`
2. otherwise `raw["text"]`
3. otherwise a JSON dump of the raw event

Then `planner.plan(snapshot, goal)`:

1. Builds a JSON-heavy user prompt from the snapshot.
2. Builds a system prompt from a static planning prefix plus the live tool registry prompt block.
3. Calls `llm_call()` with a JSON schema request.
4. Parses the result into `ValidatedPlanStep` objects.
5. If parsing or validation fails, makes one repair call.
6. If repair also fails, falls back to a single `respond` step.

Important current design detail:
The planner is no longer coupled to a hard-coded tool list in the prompt alone. `ValidatedPlanStep` validates each step against `memory_engine.tool_registry`, including typed args validation through the registered Pydantic schema.

### Step 6: Execute steps with replan protection

`ingest_event()` iterates over the validated plan steps.

Before executing a step, the loop can replan for three different reasons:

1. precondition fact drift
   If any `precondition_fact_ids` point to facts that changed state, tier, version, or disappeared
2. threshold drift mode
   If `current_version - snapshot.state_version` exceeds `VERSION_DRIFT_THRESHOLD`
3. fingerprint drift mode
   If `event_version` changed and the refreshed context fingerprint differs from the snapshot fingerprint

The default active behavior from `Config` is:

- `drift_policy = "fingerprint"`
- `max_replans_per_event = 3`

In fingerprint mode, the diff tracks:

- changed tracked fact versions
- removed tracked facts
- active task set changes
- newest summary/message changes

Loop protection behavior:

- repeated replans over the same fingerprint diff are treated as a loop
- when that happens, proposal application is disabled and further drift checks are frozen for the remaining event
- if replans exceed `max_replans_per_event`, the remaining plan is replaced with one safe fallback `respond` step

This is one of the biggest differences from the older README. The current engine has explicit loop-detection and bounded replanning instead of only version-threshold revalidation.

### Step 7: Execute the step

Execution is handled by `memory_engine.executor.execute_step(step)`.

Current behavior:

- looks up the tool in `tool_registry`
- validates/coerces args against the registered schema
- calls the registered handler
- attaches step metadata to the returned result

The executor is registry-driven. It does not use a large hand-written switch anymore.

Supported registry tools are:

- `respond`
- `remember_fact`
- `create_task`
- `complete_task`
- `generate_weekly_review`
- `noop`

The handlers are intentionally narrow. They produce declarative result dicts and do not write to the database directly.

### Step 8: Persist results

After each executed step, `ingest_event()`:

1. Calls `persist_result(conn, result, event_id)`.
2. Persists assistant messages into `messages`.
3. Persists created tasks into `tasks`.
4. Marks completed tasks in `tasks`.
5. Persists generated weekly reviews into `weekly_reviews`.
6. Extracts durable facts from `result["facts"]`.
7. Inserts each fact into `facts`.
8. Inserts each fact into `embedding_outbox`.
9. Writes a `step_traces` row containing both revalidation context and the execution result.
10. Bumps global `version` again.

After the event loop exits, `ingest_event()` also attempts to write Obsidian notes for:

- the processed event
- any decision facts where `meta.kind == "decision"` or `meta.type == "decision"`
- generated weekly reviews

### Step 9: Compact old messages if needed

`_maybe_compact_messages()` runs after each executed step.

If more than 50 unsummarized messages exist:

1. Pull the oldest 30 unsummarized messages.
2. Ask the LLM to summarize them.
3. Insert one `summaries` row.
4. Mark those messages with that `summary_id`.

Later retrieval injects the newest summary back into context as a synthetic `system` message.

## 5. Database and persistence model

All durable state is centered in SQLite, initialized by `memory_engine/db.py`.

### 5.1 Schema overview

The schema creates:

- `state_versions`
  Single-row global counters for both `version` and `event_version`.
- `events`
  Raw ingested events.
- `facts`
  Durable memory units with `status`, `importance`, `tier`, source metadata, and access metadata.
- `tasks`
  Persisted active/completed tasks.
- `embedding_outbox`
  Queue for facts that still need vector indexing.
- `consolidation_proposals`
  Pending `merge` / `tier_change` operations generated by the consolidator.
- `messages`
  User and assistant dialogue history.
- `summaries`
  Summaries of compacted message windows.
- `planner_runs`
  Captured planner input snapshot, prompts, raw responses, repair data, and final steps.
- `step_traces`
  Per-step replan, rejection, execution, and result traces.
- `weekly_reviews`
  Generated weekly review artifacts.
- `facts_fts`
  SQLite FTS5 index over fact content.
- `vector_state`
  Watermark showing the newest fact version that is safe to rely on from vector search.

There is also an insert trigger that mirrors each new fact into `facts_fts`.

### 5.2 Migration strategy

`init_db()` does more than create tables. It also performs lightweight runtime migrations.

Notable migrations:

- upgrades `state_versions` to include both `version` and `event_version`
- adds missing columns to long-lived tables
- normalizes invalid `importance`, `tier`, and `access_count` values

That means this file is not just schema declaration. It also owns backward-compatibility for older local databases.

### 5.3 Transaction strategy

`db_transaction()` opens a fresh SQLite connection and executes `BEGIN IMMEDIATE`.

Implications:

- writes are serialized at transaction start
- helpers usually use fresh short-lived connections
- the engine depends on WAL mode and short transactions for concurrency

### 5.4 Versioning strategy

There are now three important progress markers:

1. `state_versions.version`
   Tracks every durable SQLite mutation.
2. `state_versions.event_version`
   Tracks semantic request turns, one increment per ingested event.
3. `vector_state.watermark`
   Tracks which fact versions are safely represented in Chroma.

The separation is deliberate:

- `version` is sensitive enough for write-level drift detection
- `event_version` is coarse enough to decide whether replanning is even worth checking
- `vector_state.watermark` protects retrieval correctness while indexing is asynchronous

## 6. Retrieval flow

Retrieval logic lives in `memory_engine/retrieval.py`.

### 6.1 Full-text retrieval

`fts_search(query, n)`:

1. tokenizes the query with `\w+`
2. builds an OR-based prefix query like `token1* OR token2*`
3. queries `facts_fts`
4. returns active `Fact` objects from SQLite

### 6.2 Vector retrieval

`chroma_search(query, n, vector_watermark)`:

1. lazily initializes a Chroma collection named `facts`
2. uses `MemoryEmbeddingFunction`, which delegates to `embeddings.embed(...)`
3. filters vector hits to `version_created <= vector_watermark`
4. re-loads each fact from SQLite and ignores inactive facts

SQLite remains the source of truth even for vector hits.

### 6.3 Chroma availability fallback

If Chroma import or initialization fails, retrieval disables vector search globally via `_disable_vector_store(...)` and continues with FTS plus delta facts.

This is a graceful-degradation path, not a startup failure.

### 6.4 Delta facts

`get_delta_facts(current_version, vector_watermark)` returns facts that are already in SQLite but too new to trust from the vector store yet.

This prevents newly-created facts from disappearing during the gap between durable insert and async indexing.

### 6.5 Context fingerprint

`build_context_fingerprint(...)` captures the parts of context that matter for safe execution:

- tracked fact ids with `(version_created, status, tier)`
- active persisted task ids
- newest summary id
- newest message id

This fingerprint is later refreshed and diffed to decide whether the current plan is still valid.

## 7. Planning flow

Planning logic lives in `memory_engine/planner.py`.

### 7.1 Prompt construction

The planner system prompt combines:

- a static prefix requiring valid JSON and short deterministic plans
- `registry_prompt_block()` generated from the live tool registry
- optional assistant-style guidance from `ASSISTANT_SYSTEM_PROMPT`

This means the planner prompt and the executable tool surface are generated from the same registry source.

### 7.2 JSON robustness strategy

Because LLM output may contain extra text:

- `_normalize_planner_output()` strips `<think>...</think>` blocks
- `_find_json_fragment()` searches for the first balanced JSON object/array
- `_parse_steps()` validates shape and converts items into `ValidatedPlanStep`

If parsing fails, the code performs one repair call and then falls back.

### 7.3 Planner output contract

Each step must contain:

- `action`
- `tool`
- `args`
- `precondition_fact_ids`
- `reasoning`

`ValidatedPlanStep` then enforces:

- the tool must exist in the registry
- args must validate against that tool's Pydantic schema
- `precondition_fact_ids` must normalize to numeric string ids

This is stricter than the older implementation, where tool alignment depended mostly on prompt discipline and executor branches.

## 8. Execution flow

Execution logic lives in `memory_engine/executor.py` and `memory_engine/tool_registry.py`.

The tool registry owns:

- the tool name
- the args schema
- the handler
- the planner hint used in the planning prompt

The executor does not mutate the database directly. It translates a validated step into a result payload with:

- `assistant_message`
- `facts`
- `created_tasks`
- `completed_task_ids`
- `generated_reviews`
- `meta`

That keeps durable side effects centralized in `loop.py` and `db.py`.

Notable review detail:
`executor.revalidate(...)` still exists, but the main loop now performs the active replan checks itself through fingerprint and precondition logic. That helper is not on the hot path.

## 9. Background workers

Two async workers run alongside the request path.

### 9.1 Indexer

`memory_engine/indexer.py` loops until `stop_event` is set.

Flow:

1. claim one pending outbox item
2. re-load the fact from SQLite
3. skip and mark done if the fact is missing or inactive
4. get the Chroma collection
5. generate an embedding with `embeddings.embed(...)`
6. upsert vector, document, and metadata into Chroma
7. mark the outbox row done
8. recompute the vector watermark

If embedding or upsert fails:

- log the exception
- put the outbox row back to `pending`
- increment its attempt counter

### 9.2 Consolidator

`memory_engine/consolidator.py` periodically asks the LLM for memory-tiering proposals.

Flow:

1. build a candidate set from both recent active facts and stale active facts
2. serialize each fact with id, content, importance, tier, created/access timestamps, and access count
3. ask the LLM for JSON proposals
4. normalize and validate each proposal locally
5. skip proposals already pending
6. insert up to five new pending proposals into SQLite

Important design detail:
The consolidator never applies changes directly. It only writes proposals. Actual mutation still happens synchronously inside `ingest_event()`.

## 10. LLM and embedding clients

### 10.1 `memory_engine/llm.py`

This is the active LLM client for the memory engine.

Behavior:

- sends OpenAI-style chat requests to `LLM_BASE_URL + /chat/completions`
- always uses `temperature = 0`
- tries JSON mode with `response_format = {"type": "json_object"}`
- caches whether that JSON mode is supported
- falls back to prompt-only JSON repair when the backend rejects `response_format`

### 10.2 `memory_engine/embeddings.py`

Embedding has two backends:

1. `lmstudio`
   POST to `/embeddings`
2. `sentence_transformers`
   load a local or downloaded `SentenceTransformer`

The model object is cached with `lru_cache`.

### 10.3 `memory_engine/http_client.py`

This helper changes proxy behavior based on destination host. Local loopback hosts bypass proxy env settings.

This matters because the code assumes local LLM and embedding calls should usually ignore system proxies.

### 10.4 `memory_engine/hf_auth.py`

This helper loads `.env` and mirrors `HF_TOKEN` into `HUGGINGFACE_HUB_TOKEN` if needed. It exists only to make Hugging Face model downloads work with either variable name.

## 11. In-memory state

`memory_engine/working_memory.py` is the only explicit transient state store.

It holds:

- current event-scoped constraints
- current event-scoped transient tasks
- the last update version for that transient state

Important scope rule:
This is not long-term working memory. It only mirrors transient fields provided in the current raw event payload.

## 12. Data models

`memory_engine/models.py` defines the core dataclasses and validation models:

- `Event`
- `Fact`
- `Task`
- `ContextFingerprint`
- `FingerprintDiff`
- `ValidatedPlanStep`
- `PlannerRun`
- `ContextSnapshot`

The most structurally important model change is `ValidatedPlanStep`, because it moves tool and args validation into the model layer before execution.

## 13. Interrupt model

`memory_engine/interrupt.py` defines:

- `Priority`
- `InterruptChannel`

`InterruptChannel` is an async priority queue. `ingest_event()` checks it before and after step execution and returns early if an interrupt payload is present.

In the current repository, the interrupt channel is constructed and checked, but no active code path injects interrupts. It is infrastructure for future cancellation or priority handling.

## 14. File-by-file repository map

### Root

- `main.py`
  User CLI in text mode.
- `config.yaml`
  App config consumed by the top-level planner adapter.
- `requirements.txt`
  Dependency list for the CLI, engine, Chroma, Streamlit viewer, and local embeddings.
- `readme.md`
  This review-oriented repository map.

### `modules/`

- `modules/planner.py`
  Adapter from app config into active `memory_engine` config and worker lifecycle.
- `modules/llm_client.py`
  Separate generic OpenAI-style client. It is not imported by the active runtime path.

### `memory_engine/`

- `config.py`
  Config dataclass, environment loader, validation, and active-config registry.
- `main.py`
  Engine-only CLI entry point that reads JSON events.
- `loop.py`
  Central orchestration for one ingested event.
- `planner.py`
  LLM-based plan generation and repair.
- `tool_registry.py`
  Tool definitions, schemas, handlers, and planner hints.
- `executor.py`
  Registry-based step execution wrapper.
- `db.py`
  SQLite schema, migrations, transactions, and persistence helpers.
- `retrieval.py`
  FTS + Chroma retrieval, delta facts, and context fingerprinting.
- `indexer.py`
  Async embedding worker.
- `consolidator.py`
  Async proposal generator for tiering and merge operations.
- `embeddings.py`
  Embedding backend adapter.
- `llm.py`
  Active chat-completion client for planning, summarization, and consolidation.
- `interrupt.py`
  Interrupt queue abstraction.
- `models.py`
  Shared dataclasses and validated plan-step model.
- `obsidian.py`
  Writes event, decision, and weekly-review notes into the configured Obsidian vault.
- `weekly_review.py`
  Builds weekly review summaries and Markdown drafts from persisted state.
- `working_memory.py`
  Event-scoped transient constraints and tasks.
- `http_client.py`
  HTTP helper with proxy behavior based on destination host.
- `hf_auth.py`
  Hugging Face token normalization helper.

### `Instruments/`

- `memory_viewer.py`
  Streamlit utility to inspect SQLite tables and Chroma collections. It is not part of the main runtime flow.

## 15. Config flow

There are currently two config paths:

### Main CLI path

```text
config.yaml
  -> loaded by main.py
  -> passed into modules.planner.Planner
  -> converted into Config dataclass
  -> set_active_config(...)
  -> read through get_config()
```

### Engine-only path

```text
environment variables
  -> load_config_from_env()
  -> set_active_config(...)
  -> read through get_config()
```

So the memory engine itself no longer depends on YAML parsing. It depends on an already-activated `Config`.

## 16. Review hotspots

These are the files where reviewers should spend disproportionate attention.

### 16.1 `memory_engine/loop.py`

This is still the real coordinator. It owns:

- event persistence
- version and event-version updates
- snapshot building
- consolidation application
- planning
- bounded replanning
- execution loop
- result persistence
- message compaction
- Obsidian note writes

If behavior seems surprising anywhere in the app, it usually traces back here.

### 16.2 `memory_engine/db.py`

This file owns:

- schema and migrations
- transaction boundaries
- version semantics
- summary compaction persistence
- outbox correctness
- planner/step trace durability

It is the real durable-state contract of the repository.

### 16.3 `memory_engine/retrieval.py`

This file determines what the planner can see and when a plan is considered stale. Review it for:

- FTS query quality
- vector fallback behavior
- watermark correctness
- delta fact behavior
- fingerprint correctness

### 16.4 `memory_engine/tool_registry.py`, `memory_engine/planner.py`, and `memory_engine/executor.py`

These three files now form the main contract:

- the registry defines the executable tool surface
- the planner prompt describes that tool surface
- `ValidatedPlanStep` enforces it
- the executor invokes it

Any change to one without the others can silently break planning or execution.

### 16.5 `modules/planner.py`

This is the bridge between app config and engine config. Path derivation and startup behavior happen here, so misconfiguration bugs often start here rather than deeper in the engine.

## 17. Known structural characteristics for reviewers

These are not necessarily bugs, but they matter when reading changes.

1. The active runtime is local-first.
   The LLM endpoint, embeddings, SQLite, and Chroma are all expected to be local or local-compatible.

2. Planning is LLM-driven, execution is schema-constrained.
   The LLM chooses steps, but only within the registry-defined tool set and args schemas.

3. Replanning is explicit and bounded.
   The current engine has fingerprint-based drift detection, loop detection, and a hard fallback path.

4. Vector indexing is asynchronous by design.
   The engine compensates with `delta_facts` and the vector watermark.

5. Consolidation is proposal-based.
   Background code suggests changes; foreground code applies them.

6. Some config in `config.yaml` is legacy.
   The checked-in file still includes voice, vision, and some older LLM settings that are not on the active text-only path.

7. Some repository code is support code, not active runtime code.
   `modules/llm_client.py` and `Instruments/memory_viewer.py` are useful, but they do not participate in the main request pipeline.

## 18. Short end-to-end summary

For review purposes, the most accurate single-sentence description of the repository is:

`main.py` feeds user text into `memory_engine.loop.ingest_event()`, which persists the event, builds a mixed SQLite/Chroma context plus a drift fingerprint, optionally applies pending memory-tiering proposals, asks the LLM for registry-valid structured steps, executes those steps into durable state changes, writes note artifacts, and relies on background workers to maintain embeddings and proposal-based consolidation over time.
