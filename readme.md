# Veronia Repository Review Map

This document describes the repository as it exists in code, with emphasis on execution flow, state changes, and module boundaries. It is intended to support code review, so it focuses on what runs, in what order, and where side effects happen.

## 1. What the repository is

The repository contains a local personal-assistant shell wrapped around a persistent memory engine.

There are three practical surfaces:

1. `main.py`
   The user-facing CLI. In this repository it only supports text mode.
2. `memory_engine/`
   The persistent memory runtime. This is where planning, storage, retrieval, indexing, consolidation, and LLM calls happen.
3. `Instruments/memory_viewer.py`
   A separate Streamlit inspection tool for SQLite and ChromaDB contents.

The top-level `modules/` package is very small. Its main purpose is to bridge `config.yaml` into environment variables consumed by `memory_engine`.

## 2. High-level runtime architecture

Primary interactive path:

```text
main.py
  -> PersonalAI
    -> modules.planner.Planner
      -> configure memory_engine env
      -> init SQLite
      -> start background workers
         -> memory_engine.indexer.run_indexer()
         -> memory_engine.consolidator.run_consolidator()
      -> memory_engine.loop.ingest_event({"text": user_input}, interrupt)
         -> persist input event/message
         -> update transient working memory
         -> build retrieval snapshot
         -> apply pending consolidation proposals
         -> memory_engine.planner.plan(...)
            -> memory_engine.llm.llm_call(...)
         -> memory_engine.executor.execute_step(...) for each step
         -> persist assistant message / tasks / facts
         -> enqueue facts for embedding
```

Background maintenance path:

```text
New fact inserted
  -> embedding_outbox row created
  -> indexer claims outbox item
  -> embeddings.embed(...)
  -> Chroma collection upsert
  -> vector watermark recomputed

Recent facts overlap
  -> consolidator proposes merged fact
  -> next ingest_event() applies pending proposals
  -> old facts marked superseded
  -> merged fact inserted and queued for embedding
```

Standalone engine path:

```text
memory_engine/main.py
  -> reads JSON events from stdin or prompt
  -> starts same indexer and consolidator workers
  -> feeds each event into ingest_event(...)
```

## 3. Entry points and control flow

### 3.1 `main.py`

`main.py` is the human-facing entry point.

Flow:

1. Calls `configure_huggingface_auth()` immediately so downstream embedding model downloads can use `.env` credentials.
2. Loads YAML config from `config.yaml`.
3. Builds `PersonalAI(config)`.
4. `PersonalAI` constructs `modules.planner.Planner`.
5. In text mode, loops on `input("You: ")`.
6. Each non-empty message is forwarded to `PersonalAI._on_text()`.
7. `_on_text()` awaits `self.planner.run(text)` and prints the returned assistant text.
8. Shutdown calls `Planner.close()` to stop background workers.

Important review note:
Voice and vision are stubbed out and intentionally raise runtime errors in this repository. The only working mode is text.

### 3.2 `modules/planner.py`

This file is the top-level adapter between app config and `memory_engine`.

Responsibilities:

- Normalize the LLM base URL.
- Derive default SQLite and Chroma paths from `memory` config.
- Export the app config into process environment variables:
  - `LLM_BASE_URL`
  - `LLM_MODEL`
  - `ASSISTANT_SYSTEM_PROMPT`
  - `SQLITE_PATH`
  - `CHROMA_PATH`
  - optional embedding and retrieval tuning vars
- Clear `memory_engine.config.get_config()` cache after environment changes.
- Initialize the database.
- Manage background worker lifetimes.
- Forward user input into `memory_engine.loop.ingest_event()`.

Runtime flow inside `Planner.run()`:

1. Ensure workers are running.
2. Call `ingest_event({"text": user_input}, interrupt)`.
3. Receive zero or more assistant messages.
4. Join them with newlines.

This file is structurally important because it means `memory_engine` does not read `config.yaml` directly. Instead, it reads environment variables populated here.

### 3.3 `memory_engine/main.py`

This is a second entry point for the engine itself.

It does not use `config.yaml`. It relies on environment variables already being set, then:

1. Initializes logging.
2. Materializes config via `get_config()`.
3. Calls `init_db()`.
4. Starts the same indexer and consolidator workers as `modules.planner.Planner`.
5. Reads events from stdin as JSON object, JSON array, or newline-delimited JSON.
6. Sends each event to `ingest_event(...)`.
7. Stops workers on exit.

This path is useful for testing or external orchestration, but it is not the same startup path used by the human CLI.

## 4. Core request lifecycle

The central runtime is `memory_engine.loop.ingest_event()`.

### Step 1: Normalize and persist the incoming event

`ingest_event(raw, interrupt)` begins by:

1. Extracting `raw_text = raw.get("text")`.
2. Opening a SQLite transaction with `db_transaction()`.
3. Inserting the raw event into `events`.
4. If text exists, inserting a `messages` row with role `user`.
5. Bumping the global state version `V`.

This means every user event advances state versioning, even before planning starts.

### Step 2: Update transient working memory

`working_memory.update(raw, V)` loads request-scoped constraints and transient tasks from the raw event.

What this stores:

- `constraints`: normalized list of dicts from `raw["constraints"]`
- transient `tasks`: normalized list from `raw["tasks"]`

These are in-memory only. They are not persisted unless later converted into created tasks by execution results.

### Step 3: Build context snapshot

`build_context_snapshot(V, W, raw_text)` combines several sources:

- active persisted tasks from SQLite
- transient tasks from working memory
- transient constraints from working memory
- FTS matches from SQLite `facts_fts`
- semantic matches from ChromaDB, filtered by vector watermark
- delta facts whose versions are newer than the vector watermark
- recent conversation messages, plus latest summary if one exists

This is the full planning context.

The vector watermark matters because retrieval intentionally separates:

- facts already embedded into Chroma, retrievable via vector search
- newer facts not yet embedded, surfaced via `delta_facts`

That prevents freshly inserted facts from disappearing before the async indexer catches up.

### Step 4: Apply pending consolidation proposals

Before planning, `apply_pending_proposals(V)` checks whether the consolidator has already proposed merged facts.

For each pending proposal:

1. Parse source fact IDs.
2. Re-load source facts in a transaction.
3. Reject the proposal if any source fact is missing or not active.
4. Mark source facts as `superseded`.
5. Insert one merged fact with metadata `{"merged_from": [...]}`.
6. Insert an embedding outbox row for the merged fact.
7. Mark the proposal as `applied`.

If any proposal is applied, the context snapshot is rebuilt before planning.

### Step 5: Extract the goal and plan steps

`extract_goal(raw)` prefers:

1. `raw["goal"]` if present
2. otherwise `raw["text"]`
3. otherwise JSON dump of the whole event

Then `planner.plan(snapshot, goal)`:

1. Builds a JSON-heavy planning prompt.
2. Calls `llm_call()` with a strict schema request.
3. Parses plan JSON into `PlanStep` objects.
4. If parsing fails, asks the LLM to repair output once.
5. If repair also fails, falls back to a single `respond` step.

Allowed planning tools are constrained by prompt to:

- `respond`
- `remember_fact`
- `create_task`
- `complete_task`
- `noop`

There is no dynamic tool registry. The planner prompt and executor implementation must stay aligned manually.

### Step 6: Execute the plan

`ingest_event()` iterates over `steps`.

Before each step:

1. Check the interrupt queue.
2. Compare current state version against snapshot version.
3. Revalidate all precondition fact IDs.
4. If state drift exceeds threshold or a fact changed, rebuild context and re-plan from scratch.

Execution itself is handled by `executor.execute_step(step)`.

Supported behaviors:

- `respond`
  Produces an assistant message only.
- `remember_fact`
  Produces new fact payloads to persist.
- `create_task`
  Produces task rows to persist.
- `complete_task`
  Produces task IDs to mark completed.
- `noop`
  Produces no side effects.
- unsupported tool
  Produces an error status and assistant message.

The executor is intentionally simple. It does not perform external I/O or real tool calls beyond packaging state changes.

### Step 7: Persist results

After each executed step, `ingest_event()`:

1. Calls `persist_result(conn, result, event_id)`.
2. Persists assistant messages into `messages`.
3. Persists created tasks into `tasks`.
4. Marks completed tasks in `tasks`.
5. Extracts facts from `result["facts"]`.
6. Inserts each fact into `facts`.
7. Inserts each fact into `embedding_outbox`.
8. Bumps state version again.

This means one user event can advance the global version multiple times:

- once after event ingestion
- once after each executed step

### Step 8: Compact old messages if needed

`_maybe_compact_messages()` runs after each step.

If more than 50 unsummarized messages exist:

1. Pull the oldest 30 unsummarized messages.
2. Ask the LLM to summarize them.
3. Insert one `summaries` row.
4. Mark those messages with `summary_id`.

Later retrieval injects the newest summary back into context as a synthetic system message.

## 5. Database and persistence model

All durable state is centered in SQLite, initialized by `memory_engine/db.py`.

### 5.1 Schema overview

The schema creates:

- `state_versions`
  Single-row global version counter.
- `events`
  Raw ingested events.
- `facts`
  Durable memory units, optionally superseded later.
- `tasks`
  Persisted active/completed tasks.
- `embedding_outbox`
  Queue for facts that still need vector indexing.
- `consolidation_proposals`
  Candidate merges generated by the consolidator.
- `messages`
  User and assistant dialogue history.
- `summaries`
  Summaries of compacted message windows.
- `facts_fts`
  SQLite FTS5 index over fact content.
- `vector_state`
  Tracks the latest version known to be safely represented in the vector store.

There is also an insert trigger that mirrors each new fact into `facts_fts`.

### 5.2 Transaction strategy

`db_transaction()` opens a fresh SQLite connection and executes `BEGIN IMMEDIATE`.

Implications:

- writes are serialized at transaction start
- each helper call usually uses a new connection
- the engine depends on WAL mode and short transactions for concurrency

### 5.3 Versioning strategy

There are two related but different progress markers:

1. `state_versions.version`
   Tracks logical state changes in SQLite.
2. `vector_state.watermark`
   Tracks which fact versions are already safe to rely on from vector search.

The separation is central to correctness. New facts become visible immediately in SQLite and `delta_facts`, but only later in ChromaDB after indexing succeeds.

### 5.4 Message storage strategy

Conversation history is split between:

- raw user/assistant rows in `messages`
- coarse summaries in `summaries`

`get_recent_messages()` returns:

- the latest summary as a synthetic `system` message, if present
- then the most recent unsummarized messages

This is the only place old conversation is surfaced back to planning.

## 6. Retrieval flow

Retrieval logic lives in `memory_engine/retrieval.py`.

### 6.1 Full-text retrieval

`fts_search(query, n)`:

1. Tokenizes the query using `\w+`.
2. Builds an OR-based prefix query like `token1* OR token2*`.
3. Queries the SQLite FTS table.
4. Returns active `Fact` objects.

### 6.2 Vector retrieval

`chroma_search(query, n, vector_watermark)`:

1. Lazily initializes a Chroma collection called `facts`.
2. Uses the local embedding function wrapper `MemoryEmbeddingFunction`.
3. Filters results to `version_created <= vector_watermark`.
4. Re-loads fact IDs from SQLite and ignores inactive facts.

The implementation treats SQLite as source of truth even for vector hits.

### 6.3 Chroma availability fallback

If Chroma import or initialization fails, retrieval disables vector search globally via `_disable_vector_store(...)` and continues with FTS plus delta facts.

This is a graceful degradation path rather than a hard startup failure.

### 6.4 Context composition

`build_context_snapshot(...)` merges:

- persisted active tasks
- transient working-memory tasks
- constraints
- FTS fact hits
- vector fact hits
- delta facts not yet reflected in vector storage
- recent messages

This snapshot is passed as JSON to the planner LLM.

## 7. Planning flow

Planning logic lives in `memory_engine/planner.py`.

### 7.1 Prompt construction

The planner system prompt hard-codes:

- output must be valid JSON
- allowed tools
- preference for short deterministic plans
- respond directly when possible

If `ASSISTANT_SYSTEM_PROMPT` exists, it is appended only as style guidance for `respond` steps.

### 7.2 JSON robustness strategy

Because LLM output may contain extra text:

- `_normalize_planner_output()` removes `<think>...</think>` blocks
- `_find_json_fragment()` scans for the first balanced JSON object or array
- `_parse_steps()` converts parsed payload into `PlanStep` dataclasses

If parsing still fails, the code performs one repair call and then falls back.

### 7.3 Planner output contract

Each step must contain:

- `action`
- `tool`
- `args`
- `precondition_fact_ids`
- `reasoning`

Executor behavior depends on this contract, but there is no runtime schema validation outside JSON parsing and field coercion.

## 8. Execution flow

Execution logic lives in `memory_engine/executor.py`.

The executor is a pure translator from `PlanStep` into a structured result dict.

It does not:

- call external services
- mutate the database directly
- update vector storage

Instead it returns a declarative result that `loop.py` persists afterward.

This split is useful for review because it keeps side effects centralized in `loop.py` and `db.py`.

Revalidation logic also lives here:

- if global version drift exceeds `VERSION_DRIFT_THRESHOLD`, the step is discarded and planning restarts
- if any precondition fact is missing or no longer active, planning restarts

## 9. Background workers

Two async workers run alongside the request path.

### 9.1 Indexer

`memory_engine/indexer.py` loops forever until `stop_event` is set.

Flow:

1. Claim one pending outbox item.
2. Re-load the fact from SQLite.
3. Skip and mark done if fact is missing or inactive.
4. Load Chroma collection.
5. Generate embedding with `embeddings.embed(...)`.
6. Upsert document, vector, and metadata into Chroma.
7. Mark outbox row done.
8. Recompute vector watermark.

If embedding or upsert fails:

- log exception
- mark outbox row back to `pending`
- increment attempt counter

### 9.2 Consolidator

`memory_engine/consolidator.py` periodically scans recent active facts for overlap.

Flow:

1. Load recent active facts.
2. Compare fact pairs using lexical overlap and containment heuristics.
3. Skip pairs already pending.
4. Ask the LLM to produce one merged fact JSON object.
5. Insert up to five pending proposals per cycle.

Important design detail:
The consolidator does not apply merges directly. It only proposes them. Actual mutation happens synchronously inside the next `ingest_event()` call.

That keeps all state mutation inside the main event-processing path.

## 10. LLM and embedding clients

### 10.1 `memory_engine/llm.py`

This is the active LLM client for the memory engine.

Behavior:

- sends OpenAI-style chat requests to `LLM_BASE_URL + /chat/completions`
- uses `temperature = 0`
- optionally requests JSON mode with `response_format = {"type": "json_object"}`
- caches whether JSON mode is supported
- falls back to prompt-only JSON repair when the backend rejects `response_format`

The code is written around LM Studio semantics, but it only requires an OpenAI-compatible local endpoint.

### 10.2 `memory_engine/embeddings.py`

Embedding has two backends:

1. `lmstudio`
   POST to `/embeddings`
2. `sentence_transformers`
   Load a local or downloaded `SentenceTransformer`

The model object is cached with `lru_cache`.

### 10.3 `memory_engine/http_client.py`

This helper disables proxy environment usage for loopback hosts. It uses `trust_env=False` for localhost-style URLs and `True` otherwise.

This matters because the code assumes local LLM calls should usually bypass system proxies.

### 10.4 `memory_engine/hf_auth.py`

This helper loads `.env` and mirrors `HF_TOKEN` into `HUGGINGFACE_HUB_TOKEN` if needed. It exists purely to make Hugging Face model downloads work with either env var name.

## 11. In-memory state

`memory_engine/working_memory.py` is the only explicit transient state store.

It holds:

- current event-scoped constraints
- current event-scoped transient tasks
- the last seen state version

It is protected by a thread lock and returns copies of data on reads.

Important scope rule:
This is not conversational working memory in the long-term sense. It only mirrors fields provided in the current raw event payload.

## 12. Data models

`memory_engine/models.py` defines the core dataclasses:

- `Event`
- `Fact`
- `Task`
- `PlanStep`
- `ContextSnapshot`

They are simple containers. They do not contain behavior.

This keeps serialization and persistence logic elsewhere, mainly in `db.py`, `planner.py`, and `retrieval.py`.

## 13. Interrupt model

`memory_engine/interrupt.py` defines:

- `Priority`
- `InterruptChannel`

`InterruptChannel` is an async priority queue. `ingest_event()` checks it before and after step execution and returns early if any interrupt payload is present.

In the current repository, the interrupt channel is constructed and checked, but no active code path sends interrupts. It is infrastructure for future cancellation or higher-priority events.

## 14. File-by-file repository map

### Root

- `main.py`
  User CLI in text mode.
- `config.yaml`
  App config consumed by top-level planner adapter.
- `requirements.txt`
  Dependency list for CLI, memory engine, Streamlit viewer, Chroma, and local embeddings.
- `redm.md`
  This review-oriented repository map.

### `modules/`

- `modules/planner.py`
  Adapter from app config into `memory_engine`.
- `modules/llm_client.py`
  Separate generic OpenAI-style LLM client. It is not imported by the active runtime path in this repository.

### `memory_engine/`

- `__init__.py`
  Re-exports `Config`, `InterruptChannel`, `Priority`, and `start`.
- `config.py`
  Loads and validates engine configuration from environment.
- `main.py`
  Engine-only CLI entry point that reads JSON events.
- `loop.py`
  Central orchestration for one ingested event.
- `planner.py`
  LLM-based step planning.
- `executor.py`
  Declarative step execution.
- `db.py`
  SQLite schema, transactions, and persistence helpers.
- `retrieval.py`
  FTS + Chroma retrieval and context assembly.
- `indexer.py`
  Async embedding worker.
- `consolidator.py`
  Async fact-merge proposal worker.
- `embeddings.py`
  Embedding backend adapter.
- `llm.py`
  Active chat-completion client for planning and summarization.
- `interrupt.py`
  Interrupt queue abstraction.
- `models.py`
  Shared dataclasses.
- `working_memory.py`
  Event-scoped in-memory constraints and transient tasks.
- `http_client.py`
  HTTP helper with proxy behavior based on destination host.
- `hf_auth.py`
  Hugging Face token normalization helper.

### `Instruments/`

- `memory_viewer.py`
  Streamlit utility to inspect SQLite tables and Chroma collections. It is not part of the main runtime flow.

### `data/`

- runtime-generated SQLite WAL files and Chroma state
- not source code, but important for understanding persistence during review

## 15. Config flow

The config path is easy to miss during review:

```text
config.yaml
  -> loaded by main.py
  -> passed into modules.planner.Planner
  -> converted into environment variables
  -> read by memory_engine.config.get_config()
```

So there are two config layers:

1. YAML app config
2. environment-backed engine config

The memory engine itself never parses YAML.

## 16. Review hotspots

These are the places where reviewers should spend disproportionate attention because they control correctness or coupling.

### 16.1 `memory_engine/loop.py`

This is the real coordinator. It owns:

- event persistence
- version bumps
- snapshot building
- consolidation application
- planning
- execution loop
- result persistence
- message compaction

If behavior seems surprising anywhere in the app, it usually traces back here.

### 16.2 `memory_engine/db.py`

This file owns the actual data model and almost all durable side effects. Review it for:

- version semantics
- transaction boundaries
- schema migration behavior
- summary compaction behavior
- outbox correctness

### 16.3 `memory_engine/retrieval.py`

This file determines what the planner can see. Review it for:

- FTS query quality
- vector fallback behavior
- deduplication gaps between FTS, vector, and delta facts
- task merging rules

### 16.4 `memory_engine/planner.py` and `memory_engine/executor.py`

These two files form a hard contract:

- planner prompt says what tools exist
- executor decides what those tools actually do

Any change to one without the other can silently break task execution.

### 16.5 `modules/planner.py`

This is the bridge between application config and engine config. Path derivation and environment mutation happen here, so misconfiguration bugs often start here rather than in `memory_engine`.

## 17. Known structural characteristics for reviewers

These are not necessarily bugs, but they are important to understand before reviewing changes.

1. The assistant path is mostly single-process and local.
   The LLM, embeddings, SQLite, and Chroma are all expected to run locally or be reachable as local-compatible services.

2. Planning is LLM-driven, execution is not.
   The LLM chooses structured steps, but side effects are restricted to a fixed executor.

3. State updates are versioned aggressively.
   One request can bump state several times.

4. Vector indexing is explicitly asynchronous.
   The engine compensates with `delta_facts` and the vector watermark.

5. Fact consolidation is proposal-based.
   Background code suggests merges, but foreground code applies them.

6. Some repository code is not on the active runtime path.
   `modules/llm_client.py` and `Instruments/memory_viewer.py` are useful support code, but they do not participate in the main request pipeline.

## 18. Short end-to-end summary

For review purposes, the most accurate single-sentence description of the repository is:

`main.py` feeds user text into `memory_engine.loop.ingest_event()`, which persists the event, builds mixed SQLite/Chroma context, asks the LLM for structured plan steps, executes those steps into durable state changes, and relies on background workers to maintain embeddings and fact consolidation over time.
