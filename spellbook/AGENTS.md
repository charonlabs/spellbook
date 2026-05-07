# AGENTS.md

This file is for humans and agents changing `spellbook/`.

The old root-level guidance still contains useful product philosophy, but this file is the contributor guide for the **core rewrite as it exists in code now**.

The code is source of truth. Design targets under `spellbook/design/` are useful context, but they drift. When a document and the implementation disagree, trust the implementation.

---

## What `spellbook/` is

`spellbook/` is the runtime spine for the rewrite. It currently includes:

- canonical IR types
- the inner generate/execute loop
- session orchestration
- transcript recording + rehydration
- request-surface construction
- Homunculus-owned awareness subsystems
- semantic block management and rendering
- fork/subsession-backed background awareness work

This is no longer a prototype sketch. Prefer durable seams over shortcuts.

---

## Core design stance

A few ideas are load-bearing here:

- **Transcript is canonical.**
- **IR is the shared language.**
- **Homunculus is the awareness seam.**
- **SessionManager is orchestration, not policy.**
- **Forking is becoming reusable substrate, not one-off feature glue.**
- **Nursery jobs are async work, not transcript truth.**
- **Derived buffers should be derived, not canonical.**
- **Self-state lives in Homunculus; mechanisms stay outside.**

If you are unsure where something belongs, ask:

1. Is this transcript truth?
2. Is this session orchestration?
3. Is this awareness / semantic state?
4. Is this provider-facing rendering?
5. Is this a generic fork/subsession concern?

Most mistakes in the rewrite come from collapsing those layers together.

---

## The key objects

### `IR*` types in `ir_types.py`

These are the canonical internal language of the core rewrite.

Important rules:

- `IRBlock` is the conversation/event-level language
- `IRRecord` is the transcript persistence language
- do not introduce parallel ad hoc shapes when an IR type should exist
- if you add a persisted record kind, update:
  - `IRRecord`
  - recorder behavior
  - rehydrator behavior
  - tests

### `SessionManager`

`SessionManager` owns the outer session state machine:

- idle/running/suspended transitions
- inbound queue handling
- session lifecycle hooks
- build-time dependency composition
- invoking the inner loop with the right collaborators

Keep it relatively thin.

It is the right place for:

- composition
- startup / resume wiring
- queue semantics

It is **not** the right place for deep awareness policy.

### `run_loop`

`run_loop()` is the inner round loop.

It alternates:

- generate
- execute
- lifecycle hooks

Keep its contract narrow and stable. It should not learn session-specific policy.

### `Homunculus`

This is the active awareness seam.

It currently owns or coordinates:

- rehydrated block state
- global block-id continuity over the active block stream
- footer-aware context rendering
- gas-gauge / telemetry integration
- block detector integration
- fork-backed awareness work

If you are adding context intelligence, semantic grouping, derived awareness, or “active self” behavior, it probably belongs here or under `spellbook/homunculus/`.

### `BlockManager`

`BlockManager` is the Homunculus-owned coordinator for semantic memory.

It owns or coordinates:

- semantic block state
- context block rendering by block mode
- block detector integration
- block summarization
- semantic block metrics
- nursery harvesting for block-related background jobs

Important rule: a semantic block becoming real, a summary becoming available,
or a token count becoming known should be integrated by `BlockManager` at an
explicit lifecycle/script boundary. Background jobs may compute these things,
but they should not mutate canonical state on completion by themselves.

### `Recorder`

`Recorder` is the write side of transcript truth.

It owns:

- session record creation
- turn/seq stamping
- block record writes
- footer queue/drain records

Do not hide transcript mutations elsewhere if they should be explicit records.

### `Rehydrator`

`Rehydrator` is the read side of transcript truth.

It reconstructs:

- blocks
- config
- tools
- pending footers
- unfinished-turn state

Be conservative here. Rehydration should fail loudly on malformed or inconsistent transcript data.

### `RequestSurfaceBuilder`

This is the provider-request assembly seam.

It combines:

- current config
- system prompt provider
- tool schemas
- IR blocks

into a backend-specific request surface.

Do not push provider-specific rendering logic upward into unrelated layers.

### `ForkRunner`

`ForkRunner` is the reusable fork/subsession substrate.

Today it supports block-detector and block-summarizer forks.

Treat it as a generic seam for:

- derived session execution
- parent-config inheritance
- fork-scoped config/result protocols
- preparing child-session work for nursery scheduling
- recording fork summon/shutdown through the parent recorder

Do not bake block-detector-only assumptions too deeply into the fork infrastructure if they should become generic.

### `Nursery`

`Nursery` is the async job manager for background work.

Its core invariant is: background jobs never silently mutate canonical state
when they finish. A finished job only makes a result available. The owning
boundary, currently usually `BlockManager.check_nursery()`, decides whether to
integrate, record, queue footers, or discard the result as stale.

Use keyed jobs when there should be at most one in-flight job for a target,
for example `summary:<block_id>` or `metrics:<block_id>`.

Use `best_effort` for live awareness work that should keep chat snappy. Current
block detection, summarization, and semantic block metrics are best-effort.
Only use `render_blocking` for work that must affect the very next provider
render.

---

## Canonical invariants

### 1. Transcript first

`transcript.jsonl` is canonical.

Provider request messages, session renderings, and detector payloads are projections.

Do not add a second co-equal history source.

### 2. Record explicit runtime events instead of mutating history

If something is truly part of runtime truth and matters on replay, prefer an explicit record kind.

Examples already in code include footer queue/drain records.

### 3. Rehydration must preserve what the recorder meant

If you add a new recorded behavior, make sure `Rehydrator` reconstructs it faithfully.

Do not silently “fix” ambiguous transcript state.

### 4. Homunculus owns awareness state, not SessionManager

If logic answers questions like:

- what semantic context is active?
- what should be grouped?
- what should be surfaced to the model?
- when should a derived analysis run?

that is Homunculus territory.

Self-state subsystems belong inside Homunculus so generation/execution updates
happen in one explicit order. Current examples include token metering, gas
gauge pressure state, and block detection. Mechanism subsystems stay outside
Homunculus and are passed in or referenced as services: footer delivery,
recording, and fork/session spawning are mechanisms, not part of the active
self-state.

### 5. Fork config/result types are protocol, not random kwargs

If a forked subsystem needs structured input/output, define it as a real type.

This should stay explicit and typed.

### 6. Derived buffers should be recomputable

If a structure can be re-derived from more canonical state, prefer that over mutating the only copy.

The block detector is the current example: the context buffer is derived from canonical source context and semantic ranges.

### 7. Global block ids must stay coherent

Homunculus currently assigns monotonic block ids over the active in-memory block stream and threads them into awareness subsystems that need stable coordinates.

If you touch that logic, keep the coordinate system coherent across:

- rehydrate
- inbound render context
- generation integration
- execution integration
- detector context slicing

### 8. Background work integrates at boundaries

Nursery jobs may run concurrently, finish out of order, fail, or become stale.

When integrating a nursery result:

- validate the target still exists
- validate stable ids, not only indexes
- record explicit transcript truth only after validation
- discard stale results without mutating semantic state
- record fork shutdown when a fork result is integrated, discarded, cancelled, or failed

For semantic blocks, this means summaries and metrics are later facts about an
already-recorded block. They should be persisted as explicit records and
rehydrated onto the block, not hidden inside an unrecorded cache update.

---

## Current seams worth preserving

### Session build-time composition

`SessionManager.build()` currently wires together the runtime world:

- backend
- request surface builder
- token counter
- recorder
- inbound queue
- awareness subsystems
- fork runner
- nursery
- generator / executor
- round lifecycle composition

This is a real seam. Prefer adding services here rather than hiding construction in arbitrary runtime code.

### Tool metadata for forked sessions

`Executor` builds mutable tool metadata from:

- `SpellbookConfig`
- optional `ForkConfig`

That metadata is how tools get fork-scoped working state.

If you add a new fork-scoped tool surface, update metadata deliberately.

### Nursery-backed awareness work

The common shape is:

1. create or update immediate Homunculus/BlockManager state only when that state is already true
2. submit best-effort background work to `Nursery`
3. harvest with `check_nursery()` at a round, script, or replay boundary
4. validate by stable ids before mutating state
5. record the explicit IR record that makes the integration replayable

Do not record from inside the background task. Tasks should return typed
results; the owner that harvests the result records transcript truth.

Replay and manual scripts may use `wait_for_all=True` to drain background work
before reporting artifacts. Live chat paths should prefer opportunistic
harvesting so background awareness work does not create periodic pauses.

### Provider independence above the backend seam

`Generator`, `Executor`, `SessionManager`, `Homunculus`, and most core services should speak IR and config, not provider-specific request/response shapes.

Keep provider logic inside backends and request-surface building.

---

## Things to be careful with

### Circular imports

The core rewrite has enough typed seams that circular imports are easy to create.

Prefer:

- local imports for runtime-only dependency checks
- `TYPE_CHECKING` imports for type-only references
- forward references where needed

Do not pull heavy runtime objects into module top levels if they are only needed for annotations.

### Runtime assertions that need real runtime types

If code does `isinstance(...)` at runtime, the type must actually exist at runtime, not just under type-only imports.

### Tool/fork protocol drift

If you change a fork config, tool metadata shape, or fork result shape, update all of:

- config type
- runtime builder
- tool metadata builder
- fork runner
- nursery integration
- tests

### Nursery job drift

If you add a new nursery job kind, update all of:

- `NurseryJobKind`
- the submitter metadata shape
- the result integration path
- stale/discard behavior
- any transcript records needed for replay
- focused tests for success, stale target, and shutdown/cancellation if applicable

### Structured prompt renderers

Some subsystems render structured XML-ish text for model-facing use. Treat those as protocol surfaces, not throwaway strings.

If you change their structure:

- preserve parseability
- escape user/model content
- update tests
- keep the contract legible

Implementation-local guidance for these renderers should live in the module docstrings near the code.

---

## Testing expectations

At minimum, changes in `spellbook/` should preserve or extend focused coverage in `tests/`.

Before committing core source changes, run:

```bash
uv run ruff check spellbook tests
uv run ruff format --check spellbook tests
uv run ty check spellbook tests
```

Current useful areas include:

- session manager
- loop
- recorder
- rehydrator
- surface builder
- footer controller
- gas gauge
- block detector
- block manager
- nursery
- fork runner
- replay scripts when background work affects artifacts
- IR type unions / record discrimination

When adding new behavior, prefer tests that validate:

- typed protocol shape
- transcript truth
- explicit lifecycle behavior
- invariants over mutable state

A failing test that reveals a real implementation gap is more valuable than a passing test that only mirrors a broken implementation.

---

## Practical heuristics

When in doubt, prefer:

- explicit typed state over hidden coupling
- durable seams over convenience
- replayable truth over implicit runtime magic
- derived projections over duplicated mutable state
- Homunculus-centered awareness logic over SessionManager sprawl

If a change makes the system more powerful but harder to explain in terms of:

- IR
- transcript
- lifecycle
- fork protocol
- Homunculus awareness

then it probably needs a cleaner seam first.
