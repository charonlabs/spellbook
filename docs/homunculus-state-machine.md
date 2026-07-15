# Homunculus state machine: the complete map

Code reference: `dev` at `7f4ea63` plus the pre-existing working-tree edits,
traced 2026-07-15. The transcript and current implementation are authoritative;
future Sleep edges are explicitly labeled as design, not runtime behavior.

The companion interactive map is
[`homunculus-state-machine.html`](homunculus-state-machine.html). It contains the
same state inventory, transition ledger, failure states, findings, and three
walkthroughs, with filtering and step highlighting.

## Reading the machine

The map combines four coordinated state machines without pretending that they
are one object:

1. `SessionManager` owns the outer phases: constructed/suspended, idle, running,
   and shutdown. `dreaming` is already admitted by the type but is not entered by
   current runtime code. [`session_manager.py:58`](../spellbook/session_manager.py#L58),
   [`session_manager.py:122-143`](../spellbook/session_manager.py#L122)
2. `run_loop` owns the round state: before-round, generate, after-generate,
   execute, after-execute, between-rounds, and loop exit.
   [`loop.py:29-105`](../spellbook/loop.py#L29)
3. `Homunculus` and `BlockManager` own awareness and memory projections: global
   context-block coordinates, semantic blocks, summaries, modes, pins, token
   pressure, TTLs, and nursery integration.
   [`homunculus.py:49-100`](../spellbook/homunculus/homunculus.py#L49),
   [`block_manager.py:81-120`](../spellbook/homunculus/block_manager.py#L81)
4. `CoreAppRuntime` owns transport-facing activity: messages, conduits, hearth,
   interrupts, process shutdown, and app event publication.
   [`runtime.py:47-110`](../spellbook/app/runtime.py#L47),
   [`runtime.py:125-320`](../spellbook/app/runtime.py#L125)

The canonical-state boundary is the transcript. Recorder writes are truth;
provider context, detector buffers, TTL-collapsed tool results, and the frontier
are projections. Recorder stamps explicit turn/sequence records, while
`Rehydrator` reconstructs blocks, semantic memory, proposals, pending footers,
runtime config, TTLs, and unfinished turns.
[`recorder.py:93-109`](../spellbook/recorder.py#L93),
[`recorder.py:188-234`](../spellbook/recorder.py#L188),
[`rehydrator.py:112-139`](../spellbook/rehydrator.py#L112),
[`rehydrator.py:178-315`](../spellbook/rehydrator.py#L178)

## 1. Outer session states and transitions

### S0 — Build or resume

`SessionManager.build()` has two entrances. A new transcript requires config,
discovers skills when the profile permits it, builds the tool registry, and
writes the initial session and refusal-policy records. A resume skips creation.
Both paths then run the same strict rehydrator and rebuild runtime collaborators.
[`session_manager.py:228-307`](../spellbook/session_manager.py#L228)

Transition `S0 -> S1 Rehydrated world`: `Rehydrator.run()` reads every nonblank
line through the typed `IRRecord` adapter and fails on malformed data; it does not
silently repair the transcript.
[`rehydrator.py:82-114`](../spellbook/rehydrator.py#L82)

### S1 — Rehydrated world

Rehydration reconstructs the canonical block stream, the last completed or
unfinished turn, semantic range/block/artifact/mode/pin state, plan proposal,
pending footers, skill catalog, TTL records, runtime config, and system
responses. Apply-mode and pin records also invalidate a rehydrated planner
proposal.
[`rehydrator.py:120-138`](../spellbook/rehydrator.py#L120),
[`rehydrator.py:185-273`](../spellbook/rehydrator.py#L185),
[`rehydrator.py:290-315`](../spellbook/rehydrator.py#L290)

Transition `S1 -> S2 Constructed/suspended`: build wires the backend, request
surface, token counter, recorder, inbound queue, footer controller, optional
timekeeper, fork runner, nursery, Homunculus, round lifecycle, generator, and
executor. Homunculus rehydrates before the manager is returned.
[`session_manager.py:332-415`](../spellbook/session_manager.py#L332),
[`session_manager.py:421-451`](../spellbook/session_manager.py#L421)

### S2 — Constructed / suspended

A newly constructed manager starts as `suspended`; this means “not running the
outer loop yet,” not “transcript closed.”
[`session_manager.py:105-126`](../spellbook/session_manager.py#L105)

Transition `S2 -> S3 Idle`: `run()` enters `_idle_phase()` while shutdown has not
been requested.
[`session_manager.py:134-150`](../spellbook/session_manager.py#L134)

### S3 — Idle

On entry the manager publishes `on_enter_idle`, then waits on the inbound queue
for a turn-eligible `turn` or `inject` message. Footer-only messages remain
queued. Slash commands are handled and recorded without starting a model turn.
[`session_manager.py:148-160`](../spellbook/session_manager.py#L148),
[`inbound.py:15-16`](../spellbook/inbound.py#L15),
[`inbound.py:60-76`](../spellbook/inbound.py#L60),
[`session_manager.py:196-211`](../spellbook/session_manager.py#L196)

Transition `S3 -> S4 Running`: a non-slash message causes `on_exit_idle(reason=
"message")`, is pushed back to the head of the queue, and is consumed by the
running phase.
[`session_manager.py:151-164`](../spellbook/session_manager.py#L151)

Transition `S3 -> S14 Shutdown`: queue shutdown makes `take_turn()` return `None`;
idle exits with `reason="shutdown"`.
[`inbound.py:60-81`](../spellbook/inbound.py#L60),
[`session_manager.py:145-155`](../spellbook/session_manager.py#L145)

### S4 — Running / turn setup

For every pending turn, the manager creates a UUID turn id, records turn start
and inbound blocks, fires `on_turn_started`, creates a new cancel token, asks the
Homunculus to render canonical memory plus the new inbound blocks, and invokes
`run_loop`.
[`session_manager.py:162-184`](../spellbook/session_manager.py#L162),
[`recorder.py:188-198`](../spellbook/recorder.py#L188),
[`homunculus.py:153-161`](../spellbook/homunculus/homunculus.py#L153)

Transition `S4 -> R0 Before round`: `run_loop` copies the rendered initial blocks,
increments the round number, clears `blocks_this_round`, and calls
`before_round`.
[`loop.py:39-53`](../spellbook/loop.py#L39)

Transition `R6 Loop exit -> S4 next queued turn`: after a loop result, the manager
records `turn_end`, clears the cancel token, fires `on_turn_ended`, and continues
while turn-eligible messages remain.
[`session_manager.py:184-187`](../spellbook/session_manager.py#L184),
[`recorder.py:199-206`](../spellbook/recorder.py#L199)

Transition `S4 -> S3 Idle`: when no pending turn remains, `_running_phase()`
returns and the outer loop enters idle again.
[`session_manager.py:134-140`](../spellbook/session_manager.py#L134),
[`session_manager.py:162-187`](../spellbook/session_manager.py#L162)

### S5 — Dreaming (reserved, not entered)

`dreaming` is present in both `SessionState` and app `RuntimeState`, but no current
assignment sets `session.state = "dreaming"`. It is a reserved runtime state, not
an implemented transition.
[`session_manager.py:58`](../spellbook/session_manager.py#L58),
[`protocol.py:36-52`](../spellbook/app/protocol.py#L36)

The future attachment is documented in [§12](#12-future-sleep-attachment-points).

### S14 — Shutdown / suspended

`shutdown()` marks shutdown requested, cancels an active turn token, and wakes the
idle queue. The outer loop then sets `suspended`, cancels and harvests Homunculus
nursery jobs, closes debug delivery, and fires `on_shutdown`.
[`session_manager.py:213-217`](../spellbook/session_manager.py#L213),
[`session_manager.py:134-143`](../spellbook/session_manager.py#L134),
[`block_manager.py:1116-1120`](../spellbook/homunculus/block_manager.py#L1116)

The app runtime first stops the hearth, asks the manager to shut down, awaits the
session task, closes subscriptions, and marks shutdown complete. The `/shutdown`
route then requests process `SIGTERM` in a background task.
[`runtime.py:505-533`](../spellbook/app/runtime.py#L505),
[`server.py:249-259`](../spellbook/app/server.py#L249),
[`server.py:71-72`](../spellbook/app/server.py#L71)

## 2. SessionProfile gates: the behavior seam

`session_type` is the persisted alias; `SessionProfile` supplies orthogonal
runtime flags. New behavior is expected to gate here rather than branch on a new
master enum.
[`profiles.py:1-5`](../spellbook/profiles.py#L1),
[`profiles.py:34-62`](../spellbook/profiles.py#L34)

| Profile | Homunculus hooks | Detection | Skills | Mid-turn injection | Ambient time | Hearth | Conduits | Provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `MAIN` | yes | yes | yes | yes | yes | yes | yes | [`profiles.py:65-76`](../spellbook/profiles.py#L65) |
| `CUSTOM` | yes | yes | conditional | no | no | no | yes | [`profiles.py:78-89`](../spellbook/profiles.py#L78) |
| `BLOCK_DETECTOR` | no | no | no | no | no | no | no | [`profiles.py:91-103`](../spellbook/profiles.py#L91) |
| `BLOCK_SUMMARIZER` | no | no | no | no | no | no | no | [`profiles.py:105-117`](../spellbook/profiles.py#L105) |
| `QUANTUM` | yes | no | no | no | no | no | no | [`profiles.py:119-131`](../spellbook/profiles.py#L119) |

Build consumes these gates when composing timekeeper, Homunculus detection,
skills, injection, footer delivery, and the hearth scheduler.
[`session_manager.py:370-385`](../spellbook/session_manager.py#L370),
[`session_manager.py:402-459`](../spellbook/session_manager.py#L402),
[`runtime.py:105-110`](../spellbook/app/runtime.py#L105)

## 3. One round, hook by hook

The composite lifecycle executes subscribers serially in list order. The app
adds its event lifecycle at the front; build then supplies recording,
Homunculus, optional skills, optional inbound injection, optional timekeeper,
and footer delivery; callers may add a post lifecycle.
[`round_lifecycle.py:66-108`](../spellbook/round_lifecycle.py#L66),
[`session_manager.py:421-459`](../spellbook/session_manager.py#L421),
[`runtime.py:93-103`](../spellbook/app/runtime.py#L93)

### R0 — `before_round`

Order and effects:

1. App lifecycle has no `before_round` behavior.
2. Recording and Homunculus lifecycles have no `before_round` behavior.
3. Skill manager refreshes the catalog, records a delta, and queues a footer if
   files changed. [`manager.py:176-203`](../spellbook/skills/manager.py#L176)
4. Main-profile inbound injection drains all `delivery="inject"` messages into
   `ctx.blocks` and `ctx.blocks_this_round`, recording each block.
   [`inbound.py:84-94`](../spellbook/inbound.py#L84)
5. Timekeeper observes hour/date rollover and queues any time footer.
   [`timekeeper.py:216-221`](../spellbook/timekeeper.py#L216),
   [`timekeeper.py:60-65`](../spellbook/timekeeper.py#L60)
6. Footer lifecycle drains inbound footer messages, applies keyed
   last-write-wins dedup and priority order, records the drain, renders one
   system-origin `<spellbook>` block, appends it to `ctx.blocks`, and records the
   block. [`footer.py:53-124`](../spellbook/footer.py#L53),
   [`footer.py:164-185`](../spellbook/footer.py#L164)

Transition `R0 -> R1 Generate`: the generator builds the provider surface once,
then enters the streaming state.
[`generator.py:67-91`](../spellbook/generator.py#L67)

### R1 — Generate / stream

The generator races each next stream event against cancellation. Stream events
are published through `on_stream_event`. Cancellation returns the backend's
partial current response with `stop_reason="cancelled"`; stream exhaustion
returns the final response.
[`generator.py:126-170`](../spellbook/generator.py#L126)

Transition `R1 -> R2 After generate`: generated blocks extend both the full loop
stream and `blocks_this_round`, then every lifecycle sees the generation.
[`loop.py:53-59`](../spellbook/loop.py#L53)

### R2 — `after_generate`

App publishes block-added events; Recorder persists each generation block;
Homunculus appends the same blocks to the canonical in-memory context, assigns
their global ids, feeds token observations and detector accumulation, and sends
input usage to the gas gauge.
[`lifecycle.py:35-42`](../spellbook/app/lifecycle.py#L35),
[`recorder.py:343-350`](../spellbook/recorder.py#L343),
[`homunculus.py:499-505`](../spellbook/homunculus/homunculus.py#L499),
[`block_manager.py:135-151`](../spellbook/homunculus/block_manager.py#L135)

Transition `R2 -> R6 Loop exit`: any generation stop reason other than
`tool_use` exits immediately after `on_loop_exit`.
[`loop.py:60-68`](../spellbook/loop.py#L60)

Transition `R2 -> R3 Execute`: `tool_use` dispatches the generated tool-call
blocks in declared order.
[`loop.py:60-73`](../spellbook/loop.py#L60)

### R3 — Execute

Executor validates each call and races the tool task against cancellation. On
its handled path it emits one result block per declared call: unknown tools,
invalid input, and `ToolError` become error results rather than loop exceptions.
An unexpected tool exception escapes; a terminal tool result may set a loop stop
reason.
[`executor.py:61-137`](../spellbook/executor.py#L61)

Transition `R3 -> R4 After execute`: result blocks extend the loop stream and
`blocks_this_round`; every lifecycle observes the execution.
[`loop.py:70-75`](../spellbook/loop.py#L70)

### R4 — `after_execute`

App publishes result-block events; Recorder persists them; Homunculus assigns
global ids, feeds detector accumulation, and auto-registers eligible large text
or image tool results for TTL collapse.
[`lifecycle.py:43-49`](../spellbook/app/lifecycle.py#L43),
[`recorder.py:351-357`](../spellbook/recorder.py#L351),
[`homunculus.py:507-510`](../spellbook/homunculus/homunculus.py#L507),
[`tool_result_ttl.py:263-267`](../spellbook/homunculus/tool_result_ttl.py#L263)

Transition `R4 -> R6 Loop exit`: cancellation or a terminal tool stop reason
exits after `on_loop_exit`; otherwise the round reaches `between_rounds`.
[`loop.py:77-96`](../spellbook/loop.py#L77)

### R5 — `between_rounds`

Homunculus performs, in order: opportunistic nursery harvest, round-triggered
TTL tick, planner check, and conditional full rerender after any context-changing
mode/TTL operation.
[`homunculus.py:919-925`](../spellbook/homunculus/homunculus.py#L919)

Transition `R5 -> R0`: the loop checks cancellation at the next `while` guard,
increments the round, and begins `before_round` again.
[`loop.py:48-52`](../spellbook/loop.py#L48),
[`loop.py:96-98`](../spellbook/loop.py#L96)

### R6 — `on_loop_exit`

Homunculus harvests the nursery. Only `end_turn` ticks end-turn TTLs; cancellation,
refusal, `max_tokens`, and other non-tool stop reasons do not.
[`homunculus.py:927-930`](../spellbook/homunculus/homunculus.py#L927)

The manager then records the turn end and fires session-level turn-ended hooks.
[`session_manager.py:184-187`](../spellbook/session_manager.py#L184)

## 4. Context coordinates and semantic detection

### C0 — Canonical context append

`append_context_blocks` returns the batch start id, extends `context_blocks`, and
increments `next_block_id` monotonically. Generation usage records an observed
input prefix at the pre-generation boundary and an approximate total at the
generation end. Every append then enters detector accumulation.
[`block_manager.py:135-151`](../spellbook/homunculus/block_manager.py#L135),
[`token_meter.py:53-76`](../spellbook/homunculus/token_meter.py#L53)

On rehydrate, the Homunculus seeds `context_blocks` from transcript blocks and
sets `next_block_id = len(context_blocks)` before rehydrating the detector and
semantic blocks.
[`homunculus.py:105-120`](../spellbook/homunculus/homunculus.py#L105)

Transition `C0 -> C1 Accumulating`: the detector extends its canonical
`_accumulated` source stream and increments a block counter.
[`block_detector.py:108-119`](../spellbook/homunculus/block_detector.py#L108)

### C1 — Detector accumulating

The detector keeps completed semantic ranges, the latest semantic draft buffer,
the complete accumulated source slice, and a derived raw-context suffix. The
buffer is recomputed from the ranges; it is not a second history.
[`block_detector.py:1-29`](../spellbook/homunculus/block_detector.py#L1),
[`block_detector.py:484-505`](../spellbook/homunculus/block_detector.py#L484)

Transition `C1 -> C2 Detection prepared`: once the counter reaches the configured
interval (default 500 context blocks), the detector derives its buffer, subtracts
one interval, builds the structured prompt, and asks `ForkRunner` for a
`PreparedFork`.
[`config.py:32-38`](../spellbook/config.py#L32),
[`block_detector.py:117-148`](../spellbook/homunculus/block_detector.py#L117)

`force_detect(finalize=...)` takes the same edge immediately when buffered or raw
context remains.
[`block_detector.py:125-131`](../spellbook/homunculus/block_detector.py#L125)

### C2 — Detection fork in Nursery

BlockManager submits the prepared coroutine as best-effort `detect_blocks` work.
Completion only puts the job id into the ready queue; it cannot mutate memory.
[`block_manager.py:729-743`](../spellbook/homunculus/block_manager.py#L729),
[`nursery.py:115-165`](../spellbook/nursery.py#L115)

Transition `C2 -> C3 Harvest`: `check_nursery()` runs between rounds, on loop
exit, in explicit replay drains, and during shutdown. It collects only
`source="block_manager"` jobs and dispatches by job kind.
[`homunculus.py:518-522`](../spellbook/homunculus/homunculus.py#L518),
[`block_manager.py:1100-1120`](../spellbook/homunculus/block_manager.py#L1100),
[`block_manager.py:1132-1166`](../spellbook/homunculus/block_manager.py#L1132)

### C3 — Simulate and sanitize

Before any record or state mutation, `simulate_result` sorts ranges, checks that
all lie within accumulated context, closes or buffers tool pairs, accepts only a
contiguous completed prefix, carries a contiguous safe draft buffer, defers
later completed ranges, and discards ranges after holes.
[`block_detector.py:156-190`](../spellbook/homunculus/block_detector.py#L156),
[`block_detector.py:222-288`](../spellbook/homunculus/block_detector.py#L222),
[`block_detector.py:304-383`](../spellbook/homunculus/block_detector.py#L304)

Transition `C3 -> C4 Validate`: BlockManager constructs candidate semantic blocks
and validates the entire existing-plus-new prefix.
[`block_manager.py:539-554`](../spellbook/homunculus/block_manager.py#L539)

### C4 — Validate prefix and tool closure

Validation requires ordered gapless semantic indices, a gapless prefix of
context coordinates, ranges within known context, and—during new detection
integration—no semantic boundary that tears a tool-call/result envelope.
[`block_manager.py:1308-1339`](../spellbook/homunculus/block_manager.py#L1308),
[`block_manager.py:1341-1435`](../spellbook/homunculus/block_manager.py#L1341)

Transition `C4 -> C5 Crystallize`: only after successful validation does the
detector record sanitized ranges and mutate detector state. BlockManager then
appends and records each semantic block, queues a courtesy footer, schedules
stable-id-keyed metrics, and starts the summary cascade.
[`block_detector.py:192-200`](../spellbook/homunculus/block_detector.py#L192),
[`block_manager.py:602-643`](../spellbook/homunculus/block_manager.py#L602)

Transition `C4 -> F2 Graceful discard`: `ValueError` rejects the entire result,
records fork shutdown without a detection record, emits a debug alert, and
queues a courtesy footer; current semantic memory is unchanged.
[`block_manager.py:539-557`](../spellbook/homunculus/block_manager.py#L539),
[`block_manager.py:661-692`](../spellbook/homunculus/block_manager.py#L661),
[`block_detector.py:202-203`](../spellbook/homunculus/block_detector.py#L202)

Partial sanitization is an accepted-but-degraded state: contiguous work lands,
deferred/discarded ranges are disclosed in logs/debug and a keyed courtesy
footer.
[`block_manager.py:573-600`](../spellbook/homunculus/block_manager.py#L573)

### C5 — Semantic block born

A new block begins in `full` mode with no summary or token metrics. Metrics and
summaries are later facts, integrated only at nursery boundaries.
[`block_manager.py:645-659`](../spellbook/homunculus/block_manager.py#L645),
[`ir_types.py:574-589`](../spellbook/ir_types.py#L574)

Transition `C5 -> C6 Metrics known`: the metrics job counts a direct block slice;
integration verifies both index and stable block id before recording metrics and
updating `full_toks`/current `toks`.
[`block_manager.py:627-641`](../spellbook/homunculus/block_manager.py#L627),
[`block_manager.py:694-708`](../spellbook/homunculus/block_manager.py#L694),
[`token_meter.py:111-127`](../spellbook/homunculus/token_meter.py#L111)

### C7 — Summary cascade

`generate_next_summary()` scans from oldest to newest and schedules only the
first block without a ready summary, carrying at most five previous summarized
blocks as context. Stable job key `summary:<block-id>` prevents duplicate work.
[`block_manager.py:836-894`](../spellbook/homunculus/block_manager.py#L836)

Transition `C7 -> C8 Summary ready`: integration records fork shutdown first,
validates block index, stable id, and absence of an existing summary, adds the
artifact and `summary` mode, counts the short rendering, records the artifact,
and recursively schedules the next missing summary.
[`block_manager.py:745-834`](../spellbook/homunculus/block_manager.py#L745)

Transition `M0 Full without summary -> C7 On-demand bake`: Forget on a block whose
summary is missing either reports an already in-flight job, queues that block
immediately using up to five previous summaries, or reports summary work
unavailable for a profile that disables forks.
[`block_manager.py:914-953`](../spellbook/homunculus/block_manager.py#L914)

## 5. The memory surface

The three persisted semantic render states are `full`, `summary`, and
`pair_narrative`.
[`ir_types.py:453`](../spellbook/ir_types.py#L453)

### M0 — Full

Full mode projects the original canonical context-block slice.
[`block_manager.py:179-188`](../spellbook/homunculus/block_manager.py#L179),
[`block_manager.py:230-231`](../spellbook/homunculus/block_manager.py#L230)

### M1 — Summary

Summary mode projects a labeled `<spellbook-memory>` user block. Facet pins
replace selected parts with expanded original conversation, expanding same-role
edges and tool pairs so provider sequences remain valid.
[`common.py:78-106`](../spellbook/homunculus/common.py#L78),
[`block_manager.py:233-251`](../spellbook/homunculus/block_manager.py#L233),
[`block_manager.py:385-477`](../spellbook/homunculus/block_manager.py#L385)

Transition `M0 -> M1 Forget`: Forget refuses an unconfirmed block pin, refuses a
torn tool boundary, queues a missing summary, or applies `summary` mode. A
successful transition writes an append-only apply-mode record with source
`model` or `planner`, changes rendered tokens, and invalidates the Homunculus
render/count/gauge/planner caches.
[`block_manager.py:955-985`](../spellbook/homunculus/block_manager.py#L955),
[`block_manager.py:1251-1290`](../spellbook/homunculus/block_manager.py#L1251),
[`homunculus.py:326-335`](../spellbook/homunculus/homunculus.py#L326)

### M2/M3 — Pair narrative parent and child

Pair artifacts are an adjacent even/odd block pair. The parent owns rendered
narrative blocks; the child carries zero tokens and a parent pointer.
[`ir_types.py:483-528`](../spellbook/ir_types.py#L483),
[`ir_types.py:531-541`](../spellbook/ir_types.py#L531)

In `pair_narrative` mode the parent renders its artifact and the child renders
nothing. Rendering validates that both blocks are in the same mode and carry the
same narrative id.
[`block_manager.py:262-328`](../spellbook/homunculus/block_manager.py#L262)

Forget and Pin refuse to operate on either half. Recall redirects from a block
target to the chapter target, and chapter Recall returns both original source
blocks.
[`block_manager.py:1028-1080`](../spellbook/homunculus/block_manager.py#L1028),
[`block_manager.py:1082-1091`](../spellbook/homunculus/block_manager.py#L1082)

### Reflect

Reflect without a block target shows gas regime, frame tokens, pending planner
proposal, semantic ranges/modes/pins/facets, and the unblocked tail. Reflect with
a block target previews the block's summary rendering without applying it.
[`homunculus.py:163-254`](../spellbook/homunculus/homunculus.py#L163),
[`block_manager.py:192-223`](../spellbook/homunculus/block_manager.py#L192)

`ReflectToolResults` separately reports TTL status and sizes, defaulting to
pending and large-untracked results.
[`homunculus.py:256-324`](../spellbook/homunculus/homunculus.py#L256)

### Pin

Transition `M1 -> M0 Pin block`: a whole-block pin first applies `full` mode if
needed, writes a pin record, and protects the block from planner selection.
[`block_manager.py:989-1003`](../spellbook/homunculus/block_manager.py#L989),
[`planner.py:25-34`](../spellbook/homunculus/planner.py#L25)

Transition `M1 -> M1+facet Pin facet`: a facet pin records the selected facet;
summary rendering changes to include its expanded original conversation, so the
render is invalidated while mode remains summary.
[`block_manager.py:1005-1026`](../spellbook/homunculus/block_manager.py#L1005),
[`block_manager.py:233-251`](../spellbook/homunculus/block_manager.py#L233)

### Recall

Recall is a tool-result projection, not a persistent mode transition. A compacted
block returns its original source slice inside the tool result; full blocks
refuse because they are already in context.
[`block_manager.py:1028-1046`](../spellbook/homunculus/block_manager.py#L1028),
[`self_work.py:198-236`](../spellbook/tools/self_work.py#L198)

### Configure

Configure reads or writes the `tool_result_ttl` and `hearth` tenants. Valid
updates mutate live settings and write replayable `runtime_config` records;
rehydration reapplies them in transcript order.
[`homunculus.py:366-497`](../spellbook/homunculus/homunculus.py#L366),
[`homunculus.py:683-703`](../spellbook/homunculus/homunculus.py#L683),
[`homunculus.py:105-120`](../spellbook/homunculus/homunculus.py#L105)

## 6. Tool-result TTL projection

### T0 — Untracked result

After execution, enabled TTL policy auto-registers any non-error, non-skipped
tool result containing an image or text at/above the configured threshold
(default 4,000 characters). Registration saves the full output under
`tool-outputs/`, records replacement text, delivery turn, remaining TTL, and
trigger, then keeps the canonical tool result unchanged.
[`config.py:37-38`](../spellbook/config.py#L37),
[`tool_result_ttl.py:34-39`](../spellbook/homunculus/tool_result_ttl.py#L34),
[`tool_result_ttl.py:263-305`](../spellbook/homunculus/tool_result_ttl.py#L263),
[`tool_result_ttl.py:521-556`](../spellbook/homunculus/tool_result_ttl.py#L521)

Transition `T0 -> T1 Pending TTL`: auto-registration uses the configured
end-turn count (default three). Manual `ForgetToolResult` instead saves output
and registers TTL zero immediately.
[`tool_result_ttl.py:269-306`](../spellbook/homunculus/tool_result_ttl.py#L269),
[`tool_result_ttl.py:308-345`](../spellbook/homunculus/tool_result_ttl.py#L308)

Transition `T1 -> T1`: matching round (`seq`) or end-turn ticks decrement
remaining TTL. End-turn rehydration also subtracts every completed turn since
delivery, including the delivery turn itself.
[`tool_result_ttl.py:61-79`](../spellbook/homunculus/tool_result_ttl.py#L61),
[`tool_result_ttl.py:347-371`](../spellbook/homunculus/tool_result_ttl.py#L347),
[`homunculus.py:612-618`](../spellbook/homunculus/homunculus.py#L612)

Transition `T1 -> T2 Collapsed projection`: at remaining zero, render-time
projection replaces only the tool result content. Canonical transcript content
remains full; Homunculus invalidates the next render and token/gauge/planner
counts.
[`tool_result_ttl.py:169-174`](../spellbook/homunculus/tool_result_ttl.py#L169),
[`tool_result_ttl.py:511-519`](../spellbook/homunculus/tool_result_ttl.py#L511),
[`homunculus.py:612-618`](../spellbook/homunculus/homunculus.py#L612)

## 7. Token meter, gas gauge, and planner

### G0 — Token observations

Generation usage reaches two consumers: TokenMeter stores boundary prefix
observations, and GasGauge stores exact provider input tokens. Semantic block
metrics instead use direct slice counts and may fall back to repaired or chunked
approximation.
[`block_manager.py:135-149`](../spellbook/homunculus/block_manager.py#L135),
[`homunculus.py:499-505`](../spellbook/homunculus/homunculus.py#L499),
[`token_meter.py:111-217`](../spellbook/homunculus/token_meter.py#L111)

Transition `G0 -> G1 New 50K bucket`: the gauge emits only when
`input_tokens // 50_000` changes. The keyed `gas_gauge` footer replaces an older
pending gauge and has priority 10.
[`gas_gauge.py:28-63`](../spellbook/homunculus/gas_gauge.py#L28)

Regime transitions use configured numeric thresholds: calm below 700K, warning
at 700K, forced at 850K, critical at 933K by default, unknown while invalidated.
[`config.py:32-35`](../spellbook/config.py#L32),
[`common.py:60-69`](../spellbook/homunculus/common.py#L60)

Any memory/TTL projection change invalidates token prefixes, makes gauge input
unknown, clears the pending gauge footer, and invalidates the planner proposal.
The next provider generation observation establishes the new count and can show
the post-compaction drop.
[`homunculus.py:860-878`](../spellbook/homunculus/homunculus.py#L860),
[`gas_gauge.py:65-68`](../spellbook/homunculus/gas_gauge.py#L65)

### P0 — No proposal

Below the soft threshold, Planner does nothing. At or above soft pressure it
selects the oldest full, summary-ready, unpinned block.
[`planner.py:25-41`](../spellbook/homunculus/planner.py#L25),
[`planner.py:46-58`](../spellbook/homunculus/planner.py#L46)

Transition `P0 -> P1 Proposed`: below the medium threshold the new proposal is
recorded, exposed by Reflect, and announced through a compaction footer with the
distance to the forced threshold.
[`planner.py:51-63`](../spellbook/homunculus/planner.py#L51),
[`homunculus.py:524-573`](../spellbook/homunculus/homunculus.py#L524),
[`homunculus.py:603-610`](../spellbook/homunculus/homunculus.py#L603)

Two consumers exist:

- The entity may respond to the footer/Reflect proposal by calling Forget,
  producing a model-sourced apply-mode record.
  [`self_work.py:62-93`](../spellbook/tools/self_work.py#L62)
- If pressure reaches the medium threshold while the proposal is still valid,
  Planner itself calls Forget with source `planner`, records/applies the mode,
  and announces the action.
  [`planner.py:59-63`](../spellbook/homunculus/planner.py#L59),
  [`homunculus.py:574-610`](../spellbook/homunculus/homunculus.py#L574)

Transition `P1 -> P0 Invalidated`: any Forget, Pin, or TTL-driven render/count
change clears the in-memory proposal; rehydration likewise clears a recorded
proposal when it later sees a pin or apply-mode record.
[`homunculus.py:860-878`](../spellbook/homunculus/homunculus.py#L860),
[`rehydrator.py:246-273`](../spellbook/rehydrator.py#L246)

## 8. Forks and integrate-at-boundaries

The discriminated `ForkConfig` union has exactly three kinds:

| Kind | Input/result | Child shape | Provenance |
| --- | --- | --- | --- |
| `block_detector` | completed and still-buffered semantic ranges | New child transcript, detector orientation/tool metadata, inherited parent model by default | [`fork.py:70-80`](../spellbook/fork.py#L70), [`fork.py:201-279`](../spellbook/fork.py#L201) |
| `block_summarizer` | one summary artifact | New child transcript, summarizer orientation/tool metadata, inherited parent model (“memory should sound like you”) | [`fork.py:83-88`](../spellbook/fork.py#L83), [`fork.py:391-465`](../spellbook/fork.py#L391) |
| `quantum` | final text plus optional submitted JSON and durable transcript path | Snapshot of the same transcript and memory with a profile-controlled tool surface | [`fork.py:91-126`](../spellbook/fork.py#L91), [`fork.py:281-389`](../spellbook/fork.py#L281) |

Transition `Fork request -> PreparedFork`: `run_fork` dispatches by typed config
and returns a coroutine plus fork id. The caller chooses whether and how to
compose it in Nursery.
[`fork.py:135-138`](../spellbook/fork.py#L135),
[`fork.py:185-196`](../spellbook/fork.py#L185)

Transition `PreparedFork -> Nursery running`: BlockManager submits detector,
summarizer, and metric work as best-effort jobs. Nursery completion only marks a
result ready; boundary code performs validation, recording, mutation, footer
delivery, stale discard, and fork shutdown.
[`nursery.py:1-5`](../spellbook/nursery.py#L1),
[`nursery.py:115-165`](../spellbook/nursery.py#L115),
[`block_manager.py:1100-1166`](../spellbook/homunculus/block_manager.py#L1100)

Every detector/summarizer child gets a parent `summon_fork` record before work.
The corresponding `shutdown_fork` record is written when the parent integrates,
discards, observes cancellation/error, or shuts the nursery down.
[`recorder.py:306-327`](../spellbook/recorder.py#L306),
[`block_manager.py:1132-1158`](../spellbook/homunculus/block_manager.py#L1132)

### Quantum snapshot procedure

1. Assert the parent transcript has no unfinished turn.
   [`fork.py:508-535`](../spellbook/fork.py#L508)
2. Create `forks/<quantum-id>/`, copy the transcript, rewrite every record to
   the child session id and the session record to the quantum profile/tool
   surface, then symlink `blobs/` and `tool_outputs/`.
   [`fork.py:508-573`](../spellbook/fork.py#L508)
3. Record `summon_fork` in the parent, build the child from the copy, enqueue the
   sole instruction as a turn, and wait for either turn end or child task exit.
   [`fork.py:281-341`](../spellbook/fork.py#L281),
   [`fork.py:467-506`](../spellbook/fork.py#L467)
4. `SubmitResult` stores JSON in quantum tool metadata and ends the loop through
   a terminal `end_turn`; absent submission, final assistant/refusal text remains
   the result.
   [`quantum.py:13-30`](../spellbook/tools/quantum.py#L13),
   [`fork.py:342-359`](../spellbook/fork.py#L342),
   [`fork.py:733-740`](../spellbook/fork.py#L733)
5. The consumer applies nothing automatically. Current dreaming code awaits the
   prepared coroutine, validates result type, and explicitly writes fork
   shutdown; application is a separate consumer action.
   [`pipeline.py:393-411`](../spellbook/dreaming/pipeline.py#L393)

Fork build/run/cancellation failure preserves quantum transcripts and writes an
error-noted shutdown record; no result is fabricated.
[`fork.py:311-320`](../spellbook/fork.py#L311),
[`fork.py:367-387`](../spellbook/fork.py#L367),
[`fork.py:575-579`](../spellbook/fork.py#L575)

## 9. Hearth, inbound messages, and conduits

### H0 — Hearth waiting or skipped

The app creates `HearthScheduler` only when `profile.hearth` is true. The current
entity setting is disabled per card context, and the runtime default is also
false. Even when present, each tick skips if there is no session, settings are
disabled, local time is quiet, session is not idle, any inbound message is
pending, or idle duration is below the configured interval.
[`runtime.py:105-110`](../spellbook/app/runtime.py#L105),
[`config.py:45-47`](../spellbook/config.py#L45),
[`hearth.py:251-278`](../spellbook/hearth.py#L251)

Transition `H0 -> H1 Crackle composed`: an eligible tick reads Homunculus
awareness, composes the invariant heartbeat plus local hour, numeric gauge, and
optionally one facet from a summarized block, then builds a system-origin normal
turn with a dedup key.
[`hearth.py:149-196`](../spellbook/hearth.py#L149),
[`hearth.py:280-293`](../spellbook/hearth.py#L280),
[`hearth.py:361-391`](../spellbook/hearth.py#L361)

Transition `H1 -> S4 Running`: runtime rechecks under the command lock that the
session is still fully idle and queue-empty, then submits the crackle through
the normal inbound turn path. Crackle time is noted only after submission.
[`runtime.py:137-144`](../spellbook/app/runtime.py#L137),
[`hearth.py:290-293`](../spellbook/hearth.py#L290)

### I0 — Inbound delivery kinds

- `turn`: eligible to start or queue a new turn.
- `inject`: starts a turn while idle, but joins the next round while already
  running.
- `footer`: never starts a turn; FooterController drains it at `before_round`.
  [`inbound.py:9-58`](../spellbook/inbound.py#L9),
  [`footer.py:53-71`](../spellbook/footer.py#L53)

Runtime marks submissions queued when already running or when another turn is
pending. Active injections are the only messages intended to enter the current
turn.
[`runtime.py:358-378`](../spellbook/app/runtime.py#L358)

### I1 — Conduit routing

Profiles with `conduit_surfaces=false` refuse all conduit handling before type
routing; the three fork/bodyless profiles set this flag false.
[`runtime.py:163-169`](../spellbook/app/runtime.py#L163),
[`profiles.py:91-131`](../spellbook/profiles.py#L91)

- `context` always queues a conduit footer.
  [`runtime.py:170-195`](../spellbook/app/runtime.py#L170)
- `message` submits an injection.
  [`runtime.py:197-230`](../spellbook/app/runtime.py#L197)
- `notification` wakes an idle, turn-empty session as a framed injection;
  otherwise it becomes a footer for the current/next round.
  [`runtime.py:232-309`](../spellbook/app/runtime.py#L232)

The “crackle path” is therefore an ordinary system-origin turn, while conduit
notification wakeup is an ordinary conduit-origin injection. Both pass through
the same queue and turn machinery after their guarded submission.

## 10. Failure states are first-class

### F0 — Generator transient retry

Retryable 500/529 API errors, overload/api error types, connection errors, API
timeouts, and timeout exceptions retry up to five total attempts with capped
exponential backoff and jitter. Cancellation during backoff returns an empty
cancelled generation. Non-retryable errors and exhaustion re-raise.
[`generator.py:32-38`](../spellbook/generator.py#L32),
[`generator.py:88-124`](../spellbook/generator.py#L88),
[`generator.py:180-230`](../spellbook/generator.py#L180)

Transition `F0 -> R1 Generate`: next attempt reuses the same already-built
request surface.
[`generator.py:73-91`](../spellbook/generator.py#L73)

Transition `F0 -> F6 Unfinished/crashed turn`: an exhausted generator exception
is not caught by `run_loop` or `_running_phase`; `Recorder.end_turn` is not
reached. The app task callback reports the crash, and rehydration later marks the
open turn unfinished.
[`generator.py:92-97`](../spellbook/generator.py#L92),
[`session_manager.py:171-187`](../spellbook/session_manager.py#L171),
[`runtime.py:535-553`](../spellbook/app/runtime.py#L535),
[`rehydrator.py:290-315`](../spellbook/rehydrator.py#L290)

### F1 — Refusal with preserved debug segments

Anthropic streaming accumulates ordered text, thinking, and partial tool JSON
segments while it emits normal stream events. If the final provider stop reason
is `refusal`, those segments become canonical `IRRefusalBlock` truth plus
provider details.
[`anthropic.py:76-97`](../spellbook/backends/anthropic.py#L76),
[`anthropic.py:189-261`](../spellbook/backends/anthropic.py#L189)

Transition `F1 -> R6 Loop exit`: refusal is a non-tool generation stop reason, so
it is recorded, integrated into Homunculus coordinates, and exits without
execution.
[`loop.py:53-68`](../spellbook/loop.py#L53),
[`ir_types.py:287-297`](../spellbook/ir_types.py#L287)

On later surfaces, policy projects canonical refusal truth into partial
assistant text plus an honest system-interruption note; semantic analysis uses a
count-preserving one-block projection so detector coordinates remain stable.
[`refusal.py:75-166`](../spellbook/refusal.py#L75)

### F2 — Detection rejected or partially deferred

Detection can fail known-context validation, contiguity/overlap checks, semantic
prefix validation, or the tool-closure validator. Full failure shuts down the
fork and queues a courtesy footer without a detection record; partial failure
lands the safe prefix and discloses deferred/discarded ranges.
[`block_detector.py:156-190`](../spellbook/homunculus/block_detector.py#L156),
[`block_manager.py:539-600`](../spellbook/homunculus/block_manager.py#L539),
[`block_manager.py:661-692`](../spellbook/homunculus/block_manager.py#L661)

### F3 — Forget refuses a torn boundary

Before summary application, Forget scans tool-closure envelopes. A split or open
call/result batch returns `status="boundary_invalid"`, leaves mode unchanged,
and queues a keyed runtime footer explaining the required boundary repair.
[`block_manager.py:955-985`](../spellbook/homunculus/block_manager.py#L955),
[`block_manager.py:1364-1472`](../spellbook/homunculus/block_manager.py#L1364)

### F4 — Bash timeout, cancellation, and detach semantics

Bash starts a new process session and merges stderr into stdout. Timeout sends
SIGTERM to the entire process group, waits 250 ms, sends SIGKILL, drains output,
and returns a `ToolError` containing partial output. Executor turns that into an
error result so the model can recover.
[`filesystem.py:344-348`](../spellbook/tools/filesystem.py#L344),
[`filesystem.py:408-422`](../spellbook/tools/filesystem.py#L408),
[`filesystem.py:425-487`](../spellbook/tools/filesystem.py#L425),
[`executor.py:123-131`](../spellbook/executor.py#L123)

If the shell itself exits but a deliberate background child still holds the
output pipe, Bash waits one second, returns with an explicit incomplete-output
note, and leaves a drain task running so that detached writer is not blocked.
Cancellation differs: if the shell is still alive it kills the process group;
only a background descendant that outlived an already-exited shell can remain.
[`filesystem.py:350-405`](../spellbook/tools/filesystem.py#L350),
[`filesystem.py:489-514`](../spellbook/tools/filesystem.py#L489)

### F5 — Cooperative and forced shutdown

An interrupt or shutdown sets the turn cancel token. Generator cancellation
returns partial output; executor cancellation kills the active tool task and
emits error results for the active and all pending calls. `run_loop` records a
cancelled result through normal loop exit.
[`session_manager.py:213-226`](../spellbook/session_manager.py#L213),
[`generator.py:132-160`](../spellbook/generator.py#L132),
[`executor.py:69-112`](../spellbook/executor.py#L69),
[`loop.py:77-105`](../spellbook/loop.py#L77)

The interactive CLI grants graceful shutdown five seconds; if the session task
still has not exited, it force-cancels the task. This can bypass the manager's
final suspended/nursery lifecycle if cancellation lands inside shutdown work.
[`interactive.py:466-480`](../scripts/interactive.py#L466)

### F6 — Fork or nursery failure

Nursery captures exception/cancellation as a harvestable result. BlockManager
records fork shutdown for failed/cancelled detector and summarizer jobs, without
fabricating semantic state. Quantum forks also alert and write an error-noted
shutdown while preserving their snapshot directory.
[`nursery.py:164-202`](../spellbook/nursery.py#L164),
[`nursery.py:321-328`](../spellbook/nursery.py#L321),
[`block_manager.py:1132-1158`](../spellbook/homunculus/block_manager.py#L1132),
[`fork.py:367-387`](../spellbook/fork.py#L367)

## 11. Three narrated walkthroughs

### Walkthrough A — A message arrives

1. **Inbound.** Runtime accepts a `turn`, or an `inject` while idle, notes
   activity, and enqueues it. A queued turn waits behind the active turn.
   [`runtime.py:358-378`](../spellbook/app/runtime.py#L358)
2. **Idle exits.** `take_turn()` selects the first turn-eligible message;
   SessionManager announces exit from idle and pushes the message back for the
   running phase. [`session_manager.py:148-160`](../spellbook/session_manager.py#L148)
3. **Turn becomes transcript truth.** The manager writes turn start and inbound
   blocks, fires turn-start hooks (including timekeeper), creates a cancel token,
   and calls `Homunculus.render_context`. [`session_manager.py:162-184`](../spellbook/session_manager.py#L162)
4. **Context joins awareness.** The Homunculus assigns the inbound blocks global
   ids, accumulates them in the detector, then renders each semantic block in its
   active mode plus the raw tail and TTL collapse projection.
   [`homunculus.py:153-161`](../spellbook/homunculus/homunculus.py#L153)
5. **Before generation.** Skill/time/inbound-footer producers run; FooterController
   drains all pending footer records into one system block. This is where resume
   time, gauge, planner, detector courtesy, conduits, and other ambient signals
   become visible. [`session_manager.py:421-450`](../spellbook/session_manager.py#L421),
   [`footer.py:164-185`](../spellbook/footer.py#L164)
6. **Round heartbeat.** Generate integrates usage, block ids, detector input,
   and gauge pressure. Tool use executes, records/integrates results, and
   registers eligible TTLs. [`loop.py:52-96`](../spellbook/loop.py#L52),
   [`homunculus.py:499-510`](../spellbook/homunculus/homunculus.py#L499)
7. **Boundary work.** Between tool rounds the Homunculus harvests detection,
   metrics, and summary work; ticks round TTLs; checks planner; and rerenders if
   context projection changed. [`homunculus.py:919-925`](../spellbook/homunculus/homunculus.py#L919)
8. **Detection schedules opportunistically.** Any append that crosses the
   interval may already have placed a detector `PreparedFork` into Nursery; its
   eventual completion cannot mutate memory until a later boundary harvest.
   [`block_detector.py:108-148`](../spellbook/homunculus/block_detector.py#L108),
   [`block_manager.py:729-743`](../spellbook/homunculus/block_manager.py#L729)
9. **Turn ends.** A non-tool stop reason fires loop-exit nursery harvest; only
   `end_turn` ticks end-turn TTLs. The manager records the turn end and app
   activity time, then handles the next queued turn or returns idle.
   [`homunculus.py:927-930`](../spellbook/homunculus/homunculus.py#L927),
   [`session_manager.py:184-187`](../spellbook/session_manager.py#L184)

### Walkthrough B — The planner proposes compaction

1. **Pressure observation.** A generation reports provider input tokens. The gas
   gauge moves regimes and, on a 50K bucket crossing, queues a keyed high-priority
   gauge footer. [`homunculus.py:499-505`](../spellbook/homunculus/homunculus.py#L499),
   [`gas_gauge.py:50-63`](../spellbook/homunculus/gas_gauge.py#L50)
2. **Proposal.** At warning pressure (soft threshold), Planner selects the oldest
   full, unpinned, summary-ready block and returns a proposal.
   [`planner.py:25-41`](../spellbook/homunculus/planner.py#L25),
   [`planner.py:46-63`](../spellbook/homunculus/planner.py#L46)
3. **Truth and courtesy.** Homunculus records `context_plan_proposal`, exposes it
   through Reflect, and queues a compaction footer. Footer lifecycle delivers it
   at the next `before_round`. [`homunculus.py:538-610`](../spellbook/homunculus/homunculus.py#L538),
   [`footer.py:177-185`](../spellbook/footer.py#L177)
4. **Entity chooses Forget.** The model calls Forget before the forced threshold.
   BlockManager rechecks pin, summary availability, and tool closure; a clean
   block writes a model-sourced summary apply-mode record.
   [`self_work.py:79-93`](../spellbook/tools/self_work.py#L79),
   [`block_manager.py:955-985`](../spellbook/homunculus/block_manager.py#L955)
5. **Or Planner consumes it.** If input reaches the medium threshold first, the
   next planner check calls the same Forget path with source `planner` and
   announces the forced action. [`homunculus.py:574-610`](../spellbook/homunculus/homunculus.py#L574)
6. **Projection changes.** The full context slice becomes its memory summary
   (plus any pinned facets), and Homunculus invalidates its rendered context,
   prefix counts, gas count, and proposal. [`block_manager.py:153-190`](../spellbook/homunculus/block_manager.py#L153),
   [`homunculus.py:860-878`](../spellbook/homunculus/homunculus.py#L860)
7. **Gauge drop becomes observable.** The next provider call supplies a new exact
   input-token observation; GasGauge exits unknown, establishes the smaller
   bucket/regime, and queues the replacement gauge footer.
   [`gas_gauge.py:41-63`](../spellbook/homunculus/gas_gauge.py#L41)

### Walkthrough C — A detection completes

1. **Fork returns, but memory does not move.** The child task completes and
   Nursery marks its job ready. [`nursery.py:164-185`](../spellbook/nursery.py#L164)
2. **Boundary harvest.** Between rounds, loop exit, an explicit drain, or shutdown
   collects the result and dispatches `detect_blocks`.
   [`block_manager.py:1100-1166`](../spellbook/homunculus/block_manager.py#L1100)
3. **Simulate/sanitize.** Ranges are sorted, bounded to known context, expanded
   around tool pairs, split into a contiguous completed prefix and safe buffer,
   and classified as accepted/deferred/discarded without mutation.
   [`block_detector.py:156-190`](../spellbook/homunculus/block_detector.py#L156)
4. **Validate.** Candidate semantic blocks must tile the canonical prefix and
   completely contain every tool-call/result envelope.
   [`block_manager.py:539-554`](../spellbook/homunculus/block_manager.py#L539),
   [`block_manager.py:1308-1435`](../spellbook/homunculus/block_manager.py#L1308)
5. **Success branch: crystallize.** The parent records fork shutdown and sanitized
   detection, then records each new semantic block, queues “New block
   crystallized,” and schedules metrics plus the first missing summary.
   [`block_detector.py:192-200`](../spellbook/homunculus/block_detector.py#L192),
   [`block_manager.py:602-643`](../spellbook/homunculus/block_manager.py#L602)
6. **Summary cascade.** A harvested summary validates stable block identity,
   records the artifact, and schedules the next oldest missing summary. The
   cascade sustains itself at nursery boundaries.
   [`block_manager.py:745-894`](../spellbook/homunculus/block_manager.py#L745),
   [`block_manager.py:1108-1114`](../spellbook/homunculus/block_manager.py#L1108)
7. **Discard branch: stay healthy.** Any validation error shuts down the fork,
   leaves detector and semantic state untouched, emits debug detail, and queues
   “failed validation and was discarded” for the entity. A partial result lands
   only its safe prefix and queues the partial-defer courtesy.
   [`block_manager.py:555-600`](../spellbook/homunculus/block_manager.py#L555),
   [`block_manager.py:661-692`](../spellbook/homunculus/block_manager.py#L661)

## 12. Future Sleep attachment points

These are the only future edges drawn on the map. They are not claims about
current runtime behavior.

1. **Quiescent waking boundary -> `dreaming`.** Self-triggered Sleep is a manual
   entity action and the dreaming mind is sequentially in a different,
   mutually-exclusive state. The current quantum primitive also requires a
   transcript with no in-progress turn. Therefore the map attaches the future
   transition after the current turn is closed, not inside live generation or
   nursery mutation.
   [Memory Tree design `memory-tree.md:83-91,100-111`](../../../.forge/spellbook/workspace/design/memory-tree.md),
   [`fork.py:508-535`](../spellbook/fork.py#L508)
2. **`dreaming` -> frontier plan application -> wake/idle.** The existing
   frontier derives a pure view and computes mode transitions; it intentionally
   does not mutate or record. Slice 2 is expected to execute those transitions
   through existing apply-mode machinery, then report actual deltas/debts in a
   morning manifest. [`frontier.py:1-10`](../spellbook/dreaming/frontier.py#L1),
   [`frontier.py:294-465`](../spellbook/dreaming/frontier.py#L294),
   [`frontier.py:477-595`](../spellbook/dreaming/frontier.py#L477)
3. **Critical pressure -> future forced Sleep at a quiescent boundary.** Design
   limits forced Sleep to frontier advancement over existing consented narratives;
   it may not author new memory at the hard limit.
   [Memory Tree design `memory-tree.md:115-120`](../../../.forge/spellbook/workspace/design/memory-tree.md),
   [`frontier.py:650-658`](../spellbook/dreaming/frontier.py#L650)
4. **Wake courtesy is part of the transition.** Every wake must announce what
   moved, what remains owed, and where chapter-bearing dream transcripts live.
   [Memory Tree design `memory-tree.md:141-156`](../../../.forge/spellbook/workspace/design/memory-tree.md),
   [`frontier.py:520-574`](../spellbook/dreaming/frontier.py#L520)

No Sleep edge is attached directly to detector completion, summary completion,
hearth tick, or arbitrary quantum fork completion: current nursery and quantum
contracts make consumers integrate explicitly, and the design specifies Sleep
as the waking/dreaming state boundary.

## 13. ⚠ Surprises and findings

### ⚠ 1. Mid-turn injection crosses transcript truth without crossing Homunculus awareness

The main-profile `InboundInjectionRoundLifecycle` appends each active injection
to `RoundContext` and records it, but never calls
`Homunculus.integrate_*`/`BlockManager.append_context_blocks`.
[`inbound.py:84-94`](../spellbook/inbound.py#L84)

The next generated block *is* appended by Homunculus, using its unchanged
`next_block_id`. Live semantic coordinates therefore omit the injected block,
while restart rehydration includes every recorded block and seeds
`next_block_id = len(rehydrated.blocks)`. The live and replay coordinate systems
can diverge by one per active injection.
[`homunculus.py:105-108`](../spellbook/homunculus/homunculus.py#L105),
[`homunculus.py:499-510`](../spellbook/homunculus/homunculus.py#L499),
[`rehydrator.py:185-190`](../spellbook/rehydrator.py#L185)

This is documentation only on this card. It now has a separate high-priority
fix card.

### ⚠ 2. `dreaming` is a type-level state with no runtime entrance or exit

Both session and app state unions expose `dreaming`, but current assignments only
set `suspended`, `idle`, and `running`. Sleep slice 2 must add an actual transition
rather than treating the enum member as implemented.
[`session_manager.py:58`](../spellbook/session_manager.py#L58),
[`session_manager.py:122-140`](../spellbook/session_manager.py#L122),
[`protocol.py:43-52`](../spellbook/app/protocol.py#L43)

### ⚠ 3. End-turn TTL means “only the literal `end_turn` stop reason”

Refusal, cancellation, `max_tokens`, `error`, `pause_turn`, `stop_sequence`, and
`unspecified` all close the manager turn, but only
`stop_reason == "end_turn"` decrements end-turn TTLs. This is stricter than the
everyday meaning of “the turn ended.”
[`ir_types.py:287-297`](../spellbook/ir_types.py#L287),
[`homunculus.py:927-930`](../spellbook/homunculus/homunculus.py#L927)

### ⚠ 4. Fatal generation/tool exceptions do not pass through `on_loop_exit`

`run_loop` has no enclosing `try/finally`. Retry exhaustion or a non-`ToolError`
tool exception can escape before `Recorder.end_turn`, leaving an unfinished turn
that replay detects. This is fail-loud behavior, but it is a different failure
state from a normal `stop_reason="error"` exit.
[`loop.py:48-105`](../spellbook/loop.py#L48),
[`generator.py:92-97`](../spellbook/generator.py#L92),
[`executor.py:82-131`](../spellbook/executor.py#L82),
[`rehydrator.py:290-315`](../spellbook/rehydrator.py#L290)

### ⚠ 5. A detector fork may return “success” yet intentionally land only a prefix

Simulation can defer out-of-order completed ranges and discard later buffered
ranges while accepting a contiguous prefix. This is a degraded success, not an
all-or-nothing result; the entity gets a courtesy footer.
[`block_detector.py:165-190`](../spellbook/homunculus/block_detector.py#L165),
[`block_manager.py:573-600`](../spellbook/homunculus/block_manager.py#L573)

### ⚠ 6. Forget can start summary generation even when the normal cascade has not reached the block

Forget is not merely a mode switch. If a target summary is missing, it can bypass
the oldest-first wait by scheduling that target with the previous five available
summaries, then ask the entity to retry. This makes the user action an awareness
work scheduler.
[`block_manager.py:901-953`](../spellbook/homunculus/block_manager.py#L901)

### ⚠ 7. Bash “backgrounding” and timeout have deliberately different kill semantics

A timeout kills the process group. A shell that has already exited successfully
while a background descendant holds stdout instead returns after a one-second
drain grace and lets the descendant run. The incomplete-output note is the only
surface signal of that detached state.
[`filesystem.py:382-422`](../spellbook/tools/filesystem.py#L382),
[`filesystem.py:469-514`](../spellbook/tools/filesystem.py#L469)

## 14. Compact transition ledger

This ledger is intended for grep and code review. Its 80 rows are generated from
the same edge inventory rendered by the HTML diagram.

| Edge | Trigger / guard | Layer | Provenance |
| --- | --- | --- | --- |
| Build / resume -> Rehydrated world | typed read | session | `spellbook/session_manager.py:248-307` |
| Rehydrated world -> Suspended | wire services | session | `spellbook/session_manager.py:332-481` |
| Suspended -> Idle | run() | session | `spellbook/session_manager.py:134-150` |
| Idle -> Running / turn | eligible message | session | `spellbook/session_manager.py:151-164` |
| Running / turn -> before_round | start + render | session | `spellbook/session_manager.py:171-184` |
| on_loop_exit -> Turn ended | loop result | session | `spellbook/session_manager.py:184-187` |
| Turn ended -> Running / turn | next queued turn | session | `spellbook/session_manager.py:164-187` |
| Turn ended -> Idle | queue empty | session | `spellbook/session_manager.py:134-140` |
| Idle -> Shutdown | queue shutdown | session | `spellbook/session_manager.py:145-155` |
| Running / turn -> Shutdown | cancel active | failure | `spellbook/session_manager.py:213-217` |
| Shutdown -> Suspended | harvest + close | session | `spellbook/session_manager.py:134-143` |
| before_round -> Generate / stream | request ready | round | `spellbook/loop.py:48-53` |
| Generate / stream -> after_generate | response | round | `spellbook/loop.py:53-59` |
| after_generate -> Execute tools | tool_use | round | `spellbook/loop.py:60-75` |
| after_generate -> on_loop_exit | non-tool stop | round | `spellbook/loop.py:60-68` |
| Execute tools -> after_execute | result blocks | round | `spellbook/loop.py:70-75` |
| after_execute -> between_rounds | nonterminal | round | `spellbook/loop.py:77-96` |
| after_execute -> on_loop_exit | cancel / terminal | round | `spellbook/loop.py:77-94` |
| between_rounds -> before_round | next round | round | `spellbook/loop.py:48-52` |
| Running / turn -> Append context | inbound render | awareness | `spellbook/homunculus/homunculus.py:153-161` |
| after_generate -> Append context | generation blocks | awareness | `spellbook/homunculus/homunculus.py:499-505` |
| after_execute -> Append context | tool results | awareness | `spellbook/homunculus/homunculus.py:507-510` |
| Append context -> Detector accumulation | ids + counter | awareness | `spellbook/homunculus/block_manager.py:135-151` |
| Detector accumulation -> Prepared detector fork | interval / force | awareness | `spellbook/homunculus/block_detector.py:108-148` |
| Prepared detector fork -> Nursery running / ready | best effort | awareness | `spellbook/homunculus/block_manager.py:729-743` |
| Nursery running / ready -> Simulate + sanitize | boundary harvest | awareness | `spellbook/homunculus/block_manager.py:1100-1140` |
| Simulate + sanitize -> Validate candidates | safe candidates | awareness | `spellbook/homunculus/block_manager.py:539-554` |
| Validate candidates -> Block crystallizes | valid prefix | awareness | `spellbook/homunculus/block_manager.py:602-643` |
| Validate candidates -> Graceful discard | ValueError | failure | `spellbook/homunculus/block_manager.py:555-557,661-692` |
| Block crystallizes -> Summary job | cascade | awareness | `spellbook/homunculus/block_manager.py:642-643,836-894` |
| Block crystallizes -> Metrics integrated | stable-id count | awareness | `spellbook/homunculus/block_manager.py:627-641` |
| Summary job -> Nursery running / ready | keyed job | awareness | `spellbook/homunculus/block_manager.py:853-894` |
| Nursery running / ready -> Summary integrated | harvest result | awareness | `spellbook/homunculus/block_manager.py:1143-1158` |
| Summary integrated -> Summary job | next missing | awareness | `spellbook/homunculus/block_manager.py:817-849` |
| Detector · Summary · Quantum -> Quiescent snapshot | quantum config | awareness | `spellbook/fork.py:185-192,281-389` |
| Nursery running / ready -> Fork failure | error / cancel | failure | `spellbook/homunculus/block_manager.py:1132-1158` |
| Block crystallizes -> Full block | born full | memory | `spellbook/homunculus/block_manager.py:645-659` |
| Summary integrated -> Full block | mode now available | memory | `spellbook/homunculus/block_manager.py:795-818` |
| Full block -> Summary block | Forget / Planner | memory | `spellbook/homunculus/block_manager.py:955-985` |
| Summary block -> Pin block / facet | Pin | memory | `spellbook/homunculus/block_manager.py:989-1026` |
| Pin block / facet -> Full block | block pin restores | memory | `spellbook/homunculus/block_manager.py:989-1003` |
| Summary block -> Recall | tool result only | memory | `spellbook/homunculus/block_manager.py:1028-1046` |
| Pair parent -> Pair child | atomic chapter | memory | `spellbook/homunculus/block_manager.py:279-328` |
| Full block -> Reflect / Configure | inspect | memory | `spellbook/homunculus/homunculus.py:163-254` |
| Summary block -> Reflect / Configure | inspect / preview | memory | `spellbook/homunculus/homunculus.py:163-254` |
| Full block -> Forget refused | torn boundary | failure | `spellbook/homunculus/block_manager.py:955-985,1364-1472` |
| after_execute -> Tool result | observe result | memory | `spellbook/homunculus/tool_result_ttl.py:263-267` |
| Tool result -> Pending TTL | eligible / manual | memory | `spellbook/homunculus/tool_result_ttl.py:269-345` |
| Pending TTL -> Collapsed projection | ticks → 0 | memory | `spellbook/homunculus/tool_result_ttl.py:347-376` |
| Collapsed projection -> between_rounds | rerender | memory | `spellbook/homunculus/homunculus.py:612-618,919-925` |
| after_generate -> Token observations | usage | pressure | `spellbook/homunculus/homunculus.py:499-505` |
| Token observations -> Gas gauge | exact input | pressure | `spellbook/homunculus/gas_gauge.py:50-63` |
| Gas gauge -> Planner proposal | >= 700K | pressure | `spellbook/homunculus/planner.py:46-63` |
| Planner proposal -> Forget / forced action | entity or >=850K | pressure | `spellbook/homunculus/homunculus.py:574-610` |
| Forget / forced action -> Summary block | apply mode | pressure | `spellbook/homunculus/block_manager.py:1251-1290` |
| Summary block -> Token observations | next generation | pressure | `spellbook/homunculus/gas_gauge.py:41-63` |
| Inbound queue -> Running / turn | turn / idle inject | inbound | `spellbook/session_manager.py:148-187` |
| Inbound queue -> Active injection | active inject | inbound | `spellbook/inbound.py:46-58,84-94` |
| Active injection -> before_round | round + record only ⚠ | failure | `spellbook/inbound.py:84-94` |
| Inbound queue -> Footer pending | footer delivery | inbound | `spellbook/inbound.py:32-44` |
| Footer pending -> before_round | drain + inject | inbound | `spellbook/footer.py:164-185` |
| Conduit routing -> Active injection | message / idle notif | inbound | `spellbook/app/runtime.py:197-283` |
| Conduit routing -> Footer pending | context / busy notif | inbound | `spellbook/app/runtime.py:170-195,285-309` |
| Conduit routing -> Conduit refused | profile gate | failure | `spellbook/app/runtime.py:163-169` |
| Hearth waiting / skipped -> Crackle turn | eligible tick | inbound | `spellbook/hearth.py:251-293` |
| Crackle turn -> Inbound queue | normal turn | inbound | `spellbook/app/runtime.py:137-144` |
| Generate / stream -> Transient retry | transient | failure | `spellbook/generator.py:88-124` |
| Transient retry -> Generate / stream | same surface | failure | `spellbook/generator.py:73-91` |
| Generate / stream -> Unfinished / crashed turn | exhausted / fatal | failure | `spellbook/generator.py:92-97` |
| after_generate -> Refusal | refusal stop | failure | `spellbook/backends/anthropic.py:139-145` |
| Refusal -> on_loop_exit | non-tool exit | failure | `spellbook/loop.py:60-68` |
| Execute tools -> Bash timeout | timeout | failure | `spellbook/tools/filesystem.py:469-487` |
| Execute tools -> Detached drain | shell exited | failure | `spellbook/tools/filesystem.py:489-514` |
| Shutdown -> Forced task cancellation | CLI > 5s | failure | `scripts/interactive.py:466-480` |
| Future: Turn ended -> Dreaming | future: quiescent Sleep | future | `Forge memory-tree.md:83-111; spellbook/fork.py:508-535` |
| Future: Gas gauge -> Dreaming | future: forced hard limit | future | `Forge memory-tree.md:115-120; spellbook/dreaming/frontier.py:650-658` |
| Future: Dreaming -> Apply frontier + manifest | future: dream / plan | future | `spellbook/dreaming/frontier.py:1-10,294-465` |
| Future: Apply frontier + manifest -> Idle | future: manifest + wake | future | `Forge memory-tree.md:141-156; spellbook/dreaming/frontier.py:477-595` |
| Future: Apply frontier + manifest -> Summary block | future: apply modes | future | `spellbook/homunculus/block_manager.py:1251-1290; spellbook/dreaming/frontier.py:1-10` |
| Future: Apply frontier + manifest -> Pair parent | future: narrative mode | future | `spellbook/dreaming/frontier.py:294-465` |

## Open concerns

- The active-injection coordinate drift is a separate high-priority correctness
  issue; this artifact deliberately does not fix it.
- The implementation has no actual Sleep entry/exit yet. The future edges above
  should be revalidated when slice 2 chooses its terminal tool result, session
  lifecycle hooks, and failure recovery semantics.
- Fatal exceptions and interactive force-cancel do not guarantee all normal
  loop/session exit hooks. Sleep should not assume those hooks always ran when
  recovering a transcript flagged `is_unfinished_turn`.
- This reference cites the current working tree, which already contained
  unrelated refusal/IR/rehydrator edits. Re-run the provenance audit if those
  files change before the card lands.
