# Spellbook SDK

> Draft interface guide. This document sketches the intended shape of the
> Spellbook SDK before the API is implemented. Names and signatures may still
> change, but the lifecycle model is the design center.

The Spellbook SDK is a small Python interface for scripting Spellbook entities
from ordinary async code.

It is meant for workflows like:

- run one entity through a multi-turn task
- spin up several fresh entities from the same recipe
- attach custom tools to a scripted session
- capture returned text, raw blocks, and transcript artifacts
- use Spellbook's memory, compaction, footers, and runtime lifecycle without
  running the HTTP app server directly

The core idea is:

- `Spell` is an inert recipe.
- `Entity` is a live summoned runtime.
- `TurnResult` is the result of one message.

```python
from spellbook.sdk import Spell

spell = Spell(config=config)

async with spell.cast() as entity:
    result = await entity.send("Write the opening scene.")
    print(result.text)
```

## Mental Model

A `Spell` is not alive. It does not own a running session loop. It holds the
configuration needed to create or resume an entity:

- model/provider config
- transcript path or transcript creation policy
- optional custom surface
- optional runtime settings

An `Entity` is alive. It owns one running `CoreAppRuntime` and one active
session loop. It can receive messages, stream events, and shut down cleanly.

The context manager is the lifecycle boundary:

```python
async with spell.cast() as entity:
    ...
```

Everything inside the block talks to the same live entity. Leaving the block
shuts down the runtime.

## Basic Usage

```python
from pathlib import Path

from spellbook.config import SpellbookConfig
from spellbook.sdk import Spell

config = SpellbookConfig(
    provider="anthropic",
    model="claude-opus-4-7",
    cwd=Path.cwd(),
    system_prompt="You are a careful writing partner.",
)

spell = Spell(config=config)


async def main() -> None:
    async with spell.cast() as entity:
        result = await entity.send("Write a short paragraph about a glass city.")
        print(result.text)
```

`send()` submits a normal human turn and waits for that turn to finish.

The returned `TurnResult` preserves both convenient text and structured data:

```python
result.text          # concatenated assistant text
result.blocks        # generated assistant blocks for the turn
result.turn_id       # durable turn id
result.stop_reason   # end_turn, tool_use, cancelled, etc.
result.loop_result   # full IRLoopResult
result.stream_events # live IR stream events observed during the turn
```

## Multi-Turn Continuity

One `cast()` block means one live entity. Use this when later turns should see
earlier turns.

```python
spell = Spell(config=config)


async def write_story() -> list[str]:
    chapters = []

    async with spell.cast() as entity:
        for i in range(5):
            result = await entity.send(f"Write chapter {i}.")
            chapters.append(result.text)

    return chapters
```

In this form, chapter 4 is written by the same live entity that wrote chapters
0 through 3. It has the same transcript, same Homunculus state, same context
manager, same planner, and same memory lifecycle.

## Fresh Casts

Move the context manager inside the loop when each run should be a separate
live instance.

```python
spell = Spell(config=config)


async def write_variants() -> list[str]:
    variants = []

    for i in range(5):
        async with spell.cast() as entity:
            result = await entity.send(f"Write variant {i} of the opening scene.")
            variants.append(result.text)

    return variants
```

Whether these are totally fresh or resume from an existing transcript depends
on the spell's transcript policy.

## Transcript Semantics

Spellbook transcripts are durable session truth. The SDK should make transcript
ownership explicit but lightweight.

### Ephemeral Transcript

If no transcript path is provided, the spell can create a new temporary or
session-directory transcript for each cast.

```python
spell = Spell(config=config)

async with spell.cast() as entity:
    ...
```

This is useful for tests, one-off generation, or parallel independent runs.

### Fixed Transcript

Pass a transcript path when you want a durable entity that can resume.

```python
spell = Spell(
    config=config,
    transcript_path=Path("story-session.jsonl"),
)

async with spell.cast() as entity:
    await entity.send("Continue from where we left off.")
```

If the transcript exists, the cast resumes it. If it does not exist, the cast
initializes it from `config`.

### Per-Cast Transcript

For batch work, override the transcript path for each cast.

```python
spell = Spell(config=config)

for i in range(5):
    async with spell.cast(transcript_path=Path(f"runs/chapter-{i}.jsonl")) as entity:
        await entity.send(f"Write chapter {i}.")
```

The design goal is that callers should not have to manually build session
managers or recorders just to choose where transcripts live.

## Custom Tools

The SDK should support the same `CustomSurface` used by the app server.

```python
from pydantic import BaseModel, Field

from spellbook.custom import CustomSurface
from spellbook.ir_types import IRToolTextBlock
from spellbook.tools.common import Tool, ToolExecutionResult, ToolMetadata


class LookUpSceneInput(BaseModel):
    key: str = Field(description="The scene key to inspect.")


async def look_up_scene(
    meta: ToolMetadata,
    input: LookUpSceneInput,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        content=[IRToolTextBlock(text=f"Scene data for {input.key}")]
    )


LOOK_UP_SCENE = Tool(
    name="LookUpScene",
    input_model=LookUpSceneInput,
    exec=look_up_scene,
    category="thinking",
)

surface = CustomSurface(
    tools=[LOOK_UP_SCENE],
    include_tool_categories={"memory"},
)

spell = Spell(
    config=config.model_copy(update={"session_type": "custom"}),
    custom_surface=surface,
)
```

Custom sessions still get the core self-state lifecycle: generated assistant
blocks are absorbed into live memory, footers can be delivered, and selected
built-in categories can be exposed intentionally.

## One-Shot Sugar

The context manager should remain the primary API, but simple one-shot calls
are useful.

```python
result = await spell.once("Write a haiku about a lantern.")
```

This is equivalent to:

```python
async with spell.cast() as entity:
    result = await entity.send("Write a haiku about a lantern.")
```

The important caveat: `once()` creates a lifecycle boundary around one message.
If the spell uses an ephemeral transcript policy, each `once()` call is a fresh
entity. If it uses a fixed transcript path, each `once()` call resumes and then
shuts down the same durable transcript.

## Streaming

The non-streaming API should be the simplest path:

```python
result = await entity.send("Write the next scene.")
```

Streaming should be available without changing the lifecycle model:

```python
async with spell.cast() as entity:
    async for event in entity.stream("Write the next scene."):
        match event.kind:
            case "text_delta":
                print(event.text, end="")
```

Possible stream result shape:

```python
async with spell.cast() as entity:
    stream = entity.stream("Write the next scene.")
    async for event in stream:
        ...
    result = await stream.result()
```

The SDK should not invent a second event protocol. Stream events should be
Spellbook IR stream events or thin wrappers around them.

## Results

The default `TurnResult` should optimize for the common scripting case while
preserving full fidelity.

```python
@dataclass(frozen=True)
class TurnResult:
    text: str
    blocks: list[IRBlock]
    turn_id: str
    stop_reason: StopReason
    loop_result: IRLoopResult
    stream_events: list[IRStreamEvent]
```

`text` is the concatenated assistant text from the turn. It is convenient, not
canonical.

`blocks` and `loop_result` are the canonical structured result. If the assistant
produces thinking blocks, tool calls, or other non-text content, callers can
inspect them there.

## Event Access

Most scripts should not need to touch the event bus. But advanced callers may
want runtime events:

```python
async with spell.cast() as entity:
    async with entity.events() as events:
        await entity.send("Begin.")
        async for event in events:
            ...
```

This should be a friendly wrapper around the existing app event bus. Events are
live transport, not durable truth. Durable truth remains the transcript.

## Interrupts

The live `Entity` handle should expose interruption:

```python
async with spell.cast() as entity:
    task = asyncio.create_task(entity.send("Write a long chapter."))
    await asyncio.sleep(2)
    interrupted = await entity.interrupt()
    result = await task
```

Interrupting should preserve whatever partial generation the backend can safely
return, following the same semantics as interactive Spellbook sessions.

## Proposed API Surface

This is the intended small public surface:

```python
class Spell:
    def __init__(
        self,
        *,
        config: SpellbookConfig,
        transcript_path: Path | None = None,
        custom_surface: CustomSurface | None = None,
    ) -> None: ...

    def cast(
        self,
        *,
        transcript_path: Path | None = None,
    ) -> EntityCast: ...

    async def once(
        self,
        message: str | IRInboundMessage,
        *,
        transcript_path: Path | None = None,
    ) -> TurnResult: ...


class Entity:
    async def send(
        self,
        message: str | IRInboundMessage,
        *,
        metadata: dict | None = None,
    ) -> TurnResult: ...

    def stream(
        self,
        message: str | IRInboundMessage,
        *,
        metadata: dict | None = None,
    ) -> EntityStream: ...

    async def interrupt(self) -> bool: ...

    def health(self) -> HealthResponse: ...
    def awareness(self) -> AwarenessResponse: ...
```

The exact names can change. The lifecycle shape should not:

```python
spell = Spell(...)

async with spell.cast() as entity:
    result = await entity.send(...)
```

## Implementation Sketch

The SDK should be a thin layer over `CoreAppRuntime`, not a separate runtime.

Internally, `Spell.cast()` should:

1. Resolve the transcript path.
2. Build a `CoreAppRuntime` with `config`, `custom_surface`, and an event bus.
3. Start the runtime.
4. Return an `Entity` handle.
5. Shut down the runtime on context exit.

Internally, `Entity.send()` should:

1. Subscribe to the runtime event bus before submitting the message.
2. Attach an SDK request id to message metadata.
3. Submit an `IRInboundMessage`.
4. Wait for the matching `TurnStartedEvent`.
5. Collect stream events and generated blocks.
6. Return when the matching `TurnEndedEvent` arrives.

This keeps the SDK aligned with the app server, websocket protocol, recorder,
rehydrator, and transcript model. There should be one runtime truth.

## Design Principles

**Lifecycle must be explicit.** The context manager should make it obvious when
an entity is alive and when it shuts down.

**Durability should be opt-in but easy.** A caller should be able to get a fresh
one-off entity with almost no ceremony, and a durable resumable entity by adding
a transcript path.

**The SDK should not hide Spellbook's IR.** Convenience strings are welcome, but
structured blocks and loop results should remain available.

**Custom surfaces should feel first-class.** The same custom tool surface used by
the app server should work here.

**No parallel runtime.** The SDK should orchestrate `CoreAppRuntime`; it should
not duplicate the session loop, event bus, or lifecycle machinery.

## Open Questions

**Naming.** `Spell.cast()` and `Entity` feel natural, but alternatives remain:
`Ritual.summon()`, `Cantrip.cast()`, `Session`, `EntityHandle`.

**Transcript defaults.** Should omitted transcript paths create temp files,
session-directory files, or in-memory disposable transcripts? Spellbook's
recorder currently wants a real transcript path, so "session-directory file" is
the likely default.

**Concurrent sends.** The first version should probably serialize `send()` calls
per live entity. Concurrency can be added later with explicit queuing semantics.

**Event filtering.** `send()` should match its own turn robustly. Message metadata
with an SDK request id is probably enough, but this should be tested carefully
against queued turns and injected messages.

**Remote SDK.** This document describes the local in-process SDK. A future remote
client could expose the same shape over HTTP/websocket, but that should be a
separate layer.
