# Conduit-on-idle: wake for stranded notifications

Status: proposed design for card `45f7ca79`

## The gap

Spellbook does not currently have a separate conduit queue. Conduits are
translated by `CoreAppRuntime` into `IRInboundMessage`s and placed on the one
`InboundMessageQueue`. That queue has three delivery modes:

- `turn` and `inject` are eligible to start a turn;
- `footer` is deliberately invisible to idle waiting and is drained only by
  `FooterControllerRoundLifecycle.before_round()`.

This is almost the desired behavior. A `message` conduit is an `inject`, so a
message which misses the current turn's last round is already picked up by
`SessionManager._running_phase()` as a new turn. A `notification` conduit is
different: it is a framed injection while the session is idle, but becomes a
footer whenever the session is running or another turn is pending. See
[`runtime.py`](../spellbook/app/runtime.py#L197),
[`runtime.py`](../spellbook/app/runtime.py#L232), and
[`inbound.py`](../spellbook/inbound.py#L15).

That creates a narrow but consequential race:

1. the last `before_round` of turn N has already drained footers;
2. the model is streaming its final response, so runtime state is still
   `running`;
3. a `notification` conduit arrives and is queued as `footer`;
4. the model ends the turn;
5. `has_pending_turn()` ignores the footer, so the running phase ends and the
   session enters idle;
6. idle `take_turn()` also ignores the footer, leaving it unseen until some
   later external message starts a turn.

This is the observed Minecraft path, not a hypothetical one: the bridge sends
both chat and world events as `type="notification"`. During a tool-heavy turn,
most notifications are drained normally at subsequent round boundaries. The
last one can instead be stranded during stream-out.

The fix should preserve the useful distinction among conduit types:

| Conduit type | While a turn is active | If no turn is active |
| --- | --- | --- |
| `context` | footer at a later round | remain pending; do not wake |
| `message` | inject at a later round | start a turn |
| `notification` | footer at a later round | start a turn if the footer would otherwise be stranded |

## Proposed contract

Add a typed wake policy to `IRInboundMessage` rather than teaching
`SessionManager` what a conduit is or treating every footer as turn-eligible.

Illustrative sketch:

```python
class IRInboundMessage(BaseModel, frozen=True):
    blocks: list[IRInboundBlock]
    source_metadata: dict = Field(default_factory=dict)
    delivery: InboundDelivery
    wake_on_idle: bool = False
```

`wake_on_idle=True` means:

> This message keeps its normal delivery behavior while a turn can consume it.
> If it is the only remaining work when the session would become idle, start a
> turn so that normal delivery can occur.

For this slice, validate that the flag is only legal with `delivery="footer"`.
`turn` and `inject` already wake by definition, so accepting the flag on them
would introduce two names for one behavior. Ordinary footer producers keep the
default `False`.

`CoreAppRuntime._queue_conduit_footer()` sets `wake_on_idle=True` only when
`conduit_type == "notification"`. A `context` conduit remains an ordinary
non-waking footer. The current idle-notification path may remain a framed
`inject`; the new flag covers busy notifications and the small race in which a
notification was classified while running but is enqueued after the manager
has crossed into idle.

The conduit footer's `source_metadata` should also retain `source`, `origin`,
and `conduit_type` in addition to the existing footer normalization fields.
This is provenance and observability, not a second copy of conversation truth;
the footer text remains the payload that will be recorded.

## Queue behavior

`InboundMessageQueue` should distinguish two questions:

```python
def _starts_turn_directly(msg: IRInboundMessage) -> bool:
    return msg.delivery in {"turn", "inject"}

def _can_trigger_turn(msg: IRInboundMessage) -> bool:
    return _starts_turn_directly(msg) or msg.wake_on_idle
```

`has_pending_turn()` then means "the session has enough queued work to run
another turn," and uses `_can_trigger_turn`. The name remains useful to callers
such as `SessionManager` and the notification routing check, even though the
content may still be delivered by the footer controller.

`take_turn()` should use this selection order:

1. Return the first `turn` or `inject` message, preserving the order of all
   retained messages. A real queued turn takes precedence over a synthetic
   wake, even if a wakeable footer appears earlier in the deque; that footer
   will be drained into the real turn.
2. If no direct turn exists but at least one `wake_on_idle` footer exists,
   return a synthetic empty `turn` trigger. Do **not** remove the footer. Copy
   the first wakeable footer's provenance into the trigger and add a marker such
   as `wake_reason="pending_footer"`.
3. If neither exists, wait on the condition as today.

The existing idle/running handoff then supplies the rest of the behavior:

```text
turn N ends
  -> has_pending_turn() sees the wakeable footer
  -> running phase asks take_turn() for work
  -> take_turn() returns an empty synthetic turn trigger
  -> turn N+1 starts without an intervening idle transition
  -> FooterController drains and records the original notification footer
  -> the first provider round sees the notification in <spellbook>
```

The trigger has no canonical content of its own. It exists to open the
lifecycle boundary in which the original footer becomes transcript truth. An
empty turn input is already legal in `Recorder.start_turn()`, and the real
content is recorded in the same order as any other footer:

1. `turn_start` for N+1;
2. `footer_queue` and `footer_drain` records;
3. the rendered system-origin `<spellbook>` block;
4. generation/execution records;
5. `turn_end`.

This preserves the recorder/rehydrator meaning described by
[`footer.py`](../spellbook/footer.py#L1) and avoids a special transcript record
for a transient in-memory wake decision. Rehydration only needs to know what
was delivered, not why the already-completed turn began.

`FooterController.drain_footer_messages()` does not need special handling for
the flag. It drains wakeable and ordinary footers alike. If several
notifications have accumulated, one synthetic turn is enough; the controller
batches, priority-sorts, and key-deduplicates all of them before the first
generation. If a human or conduit `message` turn is already queued, no
synthetic turn is created and those same footers join that real turn instead.

Shutdown continues to win. `_running_phase()` already checks
`not self._shutdown_requested`; a pending notification must not resurrect a
session whose owner requested shutdown.

## Why this shape

The tempting one-line fix is to route every notification as `inject`. That
would make it turn-eligible, but it would also change notification behavior
throughout an active turn: notifications would become explicit conduit blocks,
bypass footer priority and key deduplication, stop producing footer queue/drain
records, change the busy response from `queued_as_context` to
`queued_as_message`, and emit `MessageQueuedEvent`s whose current UI accounting
assumes a later `TurnStartedEvent`. The card only requires a notification that
missed the final round to wake the entity. It does not require those broader
surface changes.

Scanning for `footer_type == "conduit"` in `SessionManager` is also too broad.
It would wake for `context` conduits, contradicting their non-interrupting
contract, and would put app transport policy into the session orchestrator.
The `wake_on_idle` flag is generic queue policy; `CoreAppRuntime` decides which
app inputs receive it, while `SessionManager` continues to ask only whether a
turn can run.

The accepted tradeoff is a synthetic turn whose `TurnStartedEvent.message` has
no blocks. The model and transcript still receive the conduit through the
normal footer block before generation. Current catchup UI already treats busy
notification footers as ambient system context rather than user/conduit cards,
so this does not hide content that was previously a visible chat item. If the
UI later needs to render autonomous wake reasons, it should use the copied
source metadata or a dedicated projection event rather than duplicating the
footer as a second canonical block.

## Boundary cases to lock down

The build should include focused coverage for these cases:

1. **The reported race.** Pause a first generation after its last
   `before_round`, enqueue a wakeable notification footer, let it return
   `end_turn`, and assert that a second turn starts immediately. There must be
   no `on_enter_idle` between the two turns, and the second provider request
   must contain the notification footer.
2. **Normal mid-turn delivery.** A wakeable footer arriving before a later tool
   round is drained into that round and does not create an extra turn.
3. **Idle enqueue race.** `take_turn()` wakes for a wakeable footer even if it
   was classified while running and put after the session reached idle.
4. **Context remains quiet.** A `context` conduit and an ordinary footer do not
   wake an idle session.
5. **Real input wins.** If a human `turn` or conduit `inject` is pending, it
   starts the next turn and the notification is delivered as a footer in that
   turn; no empty turn is inserted first.
6. **Batching.** Multiple stranded notifications cause one follow-up turn and
   are drained in footer priority order with existing key deduplication.
7. **Transcript truth.** Rehydration after the follow-up turn sees the rendered
   footer block and no pending copy of the delivered notification. Queue/drain
   records retain the normal footer protocol.
8. **Shutdown.** A shutdown request prevents a wakeable footer from starting a
   follow-up turn.

The existing tests in
[`test_session_manager.py`](../tests/test_session_manager.py#L736) are the main
home for queue and lifecycle behavior. Routing assertions belong in
[`test_app_runtime.py`](../tests/test_app_runtime.py#L336), and the field
validation belongs in [`test_ir_types.py`](../tests/test_ir_types.py).

## Uncertainties and risks

The largest product uncertainty is notification volume. A continuous stream of
Minecraft world events can keep producing follow-up turns instead of permitting
idle. That is consistent with the current meaning of `notification`—it already
wakes a fully idle session—and is necessary for live play, but it has token and
latency costs. Sources that should wait for a future human interaction must use
`context`; this change makes that type distinction more operationally
important.

The synthetic empty trigger is intentionally not durable. All inbound queue
items are currently in memory until delivery, so a process crash can already
lose a queued conduit. Crash-safe conduit intake would require recording an
arrival before delivery and replaying it into the queue; that is a separate,
larger design and should not be smuggled into this race fix.

There is also a naming choice: a fourth delivery mode such as `wake_footer`
could encode the same behavior. I prefer `wake_on_idle` because the payload is
still a footer in every model-visible and transcript-visible sense, and the
boolean composes with the existing delivery taxonomy without adding another
drain category. If future callers need multiple wake policies, the boolean may
need to become an enum; there is no evidence for that complexity yet.

## Expected implementation surface

If accepted, the build should touch:

- `spellbook/ir_types.py` — add and validate `IRInboundMessage.wake_on_idle`;
- `spellbook/inbound.py` — distinguish direct turn inputs from wakeable queued
  work, prefer direct inputs, and synthesize the empty trigger;
- `spellbook/app/runtime.py` — mark busy notification footers wakeable and retain
  conduit provenance in source metadata;
- `tests/test_ir_types.py`, `tests/test_session_manager.py`, and
  `tests/test_app_runtime.py` — cover the contract and the stream-out race;
- `docs/homunculus-state-machine.md` and its HTML companion — revise the claims
  that all footers can never start a turn and document notification-on-idle.

No `SessionProfile` flag, conduit API response variant, recorder record kind,
rehydrator branch, provider backend change, or `run_loop()` change is proposed.
The external `/conduit` request remains compatible, and a busy notification may
continue to report `queued_as_context`: it is queued as footer context first
and only becomes a turn trigger if no other turn can deliver it.
