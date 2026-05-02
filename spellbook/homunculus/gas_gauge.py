"""Gas-gauge telemetry emitted through the footer pipeline.

This module turns input-token observations into a keyed telemetry footer so the
active session can surface coarse context pressure to the model.

Important invariants:

- gas gauge state is ephemeral awareness/telemetry, not transcript history
  rewriting
- emissions flow through `FooterController`, so they participate in the same
  queue/dedup/render pipeline as other footers
- the footer key is stable (`gas_gauge`), so a new bucket replaces the previous
  gauge rather than accumulating duplicates
- bucket changes, not every token change, are the trigger surface
- regime calculation comes from `calc_regime()` and should stay aligned with the
  Homunculus token-pressure thresholds

If you change this module, keep gas-gauge bucketing, regime semantics, footer
dedup, and Homunculus generation observation coherent together.
"""

from spellbook.config import HomunculusConfig
from spellbook.footer import FooterController

from .common import RegimeType, calc_regime


class GasGauge:
    """Tracks current context pressure and emits bucket-crossing footers."""

    BUCKET_SIZE = 50_000

    def __init__(self, *, config: HomunculusConfig, footer_c: FooterController):
        self._config = config
        self._footer_c = footer_c
        self._last_bucket = -1
        self._input_tokens = 0
        self._is_invalid: bool = False

    @property
    def input_tokens(self) -> int | None:
        if self._is_invalid:
            return None
        return self._input_tokens

    @property
    def regime(self) -> RegimeType:
        return calc_regime(self._config, self.input_tokens)

    def observe(self, input_tokens: int) -> None:
        self._is_invalid = False
        self._input_tokens = input_tokens
        bucket = input_tokens // self.BUCKET_SIZE
        if bucket == self._last_bucket:
            return
        self._last_bucket = bucket
        self._footer_c.queue_footer(
            text=f"[context: {self._input_tokens // 1000}K / 1M - {self.regime}]",
            footer_type="gas_gauge",
            source="telemetry",
            key="gas_gauge",
            priority=10,  # high prio
        )

    def invalidate(self) -> None:
        self._is_invalid = True
        self._last_bucket = -1
        self._footer_c.clear_footer("gas_gauge")
