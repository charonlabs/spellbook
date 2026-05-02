"""Tests for gas gauge thresholds and footer emission."""

from __future__ import annotations

from pathlib import Path

from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.footer import FooterController
from spellbook.homunculus.common import calc_regime
from spellbook.homunculus.gas_gauge import GasGauge
from spellbook.inbound import InboundMessageQueue
from spellbook.ir_types import IRSkillCatalog
from spellbook.recorder import Recorder
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _make_recorder(tmp_path: Path, session_id: str = "s1") -> Recorder:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    recorder = Recorder(config, transcript, session_id, DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("t1", [])
    return recorder


def _make_footer_controller(tmp_path: Path) -> FooterController:
    recorder = _make_recorder(tmp_path)
    inbound = InboundMessageQueue()
    return FooterController(inbound_queue=inbound, recorder=recorder)


class TestCalcRegime:
    def test_below_soft_threshold_is_calm(self) -> None:
        config = HomunculusConfig(
            soft_threshold=100,
            medium_threshold=200,
            hard_threshold=300,
        )
        assert calc_regime(config, 99) == "calm"

    def test_at_soft_threshold_is_warning(self) -> None:
        config = HomunculusConfig(
            soft_threshold=100,
            medium_threshold=200,
            hard_threshold=300,
        )
        assert calc_regime(config, 100) == "warning"

    def test_above_soft_but_below_medium_is_warning(self) -> None:
        config = HomunculusConfig(
            soft_threshold=100,
            medium_threshold=200,
            hard_threshold=300,
        )
        assert calc_regime(config, 199) == "warning"

    def test_at_medium_threshold_is_forced(self) -> None:
        config = HomunculusConfig(
            soft_threshold=100,
            medium_threshold=200,
            hard_threshold=300,
        )
        assert calc_regime(config, 200) == "forced"

    def test_above_medium_but_below_hard_is_forced(self) -> None:
        config = HomunculusConfig(
            soft_threshold=100,
            medium_threshold=200,
            hard_threshold=300,
        )
        assert calc_regime(config, 299) == "forced"

    def test_at_hard_threshold_is_critical(self) -> None:
        config = HomunculusConfig(
            soft_threshold=100,
            medium_threshold=200,
            hard_threshold=300,
        )
        assert calc_regime(config, 300) == "critical"

    def test_above_hard_threshold_is_critical(self) -> None:
        config = HomunculusConfig(
            soft_threshold=100,
            medium_threshold=200,
            hard_threshold=300,
        )
        assert calc_regime(config, 999) == "critical"


class TestGasGaugeObserve:
    def test_first_observation_emits_gas_gauge_footer(self, tmp_path: Path) -> None:
        footer_c = _make_footer_controller(tmp_path)
        gauge = GasGauge(config=HomunculusConfig(), footer_c=footer_c)

        gauge.observe(12_000)

        pending = footer_c.peek_pending()
        assert len(pending) == 1
        footer = pending[0]
        assert footer.type == "gas_gauge"
        assert footer.source == "telemetry"
        assert footer.key == "gas_gauge"
        assert footer.priority == 10
        assert footer.text == "[context: 12K / 1M - calm]"

    def test_same_bucket_does_not_emit_duplicate(self, tmp_path: Path) -> None:
        footer_c = _make_footer_controller(tmp_path)
        gauge = GasGauge(config=HomunculusConfig(), footer_c=footer_c)

        gauge.observe(10_000)
        first = footer_c.peek_pending()
        gauge.observe(49_999)
        second = footer_c.peek_pending()

        assert len(first) == 1
        assert len(second) == 1
        assert second[0].text == "[context: 10K / 1M - calm]"
        assert gauge.input_tokens == 49_999
        assert gauge.regime == "calm"

    def test_new_bucket_replaces_existing_gas_gauge_footer(self, tmp_path: Path) -> None:
        footer_c = _make_footer_controller(tmp_path)
        gauge = GasGauge(config=HomunculusConfig(), footer_c=footer_c)

        gauge.observe(10_000)
        gauge.observe(50_000)

        pending = footer_c.peek_pending()
        assert len(pending) == 1
        footer = pending[0]
        assert footer.key == "gas_gauge"
        assert footer.text == "[context: 50K / 1M - calm]"

    def test_invalidate_hides_tokens_and_clears_pending_footer(self, tmp_path: Path) -> None:
        footer_c = _make_footer_controller(tmp_path)
        gauge = GasGauge(config=HomunculusConfig(), footer_c=footer_c)

        gauge.observe(10_000)
        assert footer_c.peek_pending()

        gauge.invalidate()

        assert gauge.input_tokens is None
        assert footer_c.peek_pending() == []

        gauge.observe(20_000)

        pending = footer_c.peek_pending()
        assert len(pending) == 1
        assert pending[0].text == "[context: 20K / 1M - calm]"

    def test_warning_regime_appears_in_footer_text(self, tmp_path: Path) -> None:
        footer_c = _make_footer_controller(tmp_path)
        config = HomunculusConfig(
            soft_threshold=100_000,
            medium_threshold=200_000,
            hard_threshold=300_000,
        )
        gauge = GasGauge(config=config, footer_c=footer_c)

        gauge.observe(120_000)

        pending = footer_c.peek_pending()
        assert len(pending) == 1
        assert pending[0].text == "[context: 120K / 1M - warning]"

    def test_forced_regime_appears_in_footer_text(self, tmp_path: Path) -> None:
        footer_c = _make_footer_controller(tmp_path)
        config = HomunculusConfig(
            soft_threshold=100_000,
            medium_threshold=200_000,
            hard_threshold=300_000,
        )
        gauge = GasGauge(config=config, footer_c=footer_c)

        gauge.observe(220_000)

        pending = footer_c.peek_pending()
        assert len(pending) == 1
        assert pending[0].text == "[context: 220K / 1M - forced]"

    def test_critical_regime_appears_in_footer_text(self, tmp_path: Path) -> None:
        footer_c = _make_footer_controller(tmp_path)
        config = HomunculusConfig(
            soft_threshold=100_000,
            medium_threshold=200_000,
            hard_threshold=300_000,
        )
        gauge = GasGauge(config=config, footer_c=footer_c)

        gauge.observe(320_000)

        pending = footer_c.peek_pending()
        assert len(pending) == 1
        assert pending[0].text == "[context: 320K / 1M - critical]"
