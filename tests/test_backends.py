"""Tests for backend factory helpers."""

from __future__ import annotations

from spellbook.backends import build_backend, infer_provider_for_model
from spellbook.backends.anthropic import AnthropicBackend
from spellbook.backends.openai import OpenAIBackend
from spellbook.config import SpellbookConfig


def test_infer_provider_for_model_maps_known_model_families() -> None:
    assert infer_provider_for_model("claude-opus-4-7") == "anthropic"
    assert infer_provider_for_model("gpt-5.5") == "openai"
    assert infer_provider_for_model("gpt-5.5-20260501") == "openai"


def test_build_backend_constructs_provider_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    anthropic = build_backend(
        SpellbookConfig(provider="anthropic", model="claude-sonnet-4-6", cwd=tmp_path)
    )
    openai = build_backend(
        SpellbookConfig(provider="openai", model="gpt-5.5", cwd=tmp_path)
    )

    assert isinstance(anthropic, AnthropicBackend)
    assert isinstance(openai, OpenAIBackend)
