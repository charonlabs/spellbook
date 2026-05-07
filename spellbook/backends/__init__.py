"""Backend factory helpers."""

from __future__ import annotations

from spellbook.config import Provider, SpellbookConfig

from .anthropic import AnthropicBackend
from .model_backend import ModelBackend
from .openai import OpenAIBackend


def infer_provider_for_model(model: str) -> Provider:
    """Infer a provider from a model slug."""
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gpt-"):
        return "openai"
    raise ValueError(f"Could not infer provider for model slug: {model!r}")


def build_backend(config: SpellbookConfig) -> ModelBackend:
    """Construct the model backend for a Spellbook config."""
    match config.provider:
        case "anthropic":
            return AnthropicBackend()
        case "openai":
            return OpenAIBackend()
        case _:
            raise NotImplementedError(f"{config.provider} is not a supported provider.")
