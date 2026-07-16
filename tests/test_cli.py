from __future__ import annotations

from pathlib import Path

import pytest

from spellbook import cli
from spellbook.config import SpellbookConfig


def test_positional_path_is_config_by_default() -> None:
    args = cli._parse_args(["serve", "philosopher.toml"])

    config_path, transcript_path = cli._serve_paths(args)

    assert config_path == Path("philosopher.toml")
    assert transcript_path is None


def test_positional_path_is_transcript_with_explicit_config() -> None:
    args = cli._parse_args(
        ["serve", "transcript.jsonl", "--config", "philosopher.toml"]
    )

    config_path, transcript_path = cli._serve_paths(args)

    assert config_path == Path("philosopher.toml")
    assert transcript_path == Path("transcript.jsonl")


def test_explicit_config_and_transcript_reject_extra_positional_path() -> None:
    args = cli._parse_args(
        [
            "serve",
            "extra.jsonl",
            "--config",
            "philosopher.toml",
            "--transcript",
            "transcript.jsonl",
        ]
    )

    with pytest.raises(cli.ServeConfigError, match="not both"):
        cli._serve_paths(args)


def test_run_serve_builds_new_config_and_creates_transcript_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "philosopher.toml"
    config_path.write_text(
        '[entity]\nmodel = "gpt-5.5"\n[prompt]\nrole = "Think deeply."\n',
        encoding="utf-8",
    )
    transcript_path = tmp_path / "sessions" / "transcript.jsonl"
    seen: dict[str, object] = {}

    def _create_app(**kwargs: object) -> object:
        seen.update(kwargs)
        return object()

    def _run(app: object, **kwargs: object) -> None:
        seen["app"] = app
        seen["run_kwargs"] = kwargs

    monkeypatch.setattr(cli, "create_app", _create_app)
    monkeypatch.setattr(cli.uvicorn, "run", _run)
    args = cli._parse_args(
        [
            "serve",
            str(config_path),
            "--transcript",
            str(transcript_path),
            "--port",
            "9000",
            "--env",
            str(tmp_path / "missing.env"),
        ]
    )

    cli._run_serve(args)

    config = seen["config"]
    assert isinstance(config, SpellbookConfig)
    assert config.model == "gpt-5.5"
    assert "Think deeply." in config.system_prompt
    assert seen["transcript_path"] == transcript_path
    assert transcript_path.parent.is_dir()
    assert not transcript_path.exists()
    assert seen["run_kwargs"] == {
        "host": "127.0.0.1",
        "port": 9000,
        "log_level": "info",
    }


def test_run_serve_resumes_transcript_without_replacing_recorded_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text("existing transcript", encoding="utf-8")
    seen: dict[str, object] = {}

    def _create_app(**kwargs: object) -> object:
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "create_app", _create_app)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)
    args = cli._parse_args(
        [
            "serve",
            "--transcript",
            str(transcript_path),
            "--model",
            "gpt-5.5",
            "--env",
            str(tmp_path / "missing.env"),
        ]
    )

    cli._run_serve(args)

    assert seen["transcript_path"] == transcript_path
    assert seen["config"] is None


def test_default_transcript_path_uses_config_stem_and_sessions_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    transcript_path = cli._resolve_transcript_path(
        None, config_path=Path("philosopher.toml")
    )

    assert transcript_path.parent == tmp_path / ".spellbook" / "sessions"
    assert transcript_path.name.startswith("philosopher_")
    assert transcript_path.suffix == ".jsonl"
