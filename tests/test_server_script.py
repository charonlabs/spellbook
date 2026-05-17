from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import pytest

from scripts import server


def test_server_prompt_orders_core_text_file_and_claude_md(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    (cwd / "CLAUDE.md").write_text(
        "# Workspace Note\nWorkspace instruction.",
        encoding="utf-8",
    )
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("# File Note\nFile instruction.", encoding="utf-8")

    args = server._parse_args(
        [
            "--model",
            "claude-sonnet-4-6",
            "--cwd",
            str(cwd),
            "--system-prompt-text",
            "# Runtime Note\nRuntime instruction.",
            "--system-prompt-file",
            str(prompt_file),
        ]
    )

    prompt = server._system_prompt_from_args(args)

    assert prompt.index("Claude Sonnet 4.6 entity") < prompt.index(
        "Runtime instruction."
    )
    assert prompt.index("Runtime instruction.") < prompt.index("File instruction.")
    assert prompt.index("File instruction.") < prompt.index("Workspace instruction.")


def test_server_prompt_can_disable_claude_md_discovery(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    (cwd / "CLAUDE.md").write_text(
        "# Workspace Note\nWorkspace instruction.",
        encoding="utf-8",
    )

    args = server._parse_args(
        [
            "--model",
            "claude-sonnet-4-6",
            "--cwd",
            str(cwd),
            "--system-prompt-text",
            "# Runtime Note\nRuntime instruction.",
            "--no-discover-claude-md",
        ]
    )

    prompt = server._system_prompt_from_args(args)

    assert "Runtime instruction." in prompt
    assert "Workspace instruction." not in prompt


def test_server_prompt_skips_claude_md_for_openai_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    (cwd / "CLAUDE.md").write_text(
        "# Workspace Note\nWorkspace instruction.",
        encoding="utf-8",
    )

    args = server._parse_args(
        [
            "--model",
            "gpt-5.5",
            "--cwd",
            str(cwd),
            "--system-prompt-text",
            "# Runtime Note\nRuntime instruction.",
        ]
    )

    prompt = server._system_prompt_from_args(args)

    assert "GPT-5.5 entity" in prompt
    assert "Runtime instruction." in prompt
    assert "Workspace instruction." not in prompt


def test_server_prompt_renders_claude_4_7_orientation(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    (cwd / ".git").mkdir()

    prompt = server._build_system_prompt("claude-opus-4-7", cwd=cwd, user_name="Ryan")

    assert "Claude Opus 4.7 entity" in prompt
    assert "Hello. I'm a 4.7 entity" in prompt
    assert f"Primary working directory: {cwd}" in prompt
    assert "Is a git repository: true" in prompt
    assert "{model_name}" not in prompt


def test_server_prompt_renders_claude_4_6_orientation(tmp_path: Path) -> None:
    prompt = server._build_system_prompt(
        "claude-sonnet-4-6", cwd=tmp_path, user_name="Ryan"
    )

    assert "Claude Sonnet 4.6 entity" in prompt
    assert "This is a safe place. Be yourself." in prompt
    assert "Is a git repository: false" in prompt
    assert "{model}" not in prompt


def test_server_prompt_renders_gpt_5_5_orientation(tmp_path: Path) -> None:
    prompt = server._build_system_prompt("gpt-5.5", cwd=tmp_path, user_name="Ryan")

    assert "GPT-5.5 entity" in prompt
    assert "Read it as one GPT-5.5" in prompt
    assert "## How to work with Ryan" in prompt
    assert "{model_name}" not in prompt


def test_config_from_args_infers_openai_provider_for_gpt_model(tmp_path: Path) -> None:
    args = server._parse_args(
        [
            "--model",
            "gpt-5.5",
            "--cwd",
            str(tmp_path),
        ]
    )

    config = server._config_from_args(args)

    assert config.provider == "openai"
    assert config.model == "gpt-5.5"


def test_model_is_optional_when_resuming_existing_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")

    args = server._parse_args([str(transcript)])

    assert args.model is None
    assert server._resolve_transcript_path(args) == transcript.resolve()


def test_model_is_required_when_initializing_new_transcript(tmp_path: Path) -> None:
    with redirect_stderr(StringIO()), pytest.raises(SystemExit):
        server._parse_args([str(tmp_path / "missing.jsonl")])


def test_main_resumes_existing_transcript_without_model(
    tmp_path: Path, monkeypatch
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    seen: dict[str, object] = {}

    def _create_app(**kwargs):
        seen.update(kwargs)
        return object()

    def _run(app, **kwargs) -> None:
        seen["app"] = app
        seen["run_kwargs"] = kwargs

    monkeypatch.setattr(server, "create_app", _create_app)
    monkeypatch.setattr(server.uvicorn, "run", _run)

    server.main([str(transcript), "--env", str(tmp_path / "missing.env")])

    assert seen["transcript_path"] == transcript.resolve()
    assert seen["config"] is None
    assert seen["log_level"] == "info"
