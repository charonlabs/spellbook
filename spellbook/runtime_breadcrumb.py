import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimeBreadcrumb:
    package_path: str
    git_sha: str
    git_state: str


def spellbook_runtime_breadcrumb() -> RuntimeBreadcrumb:
    package_path = Path(__file__).resolve().parent
    repo_root = package_path.parent
    git_sha = _git_output(repo_root, ["rev-parse", "--short", "HEAD"]) or "unknown"
    git_status = _git_output(repo_root, ["status", "--short"])
    if git_status is None:
        git_state = "unknown"
    elif git_status:
        git_state = "dirty"
    else:
        git_state = "clean"
    return RuntimeBreadcrumb(
        package_path=str(package_path),
        git_sha=git_sha,
        git_state=git_state,
    )


def _git_output(repo_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
