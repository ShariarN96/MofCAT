"""Load API keys from a local .env file.

Environment variables set with `$env:VAR = ...` in one shell are invisible to
any other process, which makes them awkward for a pipeline launched from
different terminals or from an editor. A .env file at the project root works
regardless of how the pipeline is started.

Existing environment variables always win, so an exported key overrides .env.
"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import ENV_PATH, PROJECT_ROOT  # noqa: F401  (PROJECT_ROOT re-exported)

_loaded = False


def load_env(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Read KEY=VALUE lines into os.environ. Safe to call repeatedly."""
    global _loaded
    env_path = path or ENV_PATH
    applied: dict[str, str] = {}

    if not env_path.exists():
        _loaded = True
        return applied

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key or not value:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value

    _loaded = True
    return applied


def ensure_loaded() -> None:
    """Load .env once per process, before any key lookup."""
    if not _loaded:
        load_env()


def key_status(var: str = "OPENAI_API_KEY") -> str:
    """Human-readable status that never reveals the key itself."""
    ensure_loaded()
    value = os.environ.get(var)
    if not value:
        return f"{var}: not set"
    return f"{var}: set ({len(value)} chars, ends ...{value[-4:]})"
