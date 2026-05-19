"""Config & API key resolution."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
CONVERSATIONS_DIR = PROJECT_ROOT / "conversations"


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Resolve the API key from (in order): explicit arg, env var, ~/.lina_key file."""
    if explicit:
        return explicit.strip()
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env.strip()
    keyfile = Path.home() / ".lina_key"
    if keyfile.exists():
        try:
            content = keyfile.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception:
            pass
    return None
