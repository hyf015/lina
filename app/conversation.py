"""Conversation history persistence — one JSON file per session."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    ts: float = field(default_factory=time.time)
    # Free-form metadata. For assistant messages this carries the parsed mood
    # tag (mood / intensity / trust). Stored verbatim alongside the message.
    meta: dict | None = None

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content, "ts": self.ts}
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=d["role"],
            content=d["content"],
            ts=d.get("ts", time.time()),
            meta=d.get("meta"),
        )


def _safe_session_id(raw: str | None) -> str:
    if not raw:
        return "sess-" + uuid.uuid4().hex[:8]
    # constrain to filesystem-friendly chars
    cleaned = re.sub(r"[^A-Za-z0-9_\-.]", "_", raw.strip())
    return cleaned[:64] or ("sess-" + uuid.uuid4().hex[:8])


@dataclass
class Conversation:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    title: str = ""
    # If set, points to a saved prompt version's id; the engine for this
    # session will use that version's overrides. None = use the current
    # global overrides (the editable state shown in the prompts tab).
    prompt_version_id: str | None = None

    def add(self, role: str, content: str, meta: dict | None = None) -> Message:
        msg = Message(role=role, content=content, meta=meta)
        self.messages.append(msg)
        if not self.title and role == "user":
            self.title = content[:30]
        return msg

    def last_assistant_meta(self) -> dict | None:
        for m in reversed(self.messages):
            if m.role == "assistant" and m.meta:
                return m.meta
        return None

    def as_api_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "title": self.title,
            "prompt_version_id": self.prompt_version_id,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        return cls(
            session_id=d["session_id"],
            created_at=d.get("created_at", time.time()),
            title=d.get("title", ""),
            prompt_version_id=d.get("prompt_version_id"),
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
        )


class ConversationStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{_safe_session_id(session_id)}.json"

    def new_session(self, requested_id: str | None = None) -> Conversation:
        sid = _safe_session_id(requested_id)
        # Ensure uniqueness if a session id collides with an existing file.
        if self._path(sid).exists() and not requested_id:
            sid = "sess-" + uuid.uuid4().hex[:8]
        conv = Conversation(session_id=sid)
        self.save(conv)
        return conv

    def load(self, session_id: str) -> Conversation:
        p = self._path(session_id)
        if not p.exists():
            return self.new_session(session_id)
        data = json.loads(p.read_text(encoding="utf-8"))
        return Conversation.from_dict(data)

    def save(self, conv: Conversation) -> None:
        self._path(conv.session_id).write_text(
            json.dumps(conv.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def delete(self, session_id: str) -> bool:
        p = self._path(session_id)
        if p.exists():
            p.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict]:
        items: list[dict] = []
        for p in self.root.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                items.append(
                    {
                        "session_id": data.get("session_id", p.stem),
                        "title": data.get("title", "") or "（无标题）",
                        "created_at": data.get("created_at", p.stat().st_mtime),
                        "message_count": len(data.get("messages", [])),
                        "prompt_version_id": data.get("prompt_version_id"),
                    }
                )
            except Exception:
                continue
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items

    def iter_sessions(self) -> Iterator[Conversation]:
        for p in self.root.glob("*.json"):
            try:
                yield Conversation.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
