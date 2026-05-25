"""Flask web GUI for chatting with 西比莉娜.

Run via `python run_web.py` (see project root). The server keeps a single
shared CharacterEngine in memory once an API key is provided either via the
environment, ~/.lina_key, or the "/api/auth" endpoint at runtime.
"""

from __future__ import annotations

import threading
from pathlib import Path

import json

from flask import Flask, Response, jsonify, render_template, request

from . import character as character_mod
from .character import CharacterEngine, DEFAULT_MODEL
from .config import CONVERSATIONS_DIR, PROJECT_ROOT, STATIC_DIR, resolve_api_key
from .conversation import ConversationStore


_engine: CharacterEngine | None = None
_engine_lock = threading.Lock()
_store = ConversationStore(CONVERSATIONS_DIR)

# ---------- Prompt-override layer ----------
# Editable from the web UI. Stored locally (gitignored), reapplied on
# engine creation and on every override edit.

PROMPT_COMPONENTS: list[tuple[str, str, str, str]] = [
    # (key, label, source, hint)
    ("person_setup.md", "角色核心设定", "file", "总是放进系统提示。最重要的身份信息。"),
    ("world.md", "世界观设定", "file", "总是放进系统提示。决定知识边界。"),
    ("sample_conversations.md", "示例对话", "file", "总是放进系统提示。她说话的腔调来源。"),
    ("personality.md", "人格问卷", "file", "RAG 检索：与用户话题相关时才出现。"),
    ("hobbies.md", "兴趣偏好", "file", "RAG 检索。"),
    ("others.md", "其他角色", "file", "RAG 检索。"),
    ("BEHAVIOR_RULES", "行为规则", "code", "代码常量。系统提示里的核心约束。"),
    ("MOOD_FORMAT_SPEC", "情绪标记格式", "code", "代码常量。决定 [mood: …] 的输出格式。"),
    ("SYSTEM_PROMPT_TEMPLATE", "系统提示模板", "code",
     "代码常量。包含占位符 {core_text} / {behavior_rules} / {mood_format_spec}。"),
]
_PROMPT_KEYS = {k for k, _, _, _ in PROMPT_COMPONENTS}

OVERRIDES_DIR = PROJECT_ROOT / "prompt_overrides"
OVERRIDES_FILE = OVERRIDES_DIR / "current.json"

_overrides: dict[str, str] = {}


def _load_overrides_from_disk() -> dict[str, str]:
    if not OVERRIDES_FILE.exists():
        return {}
    try:
        data = json.loads(OVERRIDES_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k in _PROMPT_KEYS and isinstance(v, str)}
    except Exception:
        pass
    return {}


def _save_overrides_to_disk(d: dict[str, str]) -> None:
    OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    OVERRIDES_FILE.write_text(
        json.dumps(d, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _default_value(key: str) -> str:
    if key.endswith(".md"):
        fpath = STATIC_DIR / key
        return fpath.read_text(encoding="utf-8") if fpath.exists() else ""
    return {
        "BEHAVIOR_RULES": character_mod.BEHAVIOR_RULES,
        "MOOD_FORMAT_SPEC": character_mod.MOOD_FORMAT_SPEC,
        "SYSTEM_PROMPT_TEMPLATE": character_mod.SYSTEM_PROMPT_TEMPLATE,
    }.get(key, "")


def _get_engine() -> CharacterEngine | None:
    return _engine


def _init_engine(api_key: str, model: str = DEFAULT_MODEL) -> CharacterEngine:
    global _engine
    with _engine_lock:
        _engine = CharacterEngine(
            api_key=api_key,
            static_dir=STATIC_DIR,
            model=model,
            overrides=_overrides,
        )
    return _engine


def _reapply_overrides_to_engine() -> None:
    if _engine is not None:
        with _engine_lock:
            _engine.apply_overrides(_overrides)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static_web"),
    )

    # Load any persisted overrides from the gitignored local folder.
    global _overrides
    _overrides = _load_overrides_from_disk()

    # Eager-init if we already have a key on disk / env.
    existing_key = resolve_api_key()
    if existing_key:
        try:
            _init_engine(existing_key)
        except Exception as e:
            app.logger.warning(f"Auto-init failed: {e}")

    @app.route("/")
    def index():
        return render_template("chat.html")

    @app.route("/api/status")
    def status():
        engine = _get_engine()
        return jsonify(
            {
                "ready": engine is not None,
                "model": engine.model if engine else None,
                "default_model": DEFAULT_MODEL,
            }
        )

    @app.route("/api/auth", methods=["POST"])
    def auth():
        data = request.get_json(force=True, silent=True) or {}
        key = (data.get("api_key") or "").strip()
        model = (data.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        if not key:
            return jsonify({"ok": False, "error": "缺少 api_key"}), 400
        try:
            _init_engine(key, model=model)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, "model": model})

    @app.route("/api/sessions", methods=["GET"])
    def list_sessions():
        return jsonify({"sessions": _store.list_sessions()})

    @app.route("/api/sessions", methods=["POST"])
    def new_session():
        data = request.get_json(force=True, silent=True) or {}
        requested = (data.get("session_id") or "").strip() or None
        conv = _store.new_session(requested)
        return jsonify({"session_id": conv.session_id, "messages": []})

    @app.route("/api/sessions/<session_id>", methods=["GET"])
    def get_session(session_id: str):
        conv = _store.load(session_id)
        return jsonify(
            {
                "session_id": conv.session_id,
                "title": conv.title,
                "messages": [m.to_dict() for m in conv.messages],
            }
        )

    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    def delete_session(session_id: str):
        ok = _store.delete(session_id)
        return jsonify({"ok": ok})

    @app.route("/api/sessions/<session_id>/export", methods=["GET"])
    def export_session(session_id: str):
        conv = _store.load(session_id)
        payload = json.dumps(conv.to_dict(), ensure_ascii=False, indent=2)
        return Response(
            payload,
            mimetype="application/json; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{conv.session_id}.json"'
                ),
            },
        )

    @app.route("/api/sessions/<session_id>/reset", methods=["POST"])
    def reset_session(session_id: str):
        conv = _store.load(session_id)
        conv.messages.clear()
        conv.title = ""
        _store.save(conv)
        return jsonify({"ok": True})

    @app.route("/api/chat", methods=["POST"])
    def chat():
        engine = _get_engine()
        if engine is None:
            return jsonify({"ok": False, "error": "请先在右上角输入 Anthropic API Key。"}), 401

        data = request.get_json(force=True, silent=True) or {}
        session_id = (data.get("session_id") or "").strip()
        message = (data.get("message") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        if not message:
            return jsonify({"ok": False, "error": "消息为空"}), 400

        conv = _store.load(session_id)
        try:
            result = engine.chat(conv, message)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        _store.save(conv)

        return jsonify(
            {
                "ok": True,
                "reply": result.text,
                "mood": result.mood,
                "retrieved": [
                    {"source": c.source, "heading": c.heading, "text": c.text}
                    for c in result.retrieved
                ],
                "retrieved_history": [
                    {"source": c.source, "heading": c.heading, "text": c.text}
                    for c in result.retrieved_history
                ],
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cache_creation_input_tokens": result.cache_creation_tokens,
                    "cache_read_input_tokens": result.cache_read_tokens,
                },
            }
        )

    # ---------- Prompt override endpoints ----------

    @app.route("/api/prompts", methods=["GET"])
    def list_prompts():
        items = []
        for key, label, source, hint in PROMPT_COMPONENTS:
            default = _default_value(key)
            overridden = key in _overrides
            current = _overrides[key] if overridden else default
            items.append(
                {
                    "key": key,
                    "label": label,
                    "source": source,
                    "hint": hint,
                    "overridden": overridden,
                    "current": current,
                    "default": default,
                }
            )
        return jsonify({"components": items, "override_count": len(_overrides)})

    @app.route("/api/prompts/<path:key>", methods=["PUT"])
    def set_prompt(key: str):
        if key not in _PROMPT_KEYS:
            return jsonify({"ok": False, "error": f"未知组件: {key}"}), 400
        data = request.get_json(force=True, silent=True) or {}
        content = data.get("content")
        if not isinstance(content, str):
            return jsonify({"ok": False, "error": "content 必须是字符串"}), 400
        # If the user submits the exact default, treat as a reset (cleaner UX).
        if content == _default_value(key):
            _overrides.pop(key, None)
        else:
            _overrides[key] = content
        _save_overrides_to_disk(_overrides)
        _reapply_overrides_to_engine()
        return jsonify({"ok": True, "overridden": key in _overrides})

    @app.route("/api/prompts/<path:key>", methods=["DELETE"])
    def reset_prompt(key: str):
        if key not in _PROMPT_KEYS:
            return jsonify({"ok": False, "error": f"未知组件: {key}"}), 400
        _overrides.pop(key, None)
        _save_overrides_to_disk(_overrides)
        _reapply_overrides_to_engine()
        return jsonify({"ok": True})

    @app.route("/api/prompts/reset-all", methods=["POST"])
    def reset_all_prompts():
        _overrides.clear()
        _save_overrides_to_disk(_overrides)
        _reapply_overrides_to_engine()
        return jsonify({"ok": True})

    @app.route("/api/prompts/export", methods=["GET"])
    def export_overrides_endpoint():
        payload = json.dumps(_overrides, ensure_ascii=False, indent=2)
        return Response(
            payload,
            mimetype="application/json; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="prompt_overrides.json"',
            },
        )

    @app.route("/api/prompts/import", methods=["POST"])
    def import_overrides_endpoint():
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "需要 JSON 对象 {key: content, ...}"}), 400
        filtered = {
            k: v for k, v in data.items()
            if k in _PROMPT_KEYS and isinstance(v, str)
        }
        ignored = [k for k in data.keys() if k not in _PROMPT_KEYS]
        _overrides.clear()
        _overrides.update(filtered)
        _save_overrides_to_disk(_overrides)
        _reapply_overrides_to_engine()
        return jsonify({"ok": True, "imported": list(filtered.keys()), "ignored": ignored})

    return app
