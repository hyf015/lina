"""Flask web GUI for chatting with 西比莉娜.

Run via `python run_web.py` (see project root). The server keeps a single
shared CharacterEngine in memory once an API key is provided either via the
environment, ~/.lina_key, or the "/api/auth" endpoint at runtime.
"""

from __future__ import annotations

import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .character import CharacterEngine, DEFAULT_MODEL
from .config import CONVERSATIONS_DIR, STATIC_DIR, resolve_api_key
from .conversation import ConversationStore


_engine: CharacterEngine | None = None
_engine_lock = threading.Lock()
_store = ConversationStore(CONVERSATIONS_DIR)


def _get_engine() -> CharacterEngine | None:
    return _engine


def _init_engine(api_key: str, model: str = DEFAULT_MODEL) -> CharacterEngine:
    global _engine
    with _engine_lock:
        _engine = CharacterEngine(api_key=api_key, static_dir=STATIC_DIR, model=model)
    return _engine


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static_web"),
    )

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

    return app
