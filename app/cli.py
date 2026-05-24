"""Command-line REPL for chatting with 西比莉娜."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from .character import CharacterEngine, DEFAULT_MODEL
from .config import CONVERSATIONS_DIR, STATIC_DIR, resolve_api_key
from .conversation import ConversationStore


HELP_TEXT = """\
命令：
  /help            显示这条帮助
  /new [id]        开启新会话（可选自定义ID）
  /load <id>       加载已有会话
  /list            列出所有会话
  /reset           清空当前会话历史
  /history         打印当前会话的完整历史
  /context         显示上一轮检索到的资料片段
  /export [path]   把当前会话以 JSON 格式导出到指定路径（默认：当前目录/<session_id>.json）
  /model <name>    切换模型（如 claude-opus-4-7、claude-sonnet-4-6、claude-haiku-4-5-20251001）
  /quit, /exit     退出

直接输入文字即可与莉娜对话。
"""

BANNER = """\
========================================
  与西比莉娜的对话  ·  炼金术士学徒
========================================
输入 /help 查看命令，/quit 退出。
"""


def _prompt_api_key() -> str:
    print("未检测到 ANTHROPIC_API_KEY 环境变量。")
    print("（也可以将 key 写入 ~/.lina_key，或用 --api-key 传入。）")
    key = getpass.getpass("请输入 Anthropic API Key（输入不会显示）: ").strip()
    if not key:
        print("未提供 API Key，退出。", file=sys.stderr)
        sys.exit(1)
    return key


def run() -> int:
    parser = argparse.ArgumentParser(description="Chat with 西比莉娜 via Claude API.")
    parser.add_argument("--api-key", help="Anthropic API key. Overrides env / keyfile.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model ID (default: {DEFAULT_MODEL}).")
    parser.add_argument("--session", help="Session ID to load or create. Default: a new auto-named session.")
    parser.add_argument("--static", default=str(STATIC_DIR), help="Path to the static character directory.")
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key) or _prompt_api_key()

    try:
        engine = CharacterEngine(api_key=api_key, static_dir=args.static, model=args.model)
    except Exception as e:
        print(f"初始化失败：{e}", file=sys.stderr)
        return 1

    store = ConversationStore(CONVERSATIONS_DIR)
    conv = store.load(args.session) if args.session else store.new_session()

    last_retrieved: list = []
    last_retrieved_history: list = []

    def _clear_context_cache() -> None:
        last_retrieved.clear()
        last_retrieved_history.clear()

    print(BANNER)
    print(f"会话ID: {conv.session_id}    模型: {engine.model}")
    if conv.messages:
        print(f"(已加载 {len(conv.messages)} 条历史消息)")
    print()

    while True:
        try:
            user_input = input("你 ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            store.save(conv)
            return 0

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, _, rest = user_input.partition(" ")
            cmd = cmd.lower()
            rest = rest.strip()

            if cmd in ("/quit", "/exit"):
                store.save(conv)
                print("再见。")
                return 0
            if cmd == "/help":
                print(HELP_TEXT)
                continue
            if cmd == "/new":
                store.save(conv)
                conv = store.new_session(rest or None)
                _clear_context_cache()
                print(f"已开启新会话: {conv.session_id}\n")
                continue
            if cmd == "/load":
                if not rest:
                    print("用法：/load <session_id>")
                    continue
                store.save(conv)
                conv = store.load(rest)
                _clear_context_cache()
                print(f"已加载会话: {conv.session_id} ({len(conv.messages)} 条消息)\n")
                continue
            if cmd == "/list":
                sessions = store.list_sessions()
                if not sessions:
                    print("（暂无会话）\n")
                else:
                    for s in sessions:
                        print(f"  {s['session_id']:30s}  {s['message_count']:>3} 条  {s['title']}")
                    print()
                continue
            if cmd == "/reset":
                conv.messages.clear()
                conv.title = ""
                store.save(conv)
                _clear_context_cache()
                print("当前会话历史已清空。\n")
                continue
            if cmd == "/history":
                if not conv.messages:
                    print("（当前会话尚无消息）\n")
                else:
                    for m in conv.messages:
                        prefix = "你" if m.role == "user" else "莉娜"
                        suffix = ""
                        if m.role == "assistant" and m.meta:
                            suffix = (
                                f"  〔{m.meta.get('mood', '?')}·"
                                f"{m.meta.get('intensity', '?')} | "
                                f"信任 {m.meta.get('trust', '?')}〕"
                            )
                        print(f"[{prefix}] {m.content}{suffix}")
                    print()
                continue
            if cmd == "/context":
                if not last_retrieved and not last_retrieved_history:
                    print("（尚无检索记录）\n")
                else:
                    if last_retrieved:
                        print("--- 上一轮检索到的角色设定片段 ---")
                        for c in last_retrieved:
                            print(c.render())
                            print()
                    if last_retrieved_history:
                        print("--- 上一轮检索到的历史记忆 ---")
                        for c in last_retrieved_history:
                            print(c.render())
                            print()
                continue
            if cmd == "/export":
                target = Path(rest).expanduser() if rest else Path.cwd() / f"{conv.session_id}.json"
                if target.is_dir():
                    target = target / f"{conv.session_id}.json"
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(
                        json.dumps(conv.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"已导出到：{target}\n")
                except OSError as e:
                    print(f"导出失败：{e}\n", file=sys.stderr)
                continue
            if cmd == "/model":
                if not rest:
                    print(f"当前模型: {engine.model}")
                else:
                    engine.model = rest
                    print(f"模型已切换为: {rest}")
                continue
            print(f"未知命令: {cmd}（输入 /help 查看帮助）\n")
            continue

        # Normal message
        try:
            result = engine.chat(conv, user_input)
        except Exception as e:
            print(f"[请求出错] {e}\n", file=sys.stderr)
            continue

        last_retrieved[:] = result.retrieved
        last_retrieved_history[:] = result.retrieved_history
        if result.mood:
            mood_str = (
                f"  〔{result.mood.get('mood', '?')}·"
                f"{result.mood.get('intensity', '?')} | "
                f"信任 {result.mood.get('trust', '?')}/10〕"
            )
        else:
            mood_str = "  〔mood: -〕"
        print(f"莉娜 ▸ {result.text}{mood_str}\n")
        store.save(conv)


if __name__ == "__main__":
    sys.exit(run())
