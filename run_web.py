#!/usr/bin/env python3
"""Launch the web GUI.

Examples:
    python run_web.py
    python run_web.py --port 8080
    ANTHROPIC_API_KEY=sk-... python run_web.py
"""

import argparse

from app.web import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Web GUI for chatting with 西比莉娜.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    print(f"\n  → 打开浏览器访问 http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
