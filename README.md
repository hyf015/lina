# 西比莉娜 · Lina

An AI character — 18-year-old alchemist's apprentice in a pre-industrial world with magic — running on the Claude API, with both a CLI and a web GUI for testing.

The character's personality, world, hobbies, and sample conversations are stored as plain markdown in `static/`. Edit those files and restart to reshape the character.

## Highlights

- **Strict in-character**: behavior rules in the system prompt enforce no AI self-reference, no post-1760 knowledge, no breaking the fourth wall.
- **Prompt caching**: the large character corpus is sent once and cached at the API level for cheap subsequent turns.
- **RAG**: a small BM25 index (character-bigram tokenization, no extra deps) retrieves relevant slices of `personality.md` / `hobbies.md` / `others.md` / `sample_conversations.md` based on each user turn.
- **Persistent history**: each session is stored as a JSON file in `conversations/`.

## Files

```
static/                    # character data — edit these to reshape her
  person_setup.md          ← always in system prompt (core identity)
  world.md                 ← always in system prompt (world setting)
  personality.md           ← RAG-indexed
  hobbies.md               ← RAG-indexed
  others.md                ← RAG-indexed
  sample_conversations.md  ← RAG-indexed

app/
  rag.py                   # BM25 retrieval over static files
  character.py             # system prompt + Claude API call
  conversation.py          # session persistence
  config.py                # API key resolution
  cli.py                   # CLI REPL
  web.py                   # Flask web server
  templates/chat.html      # web GUI

conversations/             # auto-created, holds per-session JSON
run_cli.py
run_web.py
```

## Install

```bash
pip install -r requirements.txt
```

## API key

The app resolves the key in this order:
1. `--api-key` on the command line
2. `ANTHROPIC_API_KEY` environment variable
3. `~/.lina_key` file (single line)
4. Interactive prompt (CLI) / web form (GUI)

## Run — CLI

```bash
python run_cli.py
# or with explicit key / model / session id
python run_cli.py --api-key sk-... --model claude-opus-4-7 --session my-test
```

In-session commands: `/help`, `/new [id]`, `/load <id>`, `/list`, `/reset`, `/history`, `/context`, `/model <name>`, `/quit`.

## Run — Web GUI

```bash
python run_web.py             # http://127.0.0.1:8000
python run_web.py --port 8080
```

The page has a sidebar of past sessions, a chat area, and a right-hand inspector that shows the RAG chunks retrieved for each turn plus token-usage stats (so you can see prompt caching working).

If `ANTHROPIC_API_KEY` is set when the server starts, the engine is ready immediately. Otherwise paste the key into the top-right card and click 连接.

## Models

Default is `claude-sonnet-4-6`. You can switch to `claude-opus-4-7` for higher quality or `claude-haiku-4-5-20251001` for cheaper/faster testing — via the `--model` flag in CLI or the model dropdown in the GUI.

## Tuning the character

Open the files in `static/` and edit. The core identity (`person_setup.md`, `world.md`) is always in the system prompt; everything else is retrieved per-turn — so you can grow `sample_conversations.md` indefinitely without bloating every API call.

The behavior rules (knowledge boundary, AI self-reference ban, speaking style) live in `BEHAVIOR_RULES` inside `app/character.py`.
