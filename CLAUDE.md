# CLAUDE.md

## Project Overview

**alex-a** is a personal AI assistant that runs as a Discord bot. It uses an LLM (via OpenAI-compatible API) with tool calling, semantic memory (LanceDB + embeddings), and a modular skill/plugin system.

## Tech Stack

- **Python 3.13** — managed with **uv** (`pyproject.toml` + `uv.lock`)
- **discord.py** — Discord bot framework (DM-only, privileged Message Content intent)
- **OpenAI SDK** (`openai`) — chat completions + embeddings, configured for any OpenAI-compatible provider
- **LanceDB** — vector store for semantic memory
- **pydantic-settings** — typed configuration from `.env` / environment variables

## Project Structure

```
main.py                      # Entry point: asyncio.run(main())
system_prompts/
├── alex_main.md             # Main system prompt template (loaded via SYSTEM_PROMPT_PATH)
├── alex_informal.md         # Informal/persona writing guide
└── alex_email.md            # Email writing style guide (loaded by get_email_style tool)
src/assistant/
├── config.py                # Settings (pydantic-settings, reads .env)
├── main.py                  # Wires Settings → Platform → Memory → ToolRegistry → Agent
├── core/
│   ├── agent.py             # Agent — agentic message loop with tool calling
│   ├── llm.py               # LLM + embedding client singletons (AsyncOpenAI)
│   ├── memory.py            # Memory — LanceDB-backed semantic memory store
│   └── tool_registry.py     # ToolRegistry — skill registration + tool execution
├── platforms/
│   ├── base.py              # Platform protocol + Message/User dataclasses
│   └── discord_platform.py  # DiscordPlatform — Discord implementation
├── auth/
│   └── google.py            # Shared Google OAuth (single consent flow for all Google skills)
└── skills/
    ├── calendar.py          # Google Calendar skill (list/create/update events)
    └── gmail.py             # Gmail skill (list/search/read/send/reply/trash + background poller)
```

## Key Concepts

### Agent Loop (`core/agent.py`)
The `Agent` class processes inbound messages through an iterative tool-calling loop (max 10 iterations). It builds a system prompt with the user's name, current date, timezone, and relevant memories, then runs LLM completions until a final text response is produced. `safe_process()` wraps `process()` with error handling for production use. `record_proactive(content)` injects unsolicited notifications (e.g. email alerts) into history so the agent can refer back to them.

### Memory (`core/memory.py`)
Semantic memory backed by LanceDB. Memories are embedded with `text-embedding-3-small` (OpenAI) and retrieved via nearest-neighbor search. Includes a `summarise_and_store()` method that uses the LLM to condense conversations before persisting.

### Tool Registry (`core/tool_registry.py`)
Central catalogue for tool handlers. Skills expose a `register(registry)` function; handlers are decorated with `@registry.tool(name, description)`. The registry:
- Auto-generates OpenAI function-calling schemas from Pydantic models
- Supports a `requires_confirmation` flag that gates execution behind a ✅/❌ reaction prompt
- Catches handler exceptions and returns error strings to the LLM instead of crashing

### Platform Protocol (`platforms/base.py`)
`Platform` is a `Protocol` with methods: `start`, `stop`, `send`, `listen`, `get_user`, `request_confirmation`. Currently only `DiscordPlatform` implements it.

### Skills (Plugin System)
Skills live under `assistant.skills.<name>` and are loaded dynamically by `ToolRegistry.load_skills()`. Each must expose a `register(registry: ToolRegistry) -> None` function. Enabled via the `ENABLED_SKILLS` config.

Current skills:
- **`calendar`** — `list_calendar_events`, `create_calendar_event`, `update_calendar_event` via Google Calendar API
- **`gmail`** — `list_emails`, `search_emails`, `read_email`, `send_email`, `reply_email`, `trash_email`, `get_email_style`; write tools require confirmation; includes a `GmailPoller` background task that checks for unread emails on startup (last 24h) and every `GMAIL_POLL_INTERVAL_SECONDS` seconds, notifying via DM

### Google Auth (`auth/google.py`)
Single shared OAuth consent flow covering all Google scopes (`calendar`, `gmail.modify`). Credentials downloaded from Google Cloud Console go to `GOOGLE_CREDENTIALS_PATH`; the resulting user token is cached at `GOOGLE_USER_TOKEN_PATH`. To add new Google scopes, extend `_SCOPES` in `auth/google.py` and delete the cached user token to trigger re-auth.

### System Prompts (`system_prompts/`)
Markdown files loaded at runtime. `alex_main.md` is the main template (with `{user}`, `{date}`, `{timezone}`, `{memories}` placeholders). `alex_email.md` is fetched on demand via the `get_email_style` tool before composing or replying to emails.

## Configuration

All config is in `.env` (see `.env.template`). Key variables:

| Variable | Purpose |
|---|---|
| `LLM_API_KEY` | API key for the LLM provider |
| `LLM_BASE_URL` | OpenAI-compatible API base URL (default: Anthropic) |
| `LLM_MODEL` | Model identifier (e.g. `claude-sonnet-4-6`) |
| `EMBEDDING_MODEL` | Embedding model (default: `text-embedding-3-small`) |
| `EMBEDDING_API_KEY` | Separate key for embeddings; falls back to `LLM_API_KEY` |
| `SYSTEM_PROMPT_PATH` | Path to system prompt template (default: `system_prompts/alex.md`) |
| `TIMEZONE` | IANA timezone injected into system prompt (default: `UTC`) |
| `DISCORD_TOKEN` | Discord bot token |
| `DISCORD_ALLOWED_USER_IDS` | Comma-separated allowlisted user IDs |
| `LANCEDB_PATH` | Path to LanceDB data (default: `data/lancedb`) |
| `GOOGLE_CREDENTIALS_PATH` | Google OAuth client JSON from Cloud Console |
| `GOOGLE_USER_TOKEN_PATH` | Cached user OAuth token (auto-generated) |
| `MEMORY_TOP_K` | Number of memories to retrieve per query |
| `CONTEXT_TOKEN_LIMIT` | Approximate token budget for context window |
| `GMAIL_POLL_INTERVAL_SECONDS` | Email poll frequency in seconds (default: 600) |
| `ENABLED_SKILLS` | Comma-separated skill module names to load |
| `CONFIRMATION_TIMEOUT_SECONDS` | Timeout for tool confirmation reactions |
| `SUMMARISATION_HOURS` | UTC hours for nightly summarisation job |

Comma-separated list fields (e.g. `DISCORD_ALLOWED_USER_IDS`, `ENABLED_SKILLS`) are parsed automatically — no JSON syntax required.

## Commands

```bash
# Install dependencies
uv sync

# Run the bot
uv run python main.py
```

## Architecture Notes

- **LLM-provider agnostic**: Uses the OpenAI SDK pointed at any compatible API (`LLM_BASE_URL`). Embeddings always go to OpenAI directly (separate `EMBEDDING_API_KEY`).
- **DM-only**: The Discord platform ignores guild messages and non-allowlisted users.
- **History management**: The agent maintains an in-memory conversation buffer, trimmed by a rough token estimate (~4 chars/token).
- **Error resilience**: Tool execution errors are caught and returned as strings to the LLM so it can inform the user. The `safe_process()` wrapper catches all top-level exceptions.
- **Proactive notifications**: The Gmail poller sends unsolicited DMs and calls `agent.record_proactive()` so follow-up questions have context. `DiscordPlatform.open_dm_channel(user_id)` opens a DM channel by user ID without waiting for an inbound message.
- **Startup flow** (`assistant/main.py`): Settings → DiscordPlatform → Memory.initialise() → ToolRegistry.load_skills() → Agent → platform.start() → (optional) GmailPoller task → listen loop.
