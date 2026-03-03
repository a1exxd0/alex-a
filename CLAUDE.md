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
src/assistant/
├── config.py                # Settings (pydantic-settings, reads .env)
├── core/
│   ├── agent.py             # Agent — agentic message loop with tool calling
│   ├── llm.py               # LLM + embedding client singletons (AsyncOpenAI)
│   ├── memory.py            # Memory — LanceDB-backed semantic memory store
│   └── tool_registry.py     # ToolRegistry — skill registration + tool execution
├── platforms/
│   ├── base.py              # Platform protocol + Message/User dataclasses
│   └── discord_platform.py  # DiscordPlatform — Discord implementation
├── skills/                  # Skill modules (plugin system, currently empty)
└── auth/                    # Auth modules (currently empty)
```

## Key Concepts

### Agent Loop (`core/agent.py`)
The `Agent` class processes inbound messages through an iterative tool-calling loop (max 10 iterations). It builds a system prompt with the user's name, current date, and relevant memories, then streams through LLM completions until a final text response is produced. `safe_process()` wraps `process()` with error handling for production use.

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

## Configuration

All config is in `.env` (see `.env.template`). Key variables:

| Variable | Purpose |
|---|---|
| `LLM_API_KEY` | API key for the LLM provider |
| `LLM_BASE_URL` | OpenAI-compatible API base URL |
| `LLM_MODEL` | Model identifier (e.g. `claude-sonnet-4-6`) |
| `DISCORD_TOKEN` | Discord bot token |
| `DISCORD_ALLOWED_USER_IDS` | Comma-separated allowlisted user IDs |
| `LANCEDB_PATH` | Path to LanceDB data (default: `data/lancedb`) |
| `MEMORY_TOP_K` | Number of memories to retrieve per query |
| `CONTEXT_TOKEN_LIMIT` | Approximate token budget for context window |
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

- **LLM-provider agnostic**: Uses the OpenAI SDK pointed at any compatible API (`LLM_BASE_URL`). Embeddings always go to OpenAI directly.
- **DM-only**: The Discord platform ignores guild messages and non-allowlisted users.
- **History management**: The agent maintains an in-memory conversation buffer, trimmed by a rough token estimate (~4 chars/token).
- **Error resilience**: Tool execution errors are caught and returned as strings to the LLM so it can inform the user. The `safe_process()` wrapper catches all top-level exceptions.
- **No `assistant/main.py` yet**: The root `main.py` imports `from assistant.main import main` but this module hasn't been created — it will wire together Settings → Platform → Memory → Agent.
