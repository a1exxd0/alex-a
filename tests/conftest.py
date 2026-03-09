"""Set required environment variables before any assistant.* modules are imported.

The module-level `settings = Settings()` in assistant/config.py (and the
AsyncOpenAI singletons in assistant/core/llm.py) run at import time, so these
env vars must be in place before pytest collects any test file that imports
from the assistant package.
"""
import os

os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("LLM_BASE_URL", "https://api.example.com/v1")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
os.environ.setdefault("DISCORD_ALLOWED_USER_IDS", "111222333")
