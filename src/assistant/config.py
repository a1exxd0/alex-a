from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import DotEnvSettingsSource, EnvSettingsSource


class _CommaSplitMixin:
    """
    Override prepare_field_value so that complex fields (lists) can be
    supplied as plain comma-separated strings in .env / env vars instead
    of requiring JSON syntax.  JSON arrays still work if someone provides
    them explicitly.
    """

    def prepare_field_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: Any,
        value_is_complex: bool,
    ) -> Any:
        is_complex, _ = self._field_is_complex(field)  # type: ignore[attr-defined]
        if (is_complex or value_is_complex) and isinstance(value, str):
            stripped = value.strip()
            if not (stripped.startswith("[") or stripped.startswith("{")):
                return [part.strip() for part in value.split(",") if part.strip()]
        return super().prepare_field_value(field_name, field, value, value_is_complex)  # type: ignore[misc]


class _CommaSplitEnvSource(_CommaSplitMixin, EnvSettingsSource):
    pass


class _CommaSplitDotEnvSource(_CommaSplitMixin, DotEnvSettingsSource):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # LLM
    llm_api_key: str
    llm_base_url: str
    llm_model: str

    # Embeddings (always OpenAI — set a separate key if your LLM provider differs)
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""  # falls back to llm_api_key if empty

    # System prompt file path
    system_prompt_path: Path = Path("system_prompts/alex.md")

    # Email writing style guide (injected into send/reply tool descriptions)
    email_style_path: Path = Path("system_prompts/alex_email.md")

    # User timezone (IANA format, e.g. "Europe/London")
    timezone: str = "UTC"

    # Discord
    discord_token: str
    discord_allowed_user_ids: list[int]

    # Paths
    lancedb_path: Path = Path("data/lancedb")
    google_credentials_path: Path = Path("data/google/google_token.json")
    google_user_token_path: Path = Path("data/google/user_token.json")

    # Summarisation schedule: comma-separated UTC hours, e.g. "2" or "2,14"
    summarisation_hours: list[int] = [2]

    # Memory / context tuning
    memory_top_k: int = 5
    context_token_limit: int = 8000

    # Gmail poller: how often to check for new unread emails (seconds)
    gmail_poll_interval_seconds: int = 600

    # Confirmation gate
    confirmation_timeout_seconds: int = 60

    # Skills: comma-separated module names, e.g. "calendar,email"
    enabled_skills: list[str] = []

    @field_validator("summarisation_hours")
    @classmethod
    def _validate_hours(cls, v: list[int]) -> list[int]:
        for h in v:
            if not 0 <= h < 24:
                raise ValueError(f"Hour {h} is out of range — must be 0–23 UTC")
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _CommaSplitEnvSource(settings_cls),
            _CommaSplitDotEnvSource(settings_cls),
            file_secret_settings,
        )


settings = Settings()
