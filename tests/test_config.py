import pytest
from pydantic import ValidationError

from assistant.config import Settings


def make_settings(**kwargs):
    """Instantiate Settings with required fields, plus any overrides."""
    defaults = dict(
        llm_api_key="test-key",
        llm_base_url="https://api.example.com/v1",
        llm_model="test-model",
        discord_token="test-token",
        discord_allowed_user_ids=[123456789],
    )
    defaults.update(kwargs)
    return Settings(**defaults)


class TestSettingsDefaults:
    def test_memory_top_k_default(self):
        s = make_settings()
        assert s.memory_top_k == 5

    def test_context_token_limit_default(self):
        s = make_settings()
        assert s.context_token_limit == 8000

    def test_timezone_default(self):
        s = make_settings()
        assert s.timezone == "UTC"

    def test_enabled_skills_default_empty(self):
        s = make_settings()
        assert s.enabled_skills == []

    def test_summarisation_hours_default(self):
        s = make_settings()
        assert s.summarisation_hours == [2]

    def test_embedding_model_default(self):
        s = make_settings()
        assert s.embedding_model == "text-embedding-3-small"

    def test_confirmation_timeout_default(self):
        s = make_settings()
        assert s.confirmation_timeout_seconds == 60

    def test_gmail_poll_interval_default(self):
        s = make_settings()
        assert s.gmail_poll_interval_seconds == 600


class TestSummarisationHoursValidator:
    def test_valid_single_hour(self):
        s = make_settings(summarisation_hours=[3])
        assert s.summarisation_hours == [3]

    def test_valid_multiple_hours(self):
        s = make_settings(summarisation_hours=[0, 12, 23])
        assert s.summarisation_hours == [0, 12, 23]

    def test_boundary_hour_zero(self):
        s = make_settings(summarisation_hours=[0])
        assert s.summarisation_hours == [0]

    def test_boundary_hour_23(self):
        s = make_settings(summarisation_hours=[23])
        assert s.summarisation_hours == [23]

    def test_invalid_hour_too_high(self):
        with pytest.raises(ValidationError):
            make_settings(summarisation_hours=[24])

    def test_invalid_hour_negative(self):
        with pytest.raises(ValidationError):
            make_settings(summarisation_hours=[-1])


class TestCommaSplitMixin:
    """The _CommaSplitMixin is exercised via the Env source when env vars are set."""

    def test_comma_split_enabled_skills(self, monkeypatch):
        monkeypatch.setenv("ENABLED_SKILLS", "calendar,gmail")
        s = make_settings()
        assert s.enabled_skills == ["calendar", "gmail"]

    def test_comma_split_with_spaces(self, monkeypatch):
        monkeypatch.setenv("ENABLED_SKILLS", "calendar, gmail, notes")
        s = make_settings()
        assert s.enabled_skills == ["calendar", "gmail", "notes"]

    def test_comma_split_single_item(self, monkeypatch):
        monkeypatch.setenv("ENABLED_SKILLS", "calendar")
        s = make_settings()
        assert s.enabled_skills == ["calendar"]

    def test_json_array_still_works(self, monkeypatch):
        monkeypatch.setenv("ENABLED_SKILLS", '["calendar", "gmail"]')
        s = make_settings()
        assert s.enabled_skills == ["calendar", "gmail"]

    def test_comma_split_summarisation_hours(self, monkeypatch):
        monkeypatch.setenv("SUMMARISATION_HOURS", "2,14")
        s = make_settings()
        assert s.summarisation_hours == [2, 14]
