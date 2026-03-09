from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.core.agent import Agent
from assistant.platforms.base import Message, User


def make_agent(token_limit: int = 1000) -> Agent:
    """Build an Agent wired to mock collaborators."""
    settings = MagicMock()
    settings.llm_model = "test-model"
    settings.context_token_limit = token_limit
    settings.system_prompt_path.read_text.return_value = (
        "Hi {user}, date={date}, tz={timezone}\n{memories}"
    )
    settings.timezone = "UTC"

    platform = AsyncMock()
    platform.get_user.return_value = User(id=1, display_name="TestUser")

    memory = AsyncMock()
    memory.retrieve.return_value = []

    return Agent(settings, platform, memory)


def make_message(content: str = "hello", channel_id: str = "ch1") -> Message:
    return Message(content=content, author_id=1, channel_id=channel_id)


# ── _build_system_prompt ──────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    def test_includes_display_name(self):
        agent = make_agent()
        result = agent._build_system_prompt("Alice", [])
        assert "Alice" in result

    def test_includes_timezone(self):
        agent = make_agent()
        result = agent._build_system_prompt("Bob", [])
        assert "UTC" in result

    def test_memories_block_when_provided(self):
        agent = make_agent()
        result = agent._build_system_prompt("Bob", ["Likes coffee", "Has a cat"])
        assert "Relevant memories:" in result
        assert "- Likes coffee" in result
        assert "- Has a cat" in result

    def test_no_memories_block_when_empty(self):
        agent = make_agent()
        result = agent._build_system_prompt("Bob", [])
        assert "Relevant memories:" not in result

    def test_output_is_stripped(self):
        agent = make_agent()
        result = agent._build_system_prompt("Bob", [])
        assert result == result.strip()


# ── _estimate_tokens ──────────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_empty_history(self):
        agent = make_agent()
        assert agent._estimate_tokens() == 0

    def test_single_message(self):
        agent = make_agent()
        agent._history = [{"role": "user", "content": "abcd"}]  # 4 chars → 1 token
        assert agent._estimate_tokens() == 1

    def test_multiple_messages(self):
        agent = make_agent()
        agent._history = [
            {"role": "user", "content": "a" * 8},
            {"role": "assistant", "content": "b" * 4},
        ]
        assert agent._estimate_tokens() == 3  # (8 + 4) // 4

    def test_message_with_none_content(self):
        agent = make_agent()
        agent._history = [{"role": "assistant", "content": None}]
        assert agent._estimate_tokens() == 0


# ── _trim_history ─────────────────────────────────────────────────────────────

class TestTrimHistory:
    def test_no_trim_when_under_limit(self):
        agent = make_agent(token_limit=1000)
        agent._history = [{"role": "user", "content": "short"}]
        agent._trim_history()
        assert len(agent._history) == 1

    def test_trims_oldest_when_over_limit(self):
        agent = make_agent(token_limit=1)  # very small limit
        agent._history = [
            {"role": "user", "content": "a" * 100},
            {"role": "assistant", "content": "b" * 100},
            {"role": "user", "content": "c" * 100},
        ]
        before = len(agent._history)
        agent._trim_history()
        assert len(agent._history) < before

    def test_never_trims_below_two_messages(self):
        agent = make_agent(token_limit=0)  # impossibly small limit
        agent._history = [
            {"role": "user", "content": "a" * 1000},
            {"role": "assistant", "content": "b" * 1000},
        ]
        agent._trim_history()
        assert len(agent._history) == 2

    def test_no_op_on_empty_history(self):
        agent = make_agent(token_limit=0)
        agent._trim_history()
        assert agent._history == []


# ── record_proactive ──────────────────────────────────────────────────────────

class TestRecordProactive:
    def test_appends_assistant_message(self):
        agent = make_agent()
        agent.record_proactive("You have new email!")
        assert agent._history == [{"role": "assistant", "content": "You have new email!"}]

    def test_multiple_proactive_calls(self):
        agent = make_agent()
        agent.record_proactive("First")
        agent.record_proactive("Second")
        assert len(agent._history) == 2
        assert agent._history[1]["content"] == "Second"


# ── clear_history ─────────────────────────────────────────────────────────────

class TestClearHistory:
    def test_empties_history(self):
        agent = make_agent()
        agent._history = [{"role": "user", "content": "hello"}]
        agent.clear_history()
        assert agent._history == []

    def test_idempotent_on_empty_history(self):
        agent = make_agent()
        agent.clear_history()
        assert agent._history == []


# ── conversation_for_summary ──────────────────────────────────────────────────

class TestConversationForSummary:
    def test_empty_history(self):
        agent = make_agent()
        assert agent.conversation_for_summary == []

    def test_keeps_user_and_assistant_with_content(self):
        agent = make_agent()
        agent._history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = agent.conversation_for_summary
        assert len(result) == 2

    def test_excludes_tool_messages(self):
        agent = make_agent()
        agent._history = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "abc", "content": "tool result"},
            {"role": "assistant", "content": "done"},
        ]
        result = agent.conversation_for_summary
        assert len(result) == 2
        roles = [m["role"] for m in result]
        assert "tool" not in roles

    def test_excludes_assistant_messages_with_no_content(self):
        agent = make_agent()
        agent._history = [
            {"role": "assistant", "content": None},
            {"role": "user", "content": "hello"},
        ]
        result = agent.conversation_for_summary
        assert len(result) == 1
        assert result[0]["role"] == "user"


# ── safe_process ──────────────────────────────────────────────────────────────

class TestSafeProcess:
    async def test_sends_apology_when_llm_raises(self):
        agent = make_agent()
        msg = make_message()

        with patch("assistant.core.agent.llm") as mock_llm:
            mock_llm.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("boom")
            )
            await agent.safe_process(msg)

        agent._platform.send.assert_called_once()
        sent_text = agent._platform.send.call_args[0][0]
        assert "error" in sent_text.lower()

    async def test_does_not_raise_on_exception(self):
        agent = make_agent()
        msg = make_message()

        with patch("assistant.core.agent.llm") as mock_llm:
            mock_llm.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("boom")
            )
            # Must not propagate the exception
            await agent.safe_process(msg)

    async def test_sends_response_on_success(self):
        agent = make_agent()
        msg = make_message("hello")

        response = MagicMock()
        response.choices[0].finish_reason = "stop"
        response.choices[0].message.content = "Hello back!"
        response.choices[0].message.tool_calls = None

        with patch("assistant.core.agent.llm") as mock_llm:
            mock_llm.chat.completions.create = AsyncMock(return_value=response)
            await agent.safe_process(msg)

        agent._platform.send.assert_called_once_with("Hello back!", "ch1")

    async def test_appends_user_and_assistant_to_history(self):
        agent = make_agent()
        msg = make_message("hello")

        response = MagicMock()
        response.choices[0].finish_reason = "stop"
        response.choices[0].message.content = "Hello back!"
        response.choices[0].message.tool_calls = None

        with patch("assistant.core.agent.llm") as mock_llm:
            mock_llm.chat.completions.create = AsyncMock(return_value=response)
            await agent.safe_process(msg)

        roles = [m["role"] for m in agent._history]
        assert "user" in roles
        assert "assistant" in roles

    async def test_uses_no_tool_registry_gracefully(self):
        """Agent with no registry should still complete when LLM gives a text response."""
        agent = make_agent()
        assert agent._tool_registry is None
        msg = make_message()

        response = MagicMock()
        response.choices[0].finish_reason = "stop"
        response.choices[0].message.content = "ok"
        response.choices[0].message.tool_calls = None

        with patch("assistant.core.agent.llm") as mock_llm:
            mock_llm.chat.completions.create = AsyncMock(return_value=response)
            await agent.safe_process(msg)

        agent._platform.send.assert_called_once_with("ok", "ch1")
