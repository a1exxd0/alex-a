from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from assistant.core.tool_registry import ToolRegistry, _extract_param_model


class GreetParams(BaseModel):
    name: str


class MathParams(BaseModel):
    x: int
    y: int


def make_registry(timeout: int = 60) -> ToolRegistry:
    platform = AsyncMock()
    settings = MagicMock()
    settings.confirmation_timeout_seconds = timeout
    settings.enabled_skills = []
    return ToolRegistry(platform, settings)


def make_tool_call(name: str, arguments: str, channel_id: str | None = "ch1"):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    tc._channel_id = channel_id
    return tc


# ── Registration ──────────────────────────────────────────────────────────────

class TestToolRegistration:
    def test_register_tool(self):
        reg = make_registry()

        @reg.tool("greet", "Say hello")
        async def greet(params: GreetParams) -> str:
            return f"Hello, {params.name}!"

        assert "greet" in reg._tools

    def test_duplicate_tool_name_raises(self):
        reg = make_registry()

        @reg.tool("greet", "First registration")
        async def greet_v1(params: GreetParams) -> str:
            return "hello"

        with pytest.raises(ValueError, match="Duplicate tool name"):
            @reg.tool("greet", "Second registration")
            async def greet_v2(params: GreetParams) -> str:
                return "hello"

    def test_registers_correct_name_and_description(self):
        reg = make_registry()

        @reg.tool("greet", "Say hello")
        async def greet(params: GreetParams) -> str:
            return "hello"

        entry = reg._tools["greet"]
        assert entry.name == "greet"
        assert entry.description == "Say hello"

    def test_requires_confirmation_stored(self):
        reg = make_registry()

        @reg.tool("risky", "Risky op", requires_confirmation=True)
        async def risky(params: GreetParams) -> str:
            return "done"

        assert reg._tools["risky"].requires_confirmation is True

    def test_requires_confirmation_defaults_false(self):
        reg = make_registry()

        @reg.tool("safe", "Safe op")
        async def safe(params: GreetParams) -> str:
            return "ok"

        assert reg._tools["safe"].requires_confirmation is False

    def test_multiple_tools_registered(self):
        reg = make_registry()

        @reg.tool("greet", "Greet")
        async def greet(params: GreetParams) -> str:
            return "hello"

        @reg.tool("add", "Add two numbers")
        async def add(params: MathParams) -> str:
            return str(params.x + params.y)

        assert len(reg._tools) == 2


# ── Schema generation ─────────────────────────────────────────────────────────

class TestSchemaGeneration:
    def test_empty_registry_returns_empty_list(self):
        reg = make_registry()
        assert reg.get_schemas() == []

    def test_schema_top_level_structure(self):
        reg = make_registry()

        @reg.tool("greet", "Say hello")
        async def greet(params: GreetParams) -> str:
            return "hello"

        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert "function" in schemas[0]

    def test_schema_function_fields(self):
        reg = make_registry()

        @reg.tool("greet", "Say hello")
        async def greet(params: GreetParams) -> str:
            return "hello"

        fn = reg.get_schemas()[0]["function"]
        assert fn["name"] == "greet"
        assert fn["description"] == "Say hello"
        assert "parameters" in fn

    def test_schema_strips_title_from_parameters(self):
        reg = make_registry()

        @reg.tool("greet", "Say hello")
        async def greet(params: GreetParams) -> str:
            return "hello"

        params_schema = reg.get_schemas()[0]["function"]["parameters"]
        assert "title" not in params_schema

    def test_schema_includes_model_properties(self):
        reg = make_registry()

        @reg.tool("greet", "Say hello")
        async def greet(params: GreetParams) -> str:
            return "hello"

        params_schema = reg.get_schemas()[0]["function"]["parameters"]
        assert "name" in params_schema["properties"]

    def test_schema_count_matches_tool_count(self):
        reg = make_registry()

        @reg.tool("greet", "Greet")
        async def greet(params: GreetParams) -> str:
            return "hello"

        @reg.tool("add", "Add")
        async def add(params: MathParams) -> str:
            return "0"

        assert len(reg.get_schemas()) == 2


# ── Execution ─────────────────────────────────────────────────────────────────

class TestExecution:
    async def test_execute_known_tool_returns_result(self):
        reg = make_registry()

        @reg.tool("greet", "Say hello")
        async def greet(params: GreetParams) -> str:
            return f"Hello, {params.name}!"

        tc = make_tool_call("greet", '{"name": "Alice"}')
        result = await reg.execute(tc)
        assert result == "Hello, Alice!"

    async def test_execute_unknown_tool_returns_error(self):
        reg = make_registry()
        tc = make_tool_call("nonexistent", "{}")
        result = await reg.execute(tc)
        assert "unknown tool" in result.lower()

    async def test_execute_bad_json_returns_error(self):
        reg = make_registry()

        @reg.tool("greet", "Say hello")
        async def greet(params: GreetParams) -> str:
            return "hello"

        tc = make_tool_call("greet", "not valid json{{{")
        result = await reg.execute(tc)
        assert "could not parse arguments" in result.lower()

    async def test_execute_missing_required_field_returns_error(self):
        reg = make_registry()

        @reg.tool("greet", "Say hello")
        async def greet(params: GreetParams) -> str:
            return "hello"

        tc = make_tool_call("greet", "{}")  # missing required "name"
        result = await reg.execute(tc)
        assert "could not parse arguments" in result.lower()

    async def test_execute_handler_exception_returns_error_string(self):
        reg = make_registry()

        @reg.tool("boom", "Explode")
        async def boom(params: GreetParams) -> str:
            raise ValueError("kaboom")

        tc = make_tool_call("boom", '{"name": "test"}')
        result = await reg.execute(tc)
        assert "Error running" in result
        assert "kaboom" in result

    async def test_execute_confirmation_approved_runs_handler(self):
        reg = make_registry()
        reg._platform.request_confirmation = AsyncMock(return_value=True)

        @reg.tool("risky", "Risky op", requires_confirmation=True)
        async def risky(params: GreetParams) -> str:
            return f"ran for {params.name}"

        tc = make_tool_call("risky", '{"name": "Alice"}')
        result = await reg.execute(tc)
        assert result == "ran for Alice"
        reg._platform.request_confirmation.assert_called_once()

    async def test_execute_confirmation_denied_returns_denial_message(self):
        reg = make_registry()
        reg._platform.request_confirmation = AsyncMock(return_value=False)

        @reg.tool("risky", "Risky op", requires_confirmation=True)
        async def risky(params: GreetParams) -> str:
            return "done"

        tc = make_tool_call("risky", '{"name": "test"}')
        result = await reg.execute(tc)
        assert "denied" in result.lower()
        # Handler must not have run
        assert "done" not in result

    async def test_execute_confirmation_no_channel_id_denies(self):
        reg = make_registry()

        @reg.tool("risky", "Risky op", requires_confirmation=True)
        async def risky(params: GreetParams) -> str:
            return "done"

        tc = make_tool_call("risky", '{"name": "test"}', channel_id=None)
        result = await reg.execute(tc)
        assert "no channel context" in result.lower()

    async def test_execute_no_confirmation_required_skips_prompt(self):
        reg = make_registry()

        @reg.tool("safe", "Safe op")
        async def safe(params: GreetParams) -> str:
            return "safe result"

        tc = make_tool_call("safe", '{"name": "Bob"}')
        result = await reg.execute(tc)
        assert result == "safe result"
        reg._platform.request_confirmation.assert_not_called()


# ── _extract_param_model ──────────────────────────────────────────────────────

class TestExtractParamModel:
    def test_valid_handler_returns_model_class(self):
        async def handler(params: GreetParams) -> str:
            return "ok"

        assert _extract_param_model(handler) is GreetParams

    def test_no_parameters_raises_type_error(self):
        async def handler() -> str:
            return "ok"

        with pytest.raises(TypeError, match="at least one parameter"):
            _extract_param_model(handler)

    def test_non_basemodel_annotation_raises_type_error(self):
        async def handler(params: str) -> str:
            return "ok"

        with pytest.raises(TypeError, match="pydantic BaseModel subclass"):
            _extract_param_model(handler)

    def test_missing_annotation_raises_type_error(self):
        async def handler(params) -> str:  # no type hint
            return "ok"

        with pytest.raises(TypeError):
            _extract_param_model(handler)

    def test_second_model_param_is_also_accepted(self):
        """First parameter is what matters; other params are ignored."""
        async def handler(params: MathParams, extra: str = "") -> str:
            return "ok"

        assert _extract_param_model(handler) is MathParams
