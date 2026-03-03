"""Tool registry — central catalogue that skills register handlers into.

Usage inside a skill module::

    from pydantic import BaseModel
    from assistant.core.tool_registry import ToolRegistry

    class SearchParams(BaseModel):
        query: str

    def register(registry: ToolRegistry) -> None:
        @registry.tool("web_search", "Search the web for information")
        async def web_search(params: SearchParams) -> str:
            ...
            return "results"

The registry is created at startup and passed to the Agent.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, get_type_hints

from pydantic import BaseModel

from assistant.config import Settings
from assistant.platforms.base import Platform

log = logging.getLogger(__name__)

# Type alias for an async handler: takes a BaseModel subclass, returns a string.
ToolHandler = Callable[..., Coroutine[Any, Any, str]]


@dataclass(slots=True)
class _ToolEntry:
    """Internal record for a registered tool."""

    name: str
    description: str
    handler: ToolHandler
    param_model: type[BaseModel]
    requires_confirmation: bool = False


class ToolRegistry:
    """Collects tool handlers from skills and exposes them to the agent loop.

    Parameters
    ----------
    platform:
        Used by the confirmation gate to prompt the user.
    settings:
        Provides ``confirmation_timeout_seconds`` and ``enabled_skills``.
    """

    def __init__(self, platform: Platform, settings: Settings) -> None:
        self._platform = platform
        self._settings = settings
        self._tools: dict[str, _ToolEntry] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def tool(
        self,
        name: str,
        description: str,
        *,
        requires_confirmation: bool = False,
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Decorator that registers an async handler as a tool.

        The handler's **first parameter** must be typed as a
        ``pydantic.BaseModel`` subclass.  The registry inspects that
        annotation to build the OpenAI function-calling JSON schema
        automatically.

        Example::

            @registry.tool("greet", "Say hello", requires_confirmation=False)
            async def greet(params: GreetParams) -> str:
                return f"Hello, {params.name}!"
        """

        def decorator(fn: ToolHandler) -> ToolHandler:
            if name in self._tools:
                raise ValueError(f"Duplicate tool name: {name!r}")

            param_model = _extract_param_model(fn)
            self._tools[name] = _ToolEntry(
                name=name,
                description=description,
                handler=fn,
                param_model=param_model,
                requires_confirmation=requires_confirmation,
            )
            log.info("Registered tool %r (confirmation=%s)", name, requires_confirmation)
            return fn

        return decorator

    # ── Schema generation ─────────────────────────────────────────────────────

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool definitions for all registered tools."""
        schemas: list[dict[str, Any]] = []
        for entry in self._tools.values():
            json_schema = entry.param_model.model_json_schema()
            # Strip the top-level 'title' that Pydantic adds — the function
            # name already serves that purpose.
            json_schema.pop("title", None)
            schemas.append({
                "type": "function",
                "function": {
                    "name": entry.name,
                    "description": entry.description,
                    "parameters": json_schema,
                },
            })
        return schemas

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute(self, tool_call: Any) -> str:
        """Deserialise arguments, run the handler, return its result string.

        If the tool has ``requires_confirmation=True``, the user is prompted
        via the platform before execution proceeds.

        Exceptions raised by the handler are caught, logged, and returned as
        an error string so the LLM can inform the user without crashing the
        agent loop.
        """
        fn_name: str = tool_call.function.name
        raw_args: str = tool_call.function.arguments

        entry = self._tools.get(fn_name)
        if entry is None:
            log.error("Unknown tool %r requested by LLM", fn_name)
            return f"Error: unknown tool '{fn_name}'."

        # Deserialise into the Pydantic model
        try:
            params = entry.param_model.model_validate_json(raw_args)
        except Exception:
            log.exception("Failed to parse arguments for tool %r", fn_name)
            return f"Error: could not parse arguments for '{fn_name}'."

        # Confirmation gate
        if entry.requires_confirmation:
            prompt = (
                f"**Tool: {entry.name}**\n"
                f"{entry.description}\n\n"
                f"```json\n{json.dumps(params.model_dump(), indent=2, default=str)}\n```\n\n"
                "React ✅ to approve or ❌ to deny."
            )
            # We need a channel_id — the tool_call object doesn't carry one,
            # so we expect it to be set externally before execute() is called.
            channel_id = getattr(tool_call, "_channel_id", None)
            if channel_id is None:
                log.warning(
                    "Confirmation required for %r but no channel_id available — denying",
                    fn_name,
                )
                return f"Error: confirmation required for '{fn_name}' but no channel context available."

            approved = await self._platform.request_confirmation(
                prompt,
                channel_id,
                self._settings.confirmation_timeout_seconds,
            )
            if not approved:
                log.info("User denied confirmation for tool %r", fn_name)
                return f"The user denied permission to run '{fn_name}'."

        # Execute
        try:
            result = await entry.handler(params)
        except Exception:
            tb = traceback.format_exc()
            log.exception("Tool %r raised an exception", fn_name)
            return f"Error running '{fn_name}': {tb.splitlines()[-1]}"

        return result

    # ── Skill loading ─────────────────────────────────────────────────────────

    def load_skills(self, enabled: list[str] | None = None) -> None:
        """Import each enabled skill module and call its ``register()`` hook.

        Skill modules live under ``assistant.skills.<name>`` and must expose a
        ``register(registry: ToolRegistry) -> None`` function.
        """
        names = enabled if enabled is not None else self._settings.enabled_skills
        for skill_name in names:
            fq = f"assistant.skills.{skill_name}"
            try:
                mod = importlib.import_module(fq)
            except ImportError:
                log.exception("Could not import skill module %r", fq)
                continue

            register_fn = getattr(mod, "register", None)
            if register_fn is None:
                log.error("Skill module %r has no register() function — skipping", fq)
                continue

            try:
                register_fn(self)
                log.info("Loaded skill %r (%d tools total)", skill_name, len(self._tools))
            except Exception:
                log.exception("register() failed for skill %r", fq)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_param_model(fn: ToolHandler) -> type[BaseModel]:
    """Inspect a handler's first parameter annotation and return the model.

    Raises ``TypeError`` if the annotation is missing or not a BaseModel
    subclass.
    """
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if not params:
        raise TypeError(
            f"Tool handler {fn.__qualname__!r} must accept at least one "
            "parameter typed as a pydantic BaseModel subclass."
        )

    first = params[0]
    model = hints.get(first.name)

    if model is None or not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError(
            f"First parameter of {fn.__qualname__!r} must be annotated with a "
            f"pydantic BaseModel subclass, got {model!r}."
        )

    return model
