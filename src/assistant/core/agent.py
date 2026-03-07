import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from assistant.config import Settings
from assistant.core.llm import llm
from assistant.core.memory import Memory
from assistant.platforms.base import Message, Platform, User

if TYPE_CHECKING:
    from assistant.core.tool_registry import ToolRegistry

log = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 10


class Agent:
    """Core message-processing loop.

    Typical call in the main run loop::

        async for message in platform.listen():
            await agent.safe_process(message)   # catches + notifies on error
            # or:
            await agent.process(message)        # raises on error
    """

    def __init__(
        self,
        settings: Settings,
        platform: Platform,
        memory: Memory,
        tool_registry: "ToolRegistry | None" = None,
    ) -> None:
        self._platform = platform
        self._memory = memory
        self._tool_registry = tool_registry
        self._llm_model = settings.llm_model
        self._token_limit = settings.context_token_limit
        self._system_prompt_template = settings.system_prompt_path.read_text(encoding="utf-8")
        self._timezone = settings.timezone

        self._history: list[dict[str, Any]] = []
        self._user_cache: dict[int, User] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    async def process(self, message: Message) -> None:
        """Process one inbound message through the full agentic loop.

        Raises any exception encountered so the caller can decide how to
        handle it (log only, send a notification, restart, etc.).
        """
        user = await self._get_user(message.author_id)
        memories = await self._memory.retrieve(message.content)
        system = self._build_system_prompt(user.display_name, memories)

        self._history.append({"role": "user", "content": message.content})
        self._trim_history()

        context = [{"role": "system", "content": system}, *self._history]
        tools = self._tool_registry.get_schemas() if self._tool_registry else []

        for iteration in range(_MAX_TOOL_ITERATIONS):
            response = await llm.chat.completions.create(
                model=self._llm_model,
                messages=context,
                tools=tools if tools else None,  # type: ignore[arg-type]
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                # Append the assistant's tool-call turn to context
                context.append({
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ],
                })

                # Execute each tool and append its result
                for tool_call in choice.message.tool_calls:
                    tool_call._channel_id = message.channel_id  # type: ignore[attr-defined]
                    result = await self._execute_tool(tool_call)
                    context.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                log.debug("Tool iteration %d/%d", iteration + 1, _MAX_TOOL_ITERATIONS)

            else:
                # Final content response
                content = choice.message.content or ""
                self._history.append({"role": "assistant", "content": content})
                self._trim_history()
                await self._platform.send(content, message.channel_id)
                return

        # Fell through the iteration cap
        log.error("Hit tool iteration limit (%d) — aborting", _MAX_TOOL_ITERATIONS)
        await self._platform.send(
            "I got stuck in a tool loop and had to stop. Please try again.",
            message.channel_id,
        )

    async def safe_process(self, message: Message) -> None:
        """Like ``process()`` but catches all exceptions.

        On failure: logs the full traceback and sends a short apology to the
        user.  Use this in the production run loop; use ``process()`` directly
        when you want failures to surface immediately (e.g. during debugging).
        """
        try:
            await self.process(message)
        except Exception:
            log.exception("Unhandled error while processing message from %s", message.author_id)
            try:
                await self._platform.send(
                    "Sorry, I ran into an error and couldn't complete your request.",
                    message.channel_id,
                )
            except Exception:
                log.exception("Failed to send error notification to user")

    @property
    def conversation_for_summary(self) -> list[dict[str, Any]]:
        """History filtered to plain user/assistant text turns.

        Strips tool-call and tool-result messages so the summariser receives
        a clean conversation transcript.
        """
        return [
            m for m in self._history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

    def record_proactive(self, content: str) -> None:
        """Append a proactively sent message to the conversation history.

        Call this after sending an unsolicited notification (e.g. an email
        summary) so the agent can refer back to it in subsequent turns.
        """
        self._history.append({"role": "assistant", "content": content})
        self._trim_history()

    def clear_history(self) -> None:
        """Discard the in-memory conversation buffer."""
        self._history.clear()
        log.debug("Conversation history cleared")

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _get_user(self, user_id: int) -> User:
        if user_id not in self._user_cache:
            self._user_cache[user_id] = await self._platform.get_user(user_id)
        return self._user_cache[user_id]

    def _build_system_prompt(self, display_name: str, memories: list[str]) -> str:
        memory_block = (
            "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories)
            if memories
            else ""
        )
        return self._system_prompt_template.format(
            user=display_name,
            date=datetime.now(UTC).strftime("%A, %-d %B %Y"),
            timezone=self._timezone,
            memories=memory_block,
        ).strip()

    async def _execute_tool(self, tool_call: Any) -> str:
        """Execute a tool call via the registry.

        Stubbed in step 4 — the ToolRegistry (step 5) fills this in fully.
        The confirmation gate is also applied inside the registry.
        """
        if self._tool_registry is None:
            log.error("Tool call received but no registry is configured")
            return "Error: tool registry not configured."
        return await self._tool_registry.execute(tool_call)

    def _trim_history(self) -> None:
        """Drop the oldest messages until the buffer is within the token limit."""
        while self._estimate_tokens() > self._token_limit and len(self._history) > 2:
            removed = self._history.pop(0)
            log.debug("Trimmed oldest message from history (%d chars)", len(removed.get("content") or ""))

    def _estimate_tokens(self) -> int:
        """Rough token estimate: ~4 characters per token."""
        return sum(len(m.get("content") or "") for m in self._history) // 4
