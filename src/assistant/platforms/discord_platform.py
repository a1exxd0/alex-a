import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import discord

from assistant.config import Settings
from assistant.platforms.base import Message, User

log = logging.getLogger(__name__)

_DISCORD_MAX_LEN = 2000


def _chunk(text: str, size: int = _DISCORD_MAX_LEN) -> list[str]:
    """Split text into chunks of at most `size` characters.

    Prefers splitting at newline boundaries to preserve formatting.
    """
    if len(text) <= size:
        return [text]
    chunks = []
    while len(text) > size:
        split_at = text.rfind("\n", 0, size)
        if split_at <= 0:
            split_at = size
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


class DiscordPlatform:
    """Discord implementation of the Platform protocol.

    Operates in DMs only. Messages from non-allowlisted users are silently
    dropped. Responses longer than 2000 characters are split into multiple
    messages at newline boundaries where possible.

    Requires the Message Content privileged intent to be enabled both here
    and in the Discord developer portal (Bot → Privileged Gateway Intents).
    """

    def __init__(self, settings: Settings) -> None:
        self._token = settings.discord_token
        self._allowed_ids = set(settings.discord_allowed_user_ids)
        self._queue: asyncio.Queue[Message] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._pending_confirmation: dict | None = None

        intents = discord.Intents.default()
        intents.message_content = True  # privileged — enable in Discord dev portal
        intents.dm_reactions = True     # needed for confirmation gate in DMs
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            log.info("Discord bot ready — logged in as %s (id=%s)", self._client.user, self._client.user.id)  # type: ignore[union-attr]

        @self._client.event
        async def on_message(msg: discord.Message) -> None:
            if msg.author.bot:
                return
            if msg.guild is not None:  # DMs only
                return
            if msg.author.id not in self._allowed_ids:
                log.warning("Ignoring message from non-allowlisted user %s", msg.author.id)
                return
            await self._queue.put(
                Message(
                    content=msg.content,
                    author_id=msg.author.id,
                    channel_id=str(msg.channel.id),
                    timestamp=datetime.now(UTC),
                )
            )

        @self._client.event
        async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
            pending = self._pending_confirmation
            if pending is None:
                return
            if (
                payload.message_id == pending["message_id"]
                and payload.user_id in self._allowed_ids
                and str(payload.emoji) in ("✅", "❌")
            ):
                pending["result"] = str(payload.emoji) == "✅"
                pending["event"].set()

    async def start(self) -> None:
        await self._client.login(self._token)
        self._task = asyncio.create_task(self._client.connect())
        await self._client.wait_until_ready()

    async def stop(self) -> None:
        await self._client.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def send(self, content: str, channel_id: str) -> None:
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        for chunk in _chunk(content):
            await channel.send(chunk)  # type: ignore[union-attr]

    async def listen(self) -> AsyncGenerator[Message, None]:
        while True:
            yield await self._queue.get()

    async def get_user(self, user_id: int) -> User:
        discord_user = await self._client.fetch_user(user_id)
        return User(id=user_id, display_name=discord_user.display_name)

    async def open_dm_channel(self, user_id: int) -> str:
        """Return the DM channel ID for *user_id*, creating it if needed."""
        discord_user = await self._client.fetch_user(user_id)
        dm = await discord_user.create_dm()
        return str(dm.id)

    async def request_confirmation(self, prompt: str, channel_id: str, timeout: int) -> bool:
        """Send *prompt*, add ✅/❌ reactions, and wait for one back.

        Returns ``True`` if the user reacts with ✅ within *timeout* seconds,
        ``False`` on ❌ or timeout.
        """
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))

        msg = await channel.send(prompt)  # type: ignore[union-attr]
        await msg.add_reaction("✅")
        await msg.add_reaction("❌")

        event = asyncio.Event()
        self._pending_confirmation = {
            "message_id": msg.id,
            "event": event,
            "result": False,
        }

        try:
            await asyncio.wait_for(event.wait(), timeout=float(timeout))
            approved = self._pending_confirmation["result"]
        except asyncio.TimeoutError:
            log.info("Confirmation timed out after %ds — treating as denied", timeout)
            approved = False
        finally:
            self._pending_confirmation = None

        return approved
