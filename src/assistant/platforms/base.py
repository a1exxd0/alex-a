from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import AsyncGenerator, Protocol, runtime_checkable


@dataclass
class Message:
    content: str
    author_id: int
    channel_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class User:
    id: int
    display_name: str


@runtime_checkable
class Platform(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, content: str, channel_id: str) -> None: ...
    async def listen(self) -> AsyncGenerator[Message, None]: ...
    async def get_user(self, user_id: int) -> User: ...
    async def request_confirmation(self, prompt: str, channel_id: str, timeout: int) -> bool: ...
