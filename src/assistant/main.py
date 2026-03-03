import asyncio
import logging

from assistant.config import Settings
from assistant.core.agent import Agent
from assistant.core.memory import Memory
from assistant.core.tool_registry import ToolRegistry
from assistant.platforms.discord_platform import DiscordPlatform

log = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = Settings()

    platform = DiscordPlatform(settings)
    memory = Memory(settings)

    await memory.initialise()

    registry = ToolRegistry(platform, settings)
    registry.load_skills()

    agent = Agent(
        settings=settings,
        platform=platform,
        memory=memory,
        tool_registry=registry,
    )

    log.info("Starting up Discord platform...")
    await platform.start()
    
    log.info("Agent listening for messages...")
    try:
        async for message in platform.listen():
            # Process each message as a separate task to avoid blocking the listen queue
            asyncio.create_task(agent.safe_process(message))
    except asyncio.CancelledError:
        log.info("Received cancellation, shutting down...")
    finally:
        await platform.stop()
