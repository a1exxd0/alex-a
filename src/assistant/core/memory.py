import logging
import uuid
from datetime import UTC, datetime

import lancedb
from lancedb.pydantic import LanceModel, Vector

from assistant.config import Settings
from assistant.core.llm import embed_client, llm

log = logging.getLogger(__name__)

_EMBED_DIM = 1536  # text-embedding-3-small output dimension
_TABLE = "memories"

_SUMMARISE_SYSTEM = (
    "You are a memory summariser. Given a conversation, write a concise 2–3 sentence "
    "summary capturing the key topics, decisions, and any context useful in future "
    "conversations. Be factual and specific. Output only the summary, no preamble."
)


class MemoryRecord(LanceModel):
    id: str
    text: str
    created_at: datetime
    vector: Vector(_EMBED_DIM)  # type: ignore[valid-type]


class Memory:
    """Semantic memory store backed by LanceDB.

    Call ``await memory.initialise()`` once at startup before using
    ``retrieve()`` or ``store()``.
    """

    def __init__(self, settings: Settings) -> None:
        self._path = str(settings.lancedb_path)
        self._top_k = settings.memory_top_k
        self._embed_model = settings.embedding_model
        self._llm_model = settings.llm_model
        self._table: lancedb.AsyncTable | None = None  # type: ignore[type-arg]

    async def initialise(self) -> None:
        db = await lancedb.connect_async(self._path)
        response = await db.list_tables()
        if _TABLE in response.tables:
            self._table = await db.open_table(_TABLE)
        else:
            self._table = await db.create_table(_TABLE, schema=MemoryRecord)
        log.info("Memory store ready at %s", self._path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        response = await embed_client.embeddings.create(
            input=text,
            model=self._embed_model,
        )
        return response.data[0].embedding

    # ── Public API ────────────────────────────────────────────────────────────

    async def store(self, text: str) -> None:
        """Embed ``text`` and persist it as a new memory record."""
        assert self._table is not None, "Call Memory.initialise() before use"
        vector = await self._embed(text)
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            text=text,
            created_at=datetime.now(UTC),
            vector=vector,
        )
        await self._table.add([record.model_dump()])
        log.debug("Stored memory (%.60s…)", text)

    async def retrieve(self, query: str) -> list[str]:
        """Return the top-k memories most relevant to ``query``.

        Each entry is formatted as ``"[YYYY-MM-DD] summary text"`` so the
        agent can inject them directly into the LLM context.
        """
        assert self._table is not None, "Call Memory.initialise() before use"
        vector = await self._embed(query)
        results = (
            await self._table.query()
            .nearest_to(vector)
            .limit(self._top_k)
            .to_pydantic(MemoryRecord)
        )
        return [f"[{r.created_at.strftime('%Y-%m-%d')}] {r.text}" for r in results]

    async def summarise_and_store(self, messages: list[dict]) -> None:
        """Summarise a conversation and store the result as a memory.

        ``messages`` should be a list of OpenAI-format dicts
        (``{"role": "user"|"assistant", "content": "..."}``)
        representing the session to be summarised.
        """
        if not messages:
            return
        conversation = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )
        response = await llm.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": _SUMMARISE_SYSTEM},
                {"role": "user", "content": conversation},
            ],
            temperature=0.3,
        )
        summary = response.choices[0].message.content
        if summary:
            await self.store(summary.strip())
            log.info("Summarised and stored session memory")
