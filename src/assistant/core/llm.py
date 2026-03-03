from openai import AsyncOpenAI

from assistant.config import settings

# Chat-completions client — targets whatever provider is configured
llm = AsyncOpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)

# Embeddings client — always targets OpenAI (text-embedding-3-small)
embed_client = AsyncOpenAI(
    api_key=settings.embedding_api_key or settings.llm_api_key,
    base_url="https://api.openai.com/v1",
)
