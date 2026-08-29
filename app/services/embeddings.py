from __future__ import annotations

from functools import lru_cache
from typing import Sequence
import os

import numpy as np
from loguru import logger
from openai import AzureOpenAI

from app.core.config import config

__all__ = ["EmbeddingService", "get_embedding_service"]


class EmbeddingService:
    def __init__(self) -> None:
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        missing_env = [name for name, value in {
            "AZURE_OPENAI_API_KEY": self.api_key,
            "AZURE_OPENAI_ENDPOINT": self.endpoint,
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": self.deployment,
            "AZURE_OPENAI_API_VERSION": self.api_version,
        }.items() if not value]

        if missing_env:
            raise RuntimeError(
                f"Missing Azure OpenAI environment variables: {', '.join(missing_env)}. "
                "Update your .env file accordingly."
            )

        self._total_tokens = 0

        try:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
            )

        except Exception as e:
            raise RuntimeError(
                "Unable to initialize Azure OpenAI client; "
                "check credentials and network connectivity."
            ) from e

        logger.info(
            f"EmbeddingService initialized with endpoint='{self.endpoint}', "
            f"deployment='{self.deployment}'"
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by all embedding calls made through this service."""
        return self._total_tokens

    @total_tokens.setter
    def total_tokens(self, value: int) -> None:
        """Setter needed to support in-place updates such as ``self.total_tokens += n``."""
        self._total_tokens = value

    def embed(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            raise ValueError("`text` must be a non-empty string.")

        logger.debug(f"Requesting embedding (len={len(text)}).")
        
        try:
            response = self.client.embeddings.create(
                model=self.deployment, 
                input=[text]
            )

            if response.usage:
                self.total_tokens += response.usage.total_tokens
                logger.debug(f"Embedding tokens used: {response.usage.total_tokens}")
            
            vector: Sequence[float] = response.data[0].embedding
            if not vector or len(vector) == 0:
                raise RuntimeError("Received empty embedding vector from API")
                
            arr: np.ndarray = np.asarray(vector, dtype=np.float32)
            if arr.size == 0:
                raise RuntimeError("Created empty numpy array from embedding vector")
                
            return arr
            
        except Exception as e:
            logger.error(
                f"Error during embedding API call for text '{text[:50]}...': "
                f"{type(e).__name__} - {str(e)}"
            )
            raise RuntimeError(f"Embedding API call failed: {str(e)}") from e

    @lru_cache(maxsize=4096)
    def embed_cached(self, text: str) -> np.ndarray:
        return self.embed(text)


_global_embedding_service: EmbeddingService | None = None

def get_embedding_service() -> EmbeddingService:
    global _global_embedding_service
    if _global_embedding_service is None:
        _global_embedding_service = EmbeddingService()
    return _global_embedding_service 