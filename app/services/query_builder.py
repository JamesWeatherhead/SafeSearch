from __future__ import annotations

import json
import time
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional

from loguru import logger
from openai import AzureOpenAI, RateLimitError, APIConnectionError, APITimeoutError

from app.core.config import config
from app.core.prompts import QueryBuilderPrompts

__all__ = ["QueryBuilderService"]


class QueryBuilderService:
    """Service for building PHI-safe search queries from sanitised axis matches."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        deployment_name: str | None = None,
        api_version: str = "2023-12-01-preview",
    ) -> None:
        api_key = api_key or config.AZURE_OPENAI_API_KEY
        azure_endpoint = azure_endpoint or config.AZURE_OPENAI_ENDPOINT
        deployment_name = deployment_name or config.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME

        if not api_key or not azure_endpoint or not deployment_name:
            raise RuntimeError(
                "Missing Azure OpenAI environment variables for QueryBuilder. "
                "Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME in your .env file."
            )

        self._deployment = deployment_name
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        self._total_tokens = 0

        logger.info(
            "QueryBuilderService initialized with deployment='{deployment}'.",
            deployment=deployment_name
        )

    def build_phi_safe_query(
        self,
        *,
        reranked_hits: Dict[str, List[Dict]],
        axis_order: List[str],
        max_retries: int = 3
    ) -> str:
        
        logger.info("--------------------------------------------------------")
        logger.info("Building PHI-safe query from sanitised axis matches")

        sanitised_matches = OrderedDict()
        for axis in axis_order:
            hits = reranked_hits.get(axis, [])
            if hits and hits[0].get("hits"):
                # Extract top match from the new structure: hits[0]["hits"][0][0]
                top_match = hits[0]["hits"][0][0]  # First tuple's first element (match_term)
                sanitised_matches[axis] = top_match

        if not sanitised_matches:
            raise RuntimeError("No sanitised axis matches available for query construction.")

        logger.debug(
            "Extracted sanitised matches: {matches}",
            matches=dict(sanitised_matches)
        )

        keywords_in_order = list(sanitised_matches.values())
        fallback_query = " ".join(keywords_in_order)
        
        logger.debug(
            "Initial keyword string: {keywords}",
            keywords=fallback_query
        )

        refined_query = self._refine_query_with_llm(
            prompt=QueryBuilderPrompts.REFINEMENT_PROMPT.format(
                keywords_in_order=keywords_in_order
            ),
            fallback=fallback_query,
            max_retries=max_retries
        )

        logger.info(
            "PHI-safe query built successfully: '{query}'",
            query=refined_query
        )

        logger.info("--------------------------------------------------------")

        return refined_query

    def _refine_query_with_llm(
        self,
        *,
        prompt: str,
        fallback: str,
        max_retries: int
    ) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    f"Attempting query refinement "
                    f"(attempt {attempt}/{max_retries})"
                )

                response = self._client.chat.completions.create(
                    model=self._deployment,
                    messages=[
                        {
                            "role": "system",
                            "content": QueryBuilderPrompts.SYSTEM_REFINER
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=80,
                    temperature=0.05
                )

                refined_query = response.choices[0].message.content.strip()

                if refined_query:
                    logger.debug(
                        f"Query refinement successful, "
                        f"query generated: {refined_query}"
                    )
                    return refined_query
                else:
                    logger.warning(
                        f"LLM returned empty response on attempt "
                        f"{attempt}/{max_retries}"
                    )
            except Exception as e:
                logger.error(
                    f"LLM query refinement failed on attempt "
                    f"{attempt}: {type(e).__name__}: {e}"
                )

        logger.warning(
            f"LLM query refinement failed after "
            f"{max_retries} attempts, using fallback query"
        )   
        return fallback 