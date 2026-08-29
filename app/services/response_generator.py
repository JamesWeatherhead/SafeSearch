from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json
import os
import textwrap
import time

from loguru import logger
from openai import RateLimitError, APIConnectionError, APITimeoutError, AzureOpenAI
from app.core.config import config

__all__ = ["ResponseGeneratorService"]

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a HIPAA-compliant clinical search assistant. "
        "Use the provided medical concepts (grouped by axis) to craft a concise, "
        "evidence-based answer for the user's query. List key citations inline "
        "as [1], [2] etc. Use markdown."
    ),
}

MAX_RETRIES = 3
MAX_TOKENS = 1200
TEMPERATURE = 0.15

class ResponseGeneratorService:
    """LLM wrapper for final answer generation."""

    def __init__(self) -> None:
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self._total_tokens = 0

        missing_env = [name for name, value in {
            "AZURE_OPENAI_API_KEY": self.api_key,
            "AZURE_OPENAI_ENDPOINT": self.azure_endpoint,
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": self.deployment_name,
            "AZURE_OPENAI_API_VERSION": self.api_version,
        }.items() if not value]

        if missing_env:
            raise RuntimeError(
                f"Missing .env variables for ResponseGenerator: {', '.join(missing_env)}. "
                "Update .env file accordingly."
            )
        
        self.client = AzureOpenAI(
            api_key=self.api_key, 
            azure_endpoint=self.azure_endpoint, 
            api_version=self.api_version
        )

        if not self.client:
            raise RuntimeError("⛔ Failed to initialize ResponseGenerator client.")


    @property
    def total_tokens(self) -> int:
        """Return total tokens used for response generation."""
        return self._total_tokens

    def _format_citations(self, citations: List[Dict]) -> str:
        """Format citations into a markdown reference list."""
        if not citations:
            return ""
        
        formatted = []
        for i, cite in enumerate(citations, 1):
            if isinstance(cite, dict):
                title = cite.get('title', 'No title').strip()
                url = cite.get('url') or cite.get('uri', 'No URL')
                formatted.append(f"[{i}] {title}\n    {url}")
            else:
                formatted.append(f"[{i}] {str(cite)}")
        return "\n".join(formatted)

    def _validate_response(self, content: str) -> bool:
        # TODO: get a more professional solution if step is necessary, else remove
        if not content or len(content.strip()) < 10:
            logger.warning("Response content too short or empty")
            return False

        if not any(md in content for md in ["#", "##", "*", "**", "["]):
            logger.warning("Response lacks basic markdown formatting")
            return False
        
        return True

    def generate(
        self, 
        *, 
        query: str, 
        reranked_hits: Dict[str, List[Dict]],
        citations: Optional[List[Dict]] = None
    ) -> str:
        """Return markdown answer for `query`."""
        axis_context = defaultdict(list)
        for axis, term_entries in reranked_hits.items():
            for entry in term_entries:
                for match_term, _cos, _hyb in entry["hits"]:
                    axis_context[axis].append(match_term)

        context_json = json.dumps(axis_context, ensure_ascii=False, indent=2)
        
        citations_str = ""
        if citations:
            citations_str = "\n\nProvided References:\n" + self._format_citations(citations)
        
        user_prompt = textwrap.dedent(
            f"""
            USER QUERY:
            {query}

            MATCHED MEDICAL CONCEPTS BY AXIS (JSON):
            {context_json}{citations_str}

            Please answer the query using these concepts as relevant. Cite each
            referenced concept once. Return markdown only.
            """
        ).strip()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Generating response (attempt {attempt}/{MAX_RETRIES})")

                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        SYSTEM_PROMPT, 
                        {
                            "role": "user", 
                            "content": user_prompt
                        }
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                
                content = response.choices[0].message.content.strip()
                
                if response.usage:
                    self._total_tokens += response.usage.total_tokens
                    logger.debug(
                        "Token usage - prompt: {prompt}, completion: {completion}, total: {total}",
                        prompt=response.usage.prompt_tokens,
                        completion=response.usage.completion_tokens,
                        total=response.usage.total_tokens
                    )
                
                if self._validate_response(content):
                    logger.info("Successfully generated response")
                    return content
                else:
                    logger.warning(f"Response validation failed on attempt {attempt}")
                    if attempt == MAX_RETRIES:
                        return "Failed to generate a valid response after multiple attempts."
                    
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                logger.warning(f"OpenAI API error on attempt {attempt}: {type(e).__name__} - {e}")
                if attempt == MAX_RETRIES:
                    return f"Failed to generate response due to API error: {type(e).__name__}"
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt}: {type(e).__name__} - {e}")
                if attempt == MAX_RETRIES:
                    return f"Failed to generate response due to unexpected error: {type(e).__name__}"
                time.sleep(2 ** attempt)
        
        return "Failed to generate a response after all retries." 