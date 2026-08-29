from __future__ import annotations
from app.core.config import config
from typing import Any, Dict, List
import os
import time
import requests
import datetime as dt

from loguru import logger

__all__ = ["PerplexityService"]


class PerplexityService:
    def __init__(
        self, 
        api_key: str | None = None, 
        timeout: float = 60.0
    ) -> None:
        api_key = api_key or config.PPLX_API_KEY
        if not api_key:
            raise RuntimeError(
                "Perplexity API key missing - "
                "set PPLX_API_KEY environment variable."
            )
        self._key = api_key
        self._timeout = timeout

    def web_search(
        self,
        *,
        refined_query: str
    ) -> Dict[str, Any]:
        temperature: float = 0.1
        timeout: float = 100.0
        today = dt.date.today()
        search_end = today.strftime("%m/%d/%Y")
        search_start = (today - dt.timedelta(days=365)).strftime("%m/%d/%Y")

        PREFERRED_DOMAINS = [
            "pubmed.ncbi.nlm.nih.gov", 
            "jamanetwork.com", 
            "nejm.org", 
            "thelancet.com",
            "bmj.com", 
            "annals.org", 
            "acc.org", 
            "ahajournals.org",
            "escardio.org", 
            "nice.org.uk",
        ]

        pplx_sys = (
            f"You are an expert clinical information-retrieval AI. Your answer MUST be "
            f"supported by citations.\n\nQuery: '{refined_query}'.\nTime window: "
            f"{search_start}–{search_end}.\nPrioritise domains: "
            f"{', '.join(PREFERRED_DOMAINS)}.\n\n"
            "1. Summarise up-to-date general guidelines.\n"
            "2. Layer any demographic-specific considerations based on the query keywords.\n"
            "Cite rigorously."
        )
        
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system", 
                    "content": pplx_sys
                },
                {
                    "role": "user", 
                    "content": refined_query
                }
            ],
            "temperature": temperature,
            "search_after_date_filter": search_start,
            "search_before_date_filter": search_end,
            "search_domain_filter": PREFERRED_DOMAINS
        }
        
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        start = time.perf_counter()
        
        try:
            response = requests.post(
                config.PPLX_API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            pplx_json = response.json()
            delta_time = time.perf_counter() - start
            logger.debug(
                "Perplexity latency: {dt:.2f}s",
                dt=delta_time
            )

            logger.info(f"Perplexity response: {pplx_json}")

            return pplx_json
        
        except Exception as e:
            logger.error("❌ Perplexity call failed: {}", e)
            raise RuntimeError(f"Perplexity call failed: {e}")