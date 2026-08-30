from __future__ import annotations

import textwrap
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from loguru import logger

from app.core.config import config

from .vector_search import VectorSearchService
from .reranking import RerankerService
from .response_generator import ResponseGeneratorService
from .phi_checker import PHICheckerService
from .perplexity import PerplexityService
from .axis_extraction import AxisExtractionService
from .query_builder import QueryBuilderService

from dotenv import load_dotenv
load_dotenv()

__all__ = ["PipelineService", "PipelineResult"]

class PipelineResult(Dict[str, Any]):
    pass

class PipelineService:
    def __init__(self) -> None:
        self.axis_extractor = AxisExtractionService()
        self.vector_search = VectorSearchService()
        self.rerank = RerankerService()
        self.query_builder = QueryBuilderService()
        self.perplexity = PerplexityService()
        self.response_generator = ResponseGeneratorService()
        self.phi_checker = PHICheckerService()

    def run(self, user_query: str) -> PipelineResult:
        axes = self.axis_extractor.extract_axes(user_query)

        retrievals = self.vector_search.search(axes)

        # Use the rerank service properly - it returns (reranked_hits, axis_order)
        reranked_hits, axis_order = self.rerank.rerank(retrievals, query=user_query)

        # Build PHI-safe search query from sanitised axis matches
        phi_safe_query = None
        try:
            if reranked_hits and axis_order:
                phi_safe_query = self.query_builder.build_phi_safe_query(
                    reranked_hits=reranked_hits,
                    axis_order=axis_order
                )
                logger.info(
                    "PHI-safe query generated: '{query}'",
                    query=phi_safe_query
                )
            else:
                logger.warning("No reranked hits or axis order available for PHI-safe query building")
        except Exception as e:
            logger.error(f"PHI-safe query building failed: {e}")

        if not phi_safe_query:
            raise RuntimeError("PHI-safe query unavailable; external search aborted.")

        search_query = phi_safe_query
        logger.info("Using PHI-safe query for Perplexity search")

        evidence = self.perplexity.web_search(refined_query=search_query)

        response = self.response_generator.generate(
            query=user_query,
            reranked_hits=reranked_hits,
            citations=evidence.get("citations", [])
        )

        wrapped_response = textwrap.fill(
            response, 
            width=80, 
            initial_indent="  ", 
            subsequent_indent="  "
        )
        logger.info(f"Response:\n{wrapped_response}")

        result: PipelineResult = {
            "query": user_query,
            "axes": axes,
            "retrievals": retrievals,
            "reranked_hits": reranked_hits,
            "axis_order": axis_order,
            "evidence": evidence,
            "phi_safe_query": phi_safe_query,
            "response": response,
            "phi_flags": {
                "query": self.phi_checker.contains_phi(user_query),
                "answer": self.phi_checker.contains_phi(response),
                "search_query": self.phi_checker.contains_phi(search_query) if phi_safe_query else None
            }
        }

        return result 