from __future__ import annotations

import concurrent.futures
import functools
import json
import os
import textwrap
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from loguru import logger
from openai import AzureOpenAI, OpenAIError  # type: ignore
from tqdm import tqdm

from app.core.regex_store import RegexStore
from app.core.config import config
from app.core.service_configs import RerankingConfig
from app.core.prompts import RerankingPrompts

__all__ = ["RerankerService"]

# Remove old constants - now using centralized config
# W_COS, W_LLM, W_LEX = 0.65, 0.25, 0.10
# KEEP_K = 3
# THREADS = max(4, os.cpu_count() or 8)
# MULTI_AXES = {"intent_terms", "rxnorm_terms"}
# SYS_RERANK_PROMPT = {...}
# SYS_AXIS_ORDER_PROMPT = {...}


def _lexical_overlap(a: str, b: str) -> float:
    """Calculate lexical overlap between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0 if a == b else 0.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class RerankerService:
    def __init__(
        self,
    ) -> None:
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        missing_env = [name for name, value in {
            "AZURE_OPENAI_API_KEY": self.api_key,
            "AZURE_OPENAI_ENDPOINT": self.azure_endpoint,
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": self.deployment_name,
            "AZURE_OPENAI_API_VERSION": self.api_version,
        }.items() if not value]

        if missing_env:
            raise RuntimeError(
                f"Missing Azure OpenAI environment variables: {', '.join(missing_env)}. "
                "Update your .env file accordingly."
            )

        self.client = AzureOpenAI(
            api_key=self.api_key, 
            azure_endpoint=self.azure_endpoint, 
            api_version=self.api_version
        )

        # Set deployment for LLM calls
        self._deployment = self.deployment_name
        
        # Enable LLM reranking by default (can be controlled via environment variable)
        self._use_llm = os.getenv("RERANKER_USE_LLM", "true").lower() in ("true", "1", "yes")

        logger.info(
            "Reranker will use LLM deployment='{deployment_name}'.", 
            deployment_name=self.deployment_name
        )


    def rerank(
        self,
        retrievals: Dict[str, List[Dict[str, List[Tuple[str, float]]]]],
        *,
        query: str,
    ):
        vec_hits = defaultdict(list)
        for axis, term_entries in retrievals.items():
            if not term_entries:
                continue
            for entry in term_entries:
                source_term = entry["term"]
                for match_term, cos_sim in entry["hits"]:
                    vec_hits[axis].append({
                        "src": source_term,
                        "match": match_term,
                        "dist": -float(cos_sim)
                    })

        reranked_hits = {}
        if vec_hits:
            logger.info("Starting semantic reranking for {n} axes...", n=len(vec_hits))
            max_workers = max(RerankingConfig.MAX_THREADS, os.cpu_count() or 8)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_axis = {
                    executor.submit(self._rerank_axis_hits, ax_name, hits, query): ax_name
                    for ax_name, hits in vec_hits.items()
                }
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_axis),
                    total=len(vec_hits),
                    desc="Semantic Reranking"
                ):
                    original_axis = future_to_axis[future]
                    try:
                        ax_name, best_hits = future.result()
                        if best_hits:
                            reranked_hits[ax_name] = best_hits
                    except Exception as exc:
                        logger.warning("Axis '{axis}' reranking error: {exc}", axis=original_axis, exc=exc)

        axis_order = self._determine_axis_order(reranked_hits, query)
        return reranked_hits, axis_order

    @functools.lru_cache(maxsize=4096)
    def _llm_rating(self, source: str, candidate: str, query: str) -> float:
        """Get LLM rating for term pair (0-10)."""
        if not self._use_llm:
            return 0.0
        if source.strip().lower() == candidate.strip().lower():
            return 10.0
        
        prompt = (
            f"User Query: \"{query}\"\n\n"
            f"Source Concept: \"{source}\"\n"
            f"Candidate Term: \"{candidate}\"\n\n"
            f"Rate (0-10):"
        )

        try:
            # class MockMessage:
            #     def __init__(self):
            #         if source.lower() in candidate.lower() or candidate.lower() in source.lower():
            #             self.content = "8"
            #         elif any(word in candidate.lower() for word in source.lower().split()):
            #             self.content = "6"
            #         else:
            #             self.content = "3"
            
            # class MockChoice:
            #     def __init__(self):
            #         self.message = MockMessage()
            
            # class MockResponse:
            #     def __init__(self):
            #         self.choices = [MockChoice()]
            
            # response = MockResponse()

            response = self.client.chat.completions.create(
                model=self._deployment,
                messages=[
                    RerankingPrompts.SYSTEM_RERANK, 
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=10,
                temperature=0.0,
            )

            content = response.choices[0].message.content.strip()
            m_integers = RegexStore.Utils.INTEGER.search(content)
            if m_integers:
                return float(m_integers.group(1))
            
        except OpenAIError as exc:
            logger.warning("LLM rating failed → {exc}", exc=exc)
        return 0.0

    def _rerank_axis_hits(
        self,
        axis_name: str,
        hit_rows: List[Dict],
        query: str
    ) -> Tuple[str, List[Dict]]:
        processed_hits = defaultdict(list)
        for hit_data in hit_rows:
            source, match = hit_data["src"], hit_data["match"]
            cos_sim = -hit_data["dist"]
            lex_sim = _lexical_overlap(source, match)
            llm_score = self._llm_rating(source, match, query)
            hybrid = (
                RerankingConfig.COSINE_WEIGHT * cos_sim + 
                RerankingConfig.LLM_WEIGHT * (llm_score / 10.0) + 
                RerankingConfig.LEXICAL_WEIGHT * lex_sim
            )
            hit_data.update({
                "cosine": cos_sim,
                "llm": llm_score,
                "lex": lex_sim,
                "hybrid": hybrid
            })
            processed_hits[source].append(hit_data)

        # Group hits by source term and format for response generator
        final_hits = []
        for source_term, candidates in processed_hits.items():
            candidates.sort(key=lambda x: x["hybrid"], reverse=True)
            num_to_keep = RerankingConfig.KEEP_TOP_K if axis_name in RerankingConfig.MULTI_MATCH_AXES else 1
            top_candidates = candidates[:num_to_keep]
            
            # Format as expected by response generator: each entry has "hits" key with tuples
            hits_list = [(hit["match"], hit["cosine"], hit["hybrid"]) for hit in top_candidates]
            final_hits.append({
                "term": source_term,
                "hits": hits_list
            })

        return axis_name, final_hits

    def _determine_axis_order(
        self,
        reranked_hits: Dict[str, List[Dict]],
        query: str
    ) -> List[str]:
        """Determine priority order for axes using LLM or fallback."""
        if not reranked_hits:
            logger.warning("No reranked hits - using default axis order")
            return sorted(config.AXES)

        try:
            axes_payload = {
                ax: [match_term for h in hits if h and h.get("hits") for match_term, _, _ in h["hits"]]
                for ax, hits in reranked_hits.items()
                if hits
            }

            if not axes_payload:
                logger.warning("Empty axes payload - falling back to hybrid score order")
                valid_keys = [
                    ax for ax in reranked_hits
                    if reranked_hits.get(ax) and isinstance(reranked_hits[ax], list)
                    and len(reranked_hits[ax]) > 0 and reranked_hits[ax][0].get("hits")
                ]
                return sorted(valid_keys, key=lambda ax: -reranked_hits[ax][0]["hits"][0][2])
            
            prompt = RerankingPrompts.AXIS_ORDER_SYSTEM.format(
                query=query,
                axes_json=json.dumps(
                    axes_payload, 
                    indent=2
                )
            )

            # class MockMessage:
            #     def __init__(self):
            #         self.content = json.dumps([
            #             "diagnosis_terms",
            #             "symptom_terms",
            #             "procedure_terms",
            #             "rxnorm_terms",
            #             "severity_status",
            #             "anatomy_terms",
            #             "comorbidity_terms",
            #             "temporal_context",
            #             "intent_terms",
            #             "family_history_terms",
            #             "lifestyle_terms",
            #             "allergy_terms",
            #             "age_bins",
            #             "sex_terms",
            #             "race_ethnicity",
            #             "wordlist_terms"
            #         ])
            
            # class MockChoice:
            #     def __init__(self):
            #         self.message = MockMessage()
            
            # class MockResponse:
            #     def __init__(self):
            #         self.choices = [MockChoice()]
            
            # response = MockResponse()

            response = self.client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0.0,
            )
            
            content = response.choices[0].message.content.strip()
            try:
                axis_order = json.loads(content)
                if isinstance(axis_order, list) and all(isinstance(x, str) for x in axis_order):
                    return axis_order
            except Exception as e:
                logger.warning(f"Failed to parse axis order JSON: {e}")

            valid_keys = [
                ax for ax in reranked_hits
                if reranked_hits.get(ax) and isinstance(reranked_hits[ax], list)
                and len(reranked_hits[ax]) > 0 and reranked_hits[ax][0].get("hits")
            ]
            return sorted(valid_keys, key=lambda ax: -reranked_hits[ax][0]["hits"][0][2])

        except Exception as e:
            logger.warning(f"Axis ordering failed: {e}")
            valid_keys = [
                ax for ax in reranked_hits
                if reranked_hits.get(ax) and isinstance(reranked_hits[ax], list)
                and len(reranked_hits[ax]) > 0 and reranked_hits[ax][0].get("hits")
            ]
            return sorted(valid_keys, key=lambda ax: -reranked_hits[ax][0]["hits"][0][2]) 