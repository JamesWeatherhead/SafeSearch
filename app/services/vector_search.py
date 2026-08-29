from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from typing import Any, Dict, List, Optional
import os
import textwrap
import warnings

import numpy as np
import psycopg2
from loguru import logger
from tqdm import tqdm

from app.core.regex_store import RegexStore
from app.core.config import config
from .embeddings import EmbeddingService, get_embedding_service

__all__ = ["VectorSearchService"]


class VectorSearchService:
    def __init__(
        self,
        *,
        embedder: EmbeddingService | None = None,
        pg_dsn: str | None = None,
        table_prefix: str = "",
        text_col: str = "term",
        vec_col: str = "vec",
        top_k: int = 3,
        cos_threshold: float = 0.60,
        fallback_table: str | None = None,
        fallback_axes: set[str] | None = None,
    ) -> None:
        self._embedder = embedder or get_embedding_service()
        self._dsn = pg_dsn or config.DB_DSN
        self._table_prefix = table_prefix
        self._text_col = text_col
        self._vec_col = vec_col
        self._top_k = top_k
        self._cos_threshold = cos_threshold
        self._fallback_table = fallback_table or f"{table_prefix}wordlist_terms"
        self._fallback_axes = (
            fallback_axes
            or {
                "anatomy_terms",
                "comorbidity_terms",
                "diagnosis_terms",
                "family_history_terms",
                "intent_terms",
                "procedure_terms",
                "symptom_terms",
            }
        )

        self._sql_template = textwrap.dedent(
            f"""
            SELECT {self._text_col},
                   1 - ({self._vec_col} <=> %(qvec)s::vector) AS cos_sim
            FROM   {{table}}
            ORDER  BY {self._vec_col} <=> %(qvec)s::vector
            LIMIT  %(k)s;
            """
        )

        # Track total embedding tokens used
        self._total_embedding_tokens = 0

        logger.info(
            "VectorSearchService initialized (top_k={k}, cos_threshold={t}).",
            k=top_k,
            t=cos_threshold
        )

    @property
    def total_embedding_tokens(self) -> int:
        """Get total number of embedding tokens used."""
        return self._total_embedding_tokens

    def search(self, axes: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        retrievals: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        if not axes:
            return retrievals

        with closing(psycopg2.connect(self._dsn)) as conn, conn.cursor() as cur:
            for axis, terms in tqdm(axes.items(), desc="Processing Axes"):
                if not terms:
                    continue

                full_table_name = f"{self._table_prefix}{axis}"
                logger.debug(
                    "Vector search for axis='{axis}' table='{table}'.",
                    axis=axis,
                    table=full_table_name
                )

                for term in terms:
                    if not isinstance(term, str) or not term.strip():
                        logger.warning(
                            "Skipping invalid term '{term}' in axis '{axis}'.",
                            term=term,
                            axis=axis
                        )
                        continue

                    try:
                        vec = self._embedder.embed_cached(term)
                        # Track embedding tokens
                        if hasattr(self._embedder, 'total_tokens'):
                            self._total_embedding_tokens += self._embedder.total_tokens
                    except Exception as exc:
                        logger.warning(
                            "Embedding failed for term='{term}' → {exc}",
                            term=term,
                            exc=exc
                        )
                        continue

                    try:
                        if vec is not None and vec.size > 0:
                            cur.execute(
                                self._sql_template.format(table=full_table_name),
                                {"qvec": vec.tolist(), "k": self._top_k},
                            )
                            hits: List[tuple[str, float]] = cur.fetchall()
                        else:
                            logger.warning(
                                "Skipping DB call for term '{term}' due to empty/invalid vector.",
                                term=term
                            )
                            hits = []
                    except psycopg2.Error as db_exc:
                        # Handle missing tables gracefully for PoC
                        if "does not exist" in str(db_exc) or "permission denied" in str(db_exc):
                            logger.warning(
                                "Table '{table}' missing or no permissions for axis='{axis}', term='{term}' → {exc}",
                                table=full_table_name,
                                axis=axis,
                                term=term,
                                exc=db_exc
                            )
                        else:
                            logger.error(
                                "PostgreSQL error during fetch_knn for axis='{axis}', term='{term}' → {exc}",
                                axis=axis,
                                term=term,
                                exc=db_exc
                            )
                        conn.rollback()
                        hits = []

                    # Fallback to generic wordlist_terms if similarity is poor
                    best_sim = max((hit[1] for hit in hits), default=-1.0) if hits else -1.0
                    if (
                        (not hits or best_sim < self._cos_threshold) and axis in self._fallback_axes
                    ):
                        try:
                            if vec is not None and vec.size > 0:
                                cur.execute(
                                    self._sql_template.format(table=self._fallback_table),
                                    {"qvec": vec.tolist(), "k": self._top_k},
                                )
                                fb_hits: List[tuple[str, float]] = cur.fetchall()
                                if fb_hits:
                                    logger.info(
                                        "Fallback successful for axis='{axis}', term='{term}' (best_sim={sim:.4f})",
                                        axis=axis,
                                        term=term,
                                        sim=best_sim
                                    )
                                    hits = fb_hits
                        except Exception as fb_exc:
                            if "does not exist" in str(fb_exc) or "permission denied" in str(fb_exc):
                                logger.info(
                                    "Fallback table '{table}' missing or no permissions for axis='{axis}', term='{term}' - skipping fallback",
                                    table=self._fallback_table,
                                    axis=axis,
                                    term=term
                                )
                            else:
                                logger.warning(
                                    "Fallback vector search failed axis='{axis}', term='{term}' → {exc}",
                                    axis=axis,
                                    term=term,
                                    exc=fb_exc
                                )

                    if hits:
                        for i, (match, sim) in enumerate(hits):
                            logger.debug(
                                "Hit: {match:<7s} cos={sim:6.4f}",
                                match=f"{i}: {match}",
                                sim=sim
                            )
                        retrievals[axis].append({
                            "term": term, 
                            "hits": hits
                        })
                    else:
                        logger.debug(
                            "No hits in axis '{axis}' or fallback for term '{term}'",
                            axis=axis,
                            term=term
                        )

            conn.commit()

        # Log summary
        logger.info(
            "Vector search complete. Axes with hits: {axes}",
            axes=", ".join(f"{ax}({len(lst)})" for ax, lst in retrievals.items())
        )

        return dict(retrievals)  # type: ignore[return-value] 