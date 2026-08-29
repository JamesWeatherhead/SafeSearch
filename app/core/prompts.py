import textwrap
from typing import Dict, Any
from dataclasses import dataclass
import json

__all__ = [
    "RerankingPrompts",
    "AxisExtractionPrompts", 
    "ResponseGeneratorPrompts",
    "QueryBuilderPrompts",
]


@dataclass(frozen=True)
class RerankingPrompts:
    """Prompts for the RerankerService."""
    
    SYSTEM_RERANK = {
        "role": "system",
        "content": (
            "Rate (0-10) how well the candidate term is a clinical synonym, subtype, "
            "or contextually correct replacement for the source concept, considering "
            "the full user query. Return ONLY the integer rating."
        ),
    }
    
    AXIS_ORDER_SYSTEM = textwrap.dedent("""
        You are a clinical information-retrieval strategist.
        USER QUERY (do NOT reveal downstream): ```{query}```
        SANITISED AXIS CANDIDATES (matched vocabulary terms for clinical axes based on the user query):
        {axes_json}
        Task: Determine importance order for these axes for a search query or summary.
        Return ONLY a JSON array of axis names in order of importance.
    """).strip()


@dataclass(frozen=True)
class AxisExtractionPrompts:
    """Prompts for the AxisExtractionService."""
    
    SYSTEM_EXTRACTION = {
        "role": "system",
        "content": textwrap.dedent(
            """
            Extract exactly these axes and return ONLY valid JSON:
            age_bins, anatomy_terms, comorbidity_terms, diagnosis_terms, family_history_terms,
            intent_terms, procedure_terms, rxnorm_terms, sex_terms, symptom_terms,
            temporal_context, severity_status, race_ethnicity, lifestyle_terms,
            allergy_terms, wordlist_terms
            • Use only strings literally present in the user's query *except* for intent_terms, age_bins and trivial sex detection.
            • For intent_terms, infer broader categories (e.g., "diagnosis", "treatment", "guideline", "prognosis", etc.) based on the query's meaning.
            • For age_bins, infer from phrases like "X-year-old" or dates like "born <Month> <Day>, <Year>" and map to predefined bins.
            • If umbrella phrases like "anticoagulation therapy" appear and are relevant to an intent, keep them in intent_terms. For rxnorm_terms, try to expand such umbrella medical concepts with specific exemplar drug names if appropriate (e.g., warfarin for anticoagulation therapy).
            • Return **pure JSON only** – no markdown, no commentary. The value for each axis key should be a list of strings. If no terms are found for an axis, the value should be the empty list.
            """
        ),
    }


@dataclass(frozen=True)
class ResponseGeneratorPrompts:
    """Prompts for the ResponseGeneratorService."""
    
    SYSTEM_RESPONSE = {
        "role": "system",
        "content": (
            "You are a HIPAA-compliant clinical search assistant. "
            "Use the provided medical concepts (grouped by axis) to craft a concise, "
            "evidence-based answer for the user's query. List key citations inline "
            "as [1], [2] etc. Use markdown."
        ),
    }


@dataclass(frozen=True)
class QueryBuilderPrompts:
    """Prompts for the QueryBuilderService."""
    
    SYSTEM_REFINER = "You are an expert clinical query refiner."

    REFINEMENT_PROMPT = textwrap.dedent("""
        You refine terse clinical keyword lists into a single, natural-language
        literature-search query.
        KEYWORDS IN PRIORITY ORDER: {keywords_in_order}
        RETURN ONLY THE REFINED QUERY.
    """).strip()
