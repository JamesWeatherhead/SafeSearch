from __future__ import annotations

from typing import Dict, List, Optional
import json
import os
import textwrap
import datetime as _dt
import warnings
from pprint import pp

from loguru import logger
from openai import AzureOpenAI
from app.core.regex_store import RegexStore
from app.core.config import config
from app.core.data_mappings import AxisExtractionMappings

__all__ = ["AxisExtractionService"]

_mappings = AxisExtractionMappings()

SYSTEM_PROMPT : Dict[str, str] = {
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

STARTING_PRIORITY = 1


def _age_bin_from_number(age: int) -> str:
    AGE_THRESHOLDS = [
        (1, "neonate"),
        (2, "infant"),
        (4, "toddler"),
        (10, "earlychildhood"),
        (13, "child"),
        (18, "adolescent"),
        (26, "youngadult"),
        (45, "adult"),
        (65, "middleaged"),
        (80, "elderly"),
    ]

    for upper_bound, label in AGE_THRESHOLDS:
        if age < upper_bound:
            return label
    return "lateelderly"


def _token_contains(term: str, norm_query: str) -> bool:
    """Check if a normalized term is present in a normalized query."""
    cleaned_term = RegexStore.Utils.NON_ALPHA_NUM.sub(" ", term.lower())
    cleaned_term = " ".join(cleaned_term.split())
    return bool(cleaned_term) and cleaned_term in norm_query


class AxisExtractionService:
    """Service for extracting clinical categories (axes) from user queries."""

    def __init__(self) -> None:
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
        
        self._total_tokens: int = 0

        logger.info(
            "QueryProcessor initialised with deployment='{deployment}'.", 
            deployment=self.deployment_name
        )

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens used for axis extraction."""
        return self._total_tokens

    def _expand_rxnorm_umbrellas(self, axes: Dict[str, List[str]]) -> None:
        """Expand umbrella medical terms into specific drug examples."""
        expanded_terms = []
        for term in axes.get("rxnorm_terms", []):
            if term.lower() in _mappings.RxnormUmbrellas:
                expanded_terms.extend(_mappings.RxnormUmbrellas[term.lower()])
        if expanded_terms:
            axes["rxnorm_terms"].extend(expanded_terms)

    def _remove_substrings_and_duplicates(self, axes: Dict[str, List[str]]) -> None:
        """Remove substring terms and duplicates while preserving order."""
        for axis in axes:
            if not axes[axis]:
                continue

            # Sort by length (descending) to process longer terms first
            sorted_terms = sorted(list(set(axes[axis])), key=len, reverse=True)
            unique_terms = []
            seen_lower = set()

            for term in sorted_terms:
                term_lower = term.lower()
                # Skip if this term is a substring of an existing term
                if any(term_lower != existing and term_lower in existing 
                      for existing in seen_lower):
                    continue
                if term_lower not in seen_lower:
                    unique_terms.append(term)
                    seen_lower.add(term_lower)

            axes[axis] = sorted(unique_terms)

    def _infer_age_from_date(self, query: str) -> Optional[int]:
        """Try to infer age from a date of birth in the query."""
        m_dob = RegexStore.Queries.DOB.search(query)
        if not m_dob:
            return None

        dob_str = m_dob.group(1).replace(",", "")
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                dob = _dt.datetime.strptime(dob_str, fmt).date()
                age = (_dt.date.today() - dob).days // 365
                return age
            except ValueError:
                continue
        return None

    def extract_axes(self, query: str) -> Dict[str, List[str]]:
        try:
            logger.debug(
                f"Calling Azure LLM for axis extraction (len={len(query)})."
            )

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    SYSTEM_PROMPT, {
                        "role": "user", 
                        "content": query
                    }
                ],
                max_tokens=600,
                temperature=0.0,
                response_format={
                    "type": "json_object"
                },
            )
            
            content: json = response.choices[0].message.content
            axes: Dict[str, List[str]] = json.loads(content)
                
        except Exception as e:
            logger.warning(
                f"Axis extraction via LLM failed -> "
                "{e}. Falling back to heuristic.",
            )

            axes = {axis: [] for axis in config.AXES}

            axes["wordlist_terms"] = [query]

        q_lower = query.lower()
        norm_q = " ".join(
            RegexStore.Utils.NON_ALPHA_NUM.sub(" ", q_lower).split()
        )

        for axis in config.AXES:
            values = axes.get(axis, [])
            if isinstance(values, str):
                axes[axis] = [values]
            elif not isinstance(values, list):
                axes[axis] = []
            else:
                axes[axis] = [
                    value 
                    for value in values 
                    if isinstance(value, str)
                ]

        # 1. guideline regex shortcut
        # m_guideline = RegexStore.Queries.GUIDELINE.search(query)
        # if m_guideline:
        #     dx = m_guideline.group(1).strip()
        #     if dx and dx not in axes["diagnosis_terms"]:
        #         axes["diagnosis_terms"].insert(0, dx)
        #         print("DIAGNOSIS TERMS:", axes["diagnosis_terms"])
        #     if "guideline" not in axes["intent_terms"]:
        #         axes["intent_terms"].append("guideline")

        # 2. intent enrichment
        candidate_intents = {i.lower() for i in axes["intent_terms"]}
        for canonical, synonyms in _mappings.INTENT_SYNONYMS.items():
            if any(s in q_lower for s in synonyms):
                candidate_intents.add(canonical)

        if not candidate_intents:
            candidate_intents.update(_mappings.DEFAULT_INTENTS)

        axes["intent_terms"] = sorted(
            candidate_intents, 
            key=lambda x: (_mappings.intent_priority_map.get(x, 99), x)
        )

        for axis in _mappings.LITERAL_FILTER_AXES:
            axes[axis] = [
                term 
                for term in axes[axis] 
                if _token_contains(term, norm_q)
            ]

        # 4. age / sex inference
        age_val: int | None = None
        m_age_years = RegexStore.Queries.AGE_YEARS.search(q_lower)
        if m_age_years:
            try:
                age_val = int(m_age_years.group(1))
            except ValueError:
                pass
        else:
            age_val = self._infer_age_from_date(query)

        if age_val:
            age_bin = _age_bin_from_number(age_val)
            if age_bin not in axes["age_bins"]:
                axes["age_bins"].append(age_bin)

        self._expand_rxnorm_umbrellas(axes)
        self._remove_substrings_and_duplicates(axes)

        for axis in config.AXES:
            if axis not in axes:
                axes[axis] = []

        print(axes)

        return axes



_global_axis_extractor: AxisExtractionService | None = None

def get_axis_extractor() -> AxisExtractionService:
    """
    Singleton AxisExtractionService instance.
    One to one with uvicorn workers.
    """
    global _global_axis_extractor
    if _global_axis_extractor is None:
        _global_axis_extractor = AxisExtractionService()
    return _global_axis_extractor 