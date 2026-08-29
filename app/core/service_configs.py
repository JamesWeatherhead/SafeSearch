from typing import Set, List
from dataclasses import dataclass

__all__ = [
    "RerankingConfig",
    "ResponseGeneratorConfig", 
    "AxisExtractionConfig",
    "VectorSearchConfig",
    "PhiConfig",
]


@dataclass(frozen=True)
class RerankingConfig:
    COSINE_WEIGHT: float = 0.65
    LLM_WEIGHT: float = 0.25
    LEXICAL_WEIGHT: float = 0.10
    
    KEEP_TOP_K: int = 3
    MAX_THREADS: int = 4
    
    MULTI_MATCH_AXES: Set[str] = frozenset({
        "intent_terms", 
        "rxnorm_terms"
    })


@dataclass(frozen=True)
class ResponseGeneratorConfig:
    MAX_RETRIES: int = 3
    MAX_TOKENS: int = 1200
    TEMPERATURE: float = 0.15
    
    RETRY_DELAY_BASE: float = 1.0
    RETRY_DELAY_MULTIPLIER: float = 2.0


@dataclass(frozen=True)
class AxisExtractionConfig:
    STARTING_PRIORITY: int = 1
    MAX_TOKENS: int = 600
    TEMPERATURE: float = 0.0
    
    DEFAULT_INTENTS: List[str] = None
    
    def __post_init__(self):
        if self.DEFAULT_INTENTS is None:
            object.__setattr__(self, 'DEFAULT_INTENTS', ["treatment"])


@dataclass(frozen=True)
class VectorSearchConfig:
    DEFAULT_TOP_K: int = 3
    DEFAULT_COSINE_THRESHOLD: float = 0.60
    
    FALLBACK_AXES: Set[str] = frozenset({
        "anatomy_terms",
        "comorbidity_terms", 
        "diagnosis_terms",
        "family_history_terms",
        "intent_terms",
        "procedure_terms",
        "symptom_terms",
    })


@dataclass(frozen=True)
class PhiConfig:
    STRICT_MODE: bool = True
    ALLOW_PARTIAL_MATCHES: bool = False 