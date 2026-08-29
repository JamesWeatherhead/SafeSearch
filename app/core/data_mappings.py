from typing import Dict, List, Set
from dataclasses import dataclass, field

__all__ = ["AxisExtractionMappings"]


@dataclass(frozen=True)
class AxisExtractionMappings:
    INTENT_SYNONYMS: Dict[str, List[str]] = field(default_factory=lambda: {
        "diagnosis": ["diagnosis", "diagnose", "diagnostic"],
        "differential diagnosis": ["differential diagnosis", "ddx", "differential"],
        "treatment": ["treatment", "treat", "therapy", "manage", "management"],
        "management": ["management", "management plan"],
        "risk assessment": ["risk assessment", "risk stratification", "risk"],
        "screening": ["screening", "screen for", "population screening"],
        "prognosis": ["prognosis", "prognostic", "outlook"],
        "guideline": ["guideline", "recommendation", "consensus statement", "protocol"],
    })
    
    RXNORM_UMBRELLAS: Dict[str, List[str]] = field(default_factory=lambda: {
        "anticoagulation therapy": ["warfarin", "apixaban", "rivaroxaban"],
        "antibiotic therapy": ["amoxicillin", "doxycycline", "azithromycin"],
        "hypertension management": ["lisinopril", "amlodipine", "hydrochlorothiazide"],
        "diabetes management": ["metformin", "insulin glargine", "semaglutide"],
        "asthma exacerbation": ["albuterol", "budesonide", "prednisone"],
        "pain management": ["ibuprofen", "acetaminophen", "oxycodone"],
    })
    
    LITERAL_FILTER_AXES: Set[str] = field(default_factory=lambda: {
        "anatomy_terms",
        "comorbidity_terms",
        "diagnosis_terms",
        "family_history_terms",
        "procedure_terms",
        "rxnorm_terms",
        "symptom_terms",
        "temporal_context",
        "severity_status",
        "race_ethnicity",
        "lifestyle_terms",
        "allergy_terms",
        "wordlist_terms",
    })
    
    DEFAULT_INTENTS: List[str] = field(default_factory=lambda: ["treatment"])
    
    @property
    def intent_priority_map(self) -> Dict[str, int]:
        STARTING_PRIORITY = 1
        return {
            intent: priority 
            for priority, intent in enumerate(
                self.INTENT_SYNONYMS.keys(), 
                STARTING_PRIORITY
            )
        } 