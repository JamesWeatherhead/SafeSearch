from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from enum import StrEnum
from typing import List

__all__ = ["config"]


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

    ENVIRONMENT: str
    ROOT_DIR: Path = Path(__file__).parent.parent.parent

    DB_HOST: str
    DB_PORT: str
    DB_NAME: str
    DB_USERNAME: str
    DB_PASSWORD: str
    DB_CONNECT_TIMEOUT: str
    DB_DSN: str

    MAX_RERANK_WORKERS: int = 4

    PPLX_API_KEY: str
    PPLX_API_ENDPOINT: str

    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: str
    AZURE_OPENAI_CHAT_MODEL: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str
    AZURE_OPENAI_EMBEDDING_MODEL: str

    PHI_QUERIES_FILE: str = "app/data/queries/phi_queries_tagged.txt"

    AXES: List[str] = [ # clinical categories
        "age_bins",
        "allergy_terms",
        "anatomy_terms",
        "comorbidity_terms",
        "diagnosis_terms",
        "family_history_terms",
        "intent_terms",
        "lifestyle_terms",
        "procedure_terms",
        "race_ethnicity",
        "rxnorm_terms",
        "severity_status",
        "sex_terms",
        "symptom_terms",
        "temporal_context",
        "wordlist_terms",
    ]

config = Config()