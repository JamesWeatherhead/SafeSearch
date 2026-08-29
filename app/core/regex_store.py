from __future__ import annotations

import re

__all__ = ["RegexStore"]

# TODO: make the nested class names more relevant


class RegexStore:
    class PHI:
        SSN = re.compile(pattern=r"\b\d{3}-\d{2}-\d{4}\b")
        PHONE_10_DIGIT = re.compile(pattern=r"\b\d{10}\b")
        TITLED_NAME = re.compile(
            pattern=r"\b(?:mr|mrs|ms|dr)\.?\s+[A-Z][a-z]+",
            flags=re.IGNORECASE,
        )

    class Queries:
        GUIDELINE = re.compile(
            pattern=r"guidelines?\s+for(?:\s+the)?\s+(?:management\s+of|managing)?\s*([^,.;?(]+)",
            flags=re.IGNORECASE,
        )

        AGE_YEARS = re.compile(r"\b(\d{1,3})\s*[-\s]?year(?:[-\s]?old)?\b")
        
        DOB = re.compile(
            pattern=r"born\s+([A-Za-z]+\s+\d{1,2}[,\s]+\d{4})",
            flags=re.IGNORECASE,
        )

        FEMALE_SEX = re.compile(
            pattern=r"\b(female|woman|women|girl|lady|she|her)\b",
            flags=re.IGNORECASE,
        )

        MALE_SEX = re.compile(
            pattern=r"\b(male|man|men|boy|gentleman|he|him|his)\b",
            flags=re.IGNORECASE,
        )

    class Utils:
        NON_ALPHA_NUM = re.compile(pattern=r"[^a-z0-9\s]")
        INTEGER = re.compile(pattern=r"\b(\d+)\b")

        PHI_BLOCK = re.compile(
            pattern=r"===QUERY===\s*(.*?)\s*(?:===PHI_TAGS===\s*(.*?)\s*)?(?=(?:===QUERY===|\Z))",
            flags=re.DOTALL | re.IGNORECASE,
        )
        JSON_STRING_ARRAY = re.compile(
            pattern=r"\[\s*(?:\".*?\"\s*,\s*)*\".*?\"\s*\]|\[\s*\]",
            flags=re.DOTALL,
        )
        CITATION_BRACKET = re.compile(pattern=r"\[\d+\]") 