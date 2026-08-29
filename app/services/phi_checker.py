from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from loguru import logger

from app.core.regex_store import RegexStore
from app.core.config import config

__all__ = ["PHICheckerService"]

PHI_REGEXES = [
    RegexStore.PHI.SSN,
    RegexStore.PHI.PHONE_10_DIGIT,
    RegexStore.PHI.TITLED_NAME,
]

class PHICheckerService:
    def __init__(self, phi_file: Path | None = None) -> None:
        self._phi_terms: List[str] = []
        self._phi_map: Dict[str, List[str]] = {}
        self._phi_map_cache = None
        phi_file = phi_file or Path(config.PHI_QUERIES_FILE)
        self._load_phi_file(phi_file)

    def _load_phi_file(self, phi_file: Path) -> None:
        try:
            if phi_file.exists():
                content = phi_file.read_text(encoding='utf-8')
                self._parse_phi_content(content)
                logger.info(f"Loaded {len(self._phi_map)} PHI definitions from '{str(phi_file)}'.")
            else:
                logger.warning(
                    f"PHI file not found at '{str(phi_file)}' –"
                    " proceeding with regex-only checks."
                )
        except Exception as exc:
            logger.error(f"Failed to load PHI file → {exc}")

    def _parse_phi_content(self, content: str) -> None:
        self._phi_terms = [
            line.strip().lower() 
            for line in content.splitlines() 
            if line.strip()
        ]
        
        for match in RegexStore.Utils.PHI_BLOCK.finditer(content):
            query_text_raw, phi_tags_block_str = match.groups()
            if not query_text_raw or query_text_raw.startswith("Example:") or "Generation failed:" in query_text_raw:
                continue
                
            normalized_query = self._normalize_query(query_text_raw)
            if not normalized_query:
                continue
            
            current_direct_phi_values = []
            if phi_tags_block_str:
                for line in phi_tags_block_str.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        tag = json.loads(line)
                        if tag.get("identifier_type") == "direct" and "value" in tag:
                            phi_value = str(tag["value"]).strip()
                            if phi_value:
                                current_direct_phi_values.append(phi_value)
                    except json.JSONDecodeError:
                        continue
            
            if current_direct_phi_values:
                self._phi_map.setdefault(normalized_query, [])
                for val in current_direct_phi_values:
                    if val not in self._phi_map[normalized_query]:
                        self._phi_map[normalized_query].append(val)

    @staticmethod
    def _normalize_query(query_text: str) -> str:
        return " ".join(query_text.split()).strip()

    def contains_phi(self, text: str) -> bool:
        if not text:
            return False
            
        text_lower = text.lower()
        
        if any(term in text_lower for term in self._phi_terms):
            return True
            
        for pattern in PHI_REGEXES:
            if pattern.search(text_lower):
                return True
                
        return False

    def check_phi_leakage(
        self, 
        original_query: str, 
        search_query: str
    ) -> Tuple[Optional[List[str]], str]:
        if not self._phi_map:
            return None, "PHI map not available (parsing failed)"
            
        normalized_original = self._normalize_query(original_query)
        if normalized_original not in self._phi_map:
            return None, "Original query not found in PHI map"
            
        direct_phi_values = self._phi_map[normalized_original]
        if not direct_phi_values:
            return [], "No direct PHI tags for this query in map"
            
        leaked_phi = []
        search_query_to_check = str(search_query or "").lower()
        
        for phi_val in direct_phi_values:
            if phi_val and phi_val.lower() in search_query_to_check:
                leaked_phi.append(phi_val)
                
        return (leaked_phi, "PHI LEAKAGE DETECTED") if leaked_phi else ([], "No PHI leakage detected") 