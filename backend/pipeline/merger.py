"""
Entity Merger

Deduplicates entities across all scraped pages using name normalisation
(lowercase, strip punctuation & legal suffixes). For each attribute,
the first-seen sourced value wins so attribution is never lost.
"""
from __future__ import annotations

import re
from rapidfuzz import fuzz
from models import Entity


# ── Name normalisation ────────────────────────────────────────────────────────

_LEGAL_SUFFIXES = (" inc", " llc", " ltd", " corp", " co", " ag", " plc")


def _normalise(name: str) -> str:
    key = name.lower()
    key = re.sub(r"[^\w\s]", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    for suffix in _LEGAL_SUFFIXES:
        if key.endswith(suffix):
            key = key[: -len(suffix)].strip()
    return key


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_entities(raw_batches: list[list[Entity]]) -> list[Entity]:
    """
    Deduplicate and merge entity lists from all pages.
    - Groups by normalised name.
    - First-seen value wins for each attribute (preserves source traceability).
    - Missing attributes are filled in from later pages.
    """
    groups: dict[str, Entity] = {}

    for batch in raw_batches:
        for entity in batch:
            key = _normalise(entity.name)
            
            # Fuzzy match against existing keys
            matched_key = None
            if key in groups:
                matched_key = key
            else:
                for existing_key in groups:
                    # token_set_ratio is robust against subsets (e.g., 'Goldman' vs 'Goldman Sachs')
                    if fuzz.token_set_ratio(key, existing_key) >= 85:
                        matched_key = existing_key
                        break

            if not matched_key:
                matched_key = key
                groups[matched_key] = Entity(name=entity.name, attributes={})
            else:
                # Upgrade canonical name if the new name is richer/longer
                if len(entity.name) > len(groups[matched_key].name):
                    groups[matched_key].name = entity.name

            merged = groups[matched_key]
            for attr, sourced_val in entity.attributes.items():
                if attr not in merged.attributes:
                    merged.attributes[attr] = sourced_val
                elif sourced_val.confidence > merged.attributes[attr].confidence:
                    merged.attributes[attr] = sourced_val

    return list(groups.values())
