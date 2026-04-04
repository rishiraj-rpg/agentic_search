"""
Gap Filler

After the initial extraction + merge, some entities may still have empty cells
because no single scraped page contained all attributes for that entity.

This targeted re-extraction pass scans all available scraped pages for
mentions of each under-populated entity and tries to fill specific missing
attributes with one focused LLM call per page.
"""
from __future__ import annotations

import asyncio
import logging
from models import Entity, ScrapedPage, SourcedValue
from utils.llm_client import chat_json

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a precise, entity-anchored data extraction assistant.

You will be given:
- A specific entity name and topic
- Known facts about that entity (to help you locate it in the text)
- A list of attribute fields to find for THAT entity specifically
- Text from a web page that may mention multiple entities

Your task: find the attribute values that belong to THIS specific entity only.

Critical rules:
- ONLY return attribute values that clearly belong to the named entity.
- If the page mentions multiple entities, isolate the section about THIS entity.
- Use the known facts to confirm you are reading about the correct entity.
- BEWARE of article authors, reviewers, or random names on the page. Do NOT extract them unless the text EXPLICITLY states they hold the requested attribute role (e.g. CEO, founder) for this specific entity.
- If the relationship is not explicitly stated in the text, you MUST omit the field.
- Do NOT guess. If you are not certain, omit the field.
- Attribute values must be short strings — not paragraphs.
- Return a JSON object with ONLY the fields you found, using the exact field names given.
- For every attribute, you MUST return a nested object containing the "value" and a "confidence" score (0.0 to 1.0) indicating how explicitly the text links this value to the entity.
- Example attribute field: "ceo": {"value": "Tim Cook", "confidence": 0.95}
"""


async def _fill_from_page(
    entity_name: str,
    missing_cols: list[str],
    page: ScrapedPage,
    sem: asyncio.Semaphore,
    topic: str = "",
    known_attrs: dict[str, str] | None = None,
) -> dict[str, str]:
    """Try to extract missing attrs from one page. Returns found key→value pairs."""
    async with sem:
        known_str = ""
        if known_attrs:
            facts = "; ".join(f"{k}: {v}" for k, v in known_attrs.items())
            known_str = f"Known facts about this entity: {facts}\n"

        user = (
            f"Topic: {topic}\n"
            f"Entity to look up: {entity_name}\n"
            f"{known_str}"
            f"Attributes to find for THIS entity: {', '.join(missing_cols)}\n\n"
            f"Page title: {page.title}\n"
            f"Page URL: {page.url}\n\n"
            f"Page text:\n{page.text}"
        )
        try:
            result = await chat_json(system=_SYSTEM, user=user)
            parsed_attrs = {}
            for k, val_obj in result.items():
                if k not in missing_cols:
                    continue
                if not isinstance(val_obj, dict) or "value" not in val_obj:
                    continue
                v_str = str(val_obj.get("value", "")).strip()
                conf = float(val_obj.get("confidence", 1.0))
                if v_str:
                    parsed_attrs[k] = {"value": v_str, "confidence": conf}
            return parsed_attrs
        except Exception:
            return {}


async def _fill_entity(
    entity: Entity,
    missing_cols: list[str],
    pages: list[ScrapedPage],
    sem: asyncio.Semaphore,
    topic: str = "",
) -> int:
    """Attempt to fill missing attrs for one entity. Returns number of cells filled."""
    # Only try pages that mention this entity (caps at 5 to improve recall)
    name_lower = entity.name.lower()
    relevant = [p for p in pages if name_lower in p.text.lower()][:5]
    if not relevant:
        return 0

    # Build known-attribute dict to anchor the LLM to the right entity
    known_attrs = {
        k: v.value
        for k, v in entity.attributes.items()
        if v.value
    }

    results = await asyncio.gather(*[
        _fill_from_page(entity.name, missing_cols, page, sem, topic, known_attrs)
        for page in relevant
    ])

    filled = 0
    for page, found in zip(relevant, results):
        for attr, attr_dict in found.items():
            val = attr_dict["value"]
            conf = attr_dict["confidence"]
            if attr in missing_cols and attr not in entity.attributes and val:
                entity.attributes[attr] = SourcedValue(
                    value=val,
                    confidence=conf,
                    source_url=page.url,
                    source_title=page.title,
                )
                filled += 1
    return filled


async def gap_fill(
    entities: list[Entity],
    target_columns: list[str],
    pages: list[ScrapedPage],
    min_empty_ratio: float = 0.4,
    topic: str = "",
) -> int:
    """
    Fill empty cells for entities missing >= min_empty_ratio of their attributes.

    Args:
        entities:        merged entity list from the pipeline
        target_columns:  expected attribute names (not including 'name')
        pages:           all scraped pages available for re-extraction
        min_empty_ratio: only gap-fill entities missing at least this fraction
        topic:           original search topic, used to anchor LLM lookups

    Returns:
        Total number of cells filled across all entities.
    """
    if not target_columns or not pages:
        return 0

    sem = asyncio.Semaphore(5)  # max 5 concurrent LLM calls during gap-fill

    tasks = []
    targeted: list[Entity] = []
    for entity in entities:
        missing = [col for col in target_columns if col not in entity.attributes]
        if missing and len(missing) / len(target_columns) >= min_empty_ratio:
            tasks.append(_fill_entity(entity, missing, pages, sem, topic))
            targeted.append(entity)

    if not tasks:
        logger.info("gap_fill: no entities needed filling")
        return 0

    counts = await asyncio.gather(*tasks)
    total = sum(counts)
    logger.info(
        "gap_fill: filled %d cells across %d/%d entities",
        total, len(targeted), len(entities),
    )
    return total

