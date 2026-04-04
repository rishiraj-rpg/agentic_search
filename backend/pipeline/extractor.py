"""
Entity Extractor Agent

Uses the LLM to pull structured entities + attributes out of scraped
page text.  Runs concurrently across all pages.

When `target_columns` is supplied (schema-first mode), the extractor
is told exactly which field names to use — this is the primary fix for
sparse/empty cells caused by inconsistent attribute key naming.

Every attribute value is tagged with source_url + source_title so the
final table is fully traceable cell-by-cell.
"""
from __future__ import annotations

import asyncio
from utils.llm_client import chat_json
from models import Entity, ScrapedPage, SourcedValue

# Used when no schema is pre-planned (fallback free-form extraction)
_SYSTEM_FREEFORM = """\
You are a structured data extraction assistant.

Given a web page's text and a topic query, extract all named entities
relevant to that topic (companies, products, places, tools, people, etc.).

For each entity, extract every attribute you can find, for example:
  company → founded, headquarters, funding, CEO, description
  restaurant → cuisine, price_range, rating, address
  tool → license, language, github_stars, description

Return a JSON object with key "entities" containing an array of objects.
Each object MUST have a "name" field. The rest of the fields should be the attributes.
For every attribute, you MUST return a nested object containing the "value" and a "confidence" score (0.0 to 1.0) indicating how explicitly the text links this value to the entity.
Example attribute field: "ceo": {{"value": "Tim Cook", "confidence": 0.95}}

Rules:
- Only include entities clearly relevant to the topic.
- Use concise snake_case attribute names.
- Attribute values must be short strings, not paragraphs.
- Omit a field if you cannot find its value — do not guess.
- Return at most 15 entities per page.
"""

# Used when a target schema is known (schema-first mode — preferred)
_SYSTEM_SCHEMA_TEMPLATE = """\
You are a structured data extraction assistant.

Given a web page's text and a topic query, extract all named entities
relevant to that topic (companies, products, places, tools, people, etc.).

For each entity, extract EXACTLY these attributes using these exact field names:
{columns}

Return a JSON object with key "entities" containing an array of objects.
Each object MUST have a "name" field.
For every other attribute, you MUST return a nested object containing the "value" and a "confidence" score (0.0 to 1.0) indicating how explicitly the text links this value to the entity.
Example attribute field: "ceo": {{"value": "Tim Cook", "confidence": 0.95}}

Rules:
- Only include entities clearly relevant to the topic.
- Use the exact field names provided above — do NOT invent new ones.
- Attribute values must be short strings, not paragraphs.
- Omit a field only if you genuinely cannot find its value on this page.
- Return at most 15 entities per page.
"""


async def extract_from_page(
    page: ScrapedPage,
    topic: str,
    target_columns: list[str] | None = None,
) -> list[Entity]:
    """Extract structured entities from one scraped page."""
    if target_columns:
        col_list = ", ".join(target_columns)
        system = _SYSTEM_SCHEMA_TEMPLATE.format(columns=col_list)
    else:
        system = _SYSTEM_FREEFORM

    user_prompt = (
        f"Topic: {topic}\n\n"
        f"Page title: {page.title}\n"
        f"Page URL: {page.url}\n\n"
        f"Page text:\n{page.text}"
    )
    try:
        result = await chat_json(system=system, user=user_prompt)
    except Exception:
        return []

    entities: list[Entity] = []
    for raw in result.get("entities", []):
        name = raw.pop("name", "")
        if isinstance(name, dict):
            name = name.get("value", "")
        name = str(name).strip()
        if not name:
            continue
        
        attrs: dict[str, SourcedValue] = {}
        for key, val_obj in raw.items():
            if not isinstance(val_obj, dict) or "value" not in val_obj:
                continue
            v_str = str(val_obj.get("value", "")).strip()
            conf = float(val_obj.get("confidence", 1.0))
            if v_str:
                attrs[key] = SourcedValue(
                    value=v_str,
                    confidence=conf,
                    source_url=page.url,
                    source_title=page.title,
                )
        entities.append(Entity(name=name, attributes=attrs))
    return entities


async def extract_all(
    pages: list[ScrapedPage],
    topic: str,
    target_columns: list[str] | None = None,
) -> list[list[Entity]]:
    """Run extraction concurrently over all pages."""
    return await asyncio.gather(
        *[extract_from_page(p, topic, target_columns) for p in pages]
    )
