"""
Query Planner Agent

Decomposes a free-text topic into multiple targeted search queries
to maximise coverage and diversity of web results.

Also infers a target column schema from search snippets so the extractor
uses consistent attribute keys across all pages (prevents sparse tables).
"""
from __future__ import annotations

import os
from utils.llm_client import chat_json

_MAX_SUBQUERIES = int(os.getenv("MAX_SUBQUERIES", "4"))

_QUERY_SYSTEM = """\
You are a search query planning assistant.

Given a user's topic, generate targeted web search queries that will surface
a comprehensive list of specific named entities (companies, products, places,
tools, people, etc.) along with their key attributes.

Vary the queries across angles: rankings, comparisons, lists, reviews, directories.

Return ONLY a JSON object with key "queries" containing an array of strings.
"""

_SCHEMA_SYSTEM = """\
You are a data schema designer for a structured search results table.

Given a user's topic and web search snippets about that topic, choose the
best attribute columns for a results table.

Rules:
- Do NOT include "name" — it is always added automatically.
- Choose 6-8 attributes that are commonly available across entities in this topic.
- Prefer factual, structured attributes (numbers, dates, locations, categories).
- Use concise snake_case names (e.g. founded_year, headquarters, funding_stage, rating).
- Return ONLY a JSON object: {"columns": ["col1", "col2", ...]}
"""


async def plan_queries(topic: str) -> list[str]:
    """Return a list of targeted search query strings for the topic."""
    result = await chat_json(
        system=_QUERY_SYSTEM,
        user=f"Topic: {topic}\n\nGenerate {_MAX_SUBQUERIES} search queries.",
    )
    queries = result.get("queries", [])
    if not queries:
        queries = [topic]   # safe fallback
    return queries[:_MAX_SUBQUERIES]


async def plan_schema(topic: str, snippets: list[str]) -> list[str]:
    """
    Infer a target column schema from the topic and search result snippets.
    Returns an ordered list of snake_case attribute names (without 'name').
    Must be called BEFORE extraction so the extractor uses consistent keys.
    """
    snippet_text = "\n".join(f"- {s}" for s in snippets[:24] if s.strip())
    try:
        result = await chat_json(
            system=_SCHEMA_SYSTEM,
            user=f"Topic: {topic}\n\nSearch snippets:\n{snippet_text}\n\nChoose 6-8 attribute columns.",
        )
        cols = result.get("columns", [])
        cols = [c for c in cols if isinstance(c, str) and c != "name"]
        return cols[:8]
    except Exception:
        return []
