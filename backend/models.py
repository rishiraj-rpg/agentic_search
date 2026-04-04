"""
Pydantic models shared across the pipeline.
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class SourcedValue(BaseModel):
    """A single cell value tagged with its origin."""
    value: str
    source_url: str
    source_title: str
    confidence: float = 1.0


class Entity(BaseModel):
    """
    A discovered entity.
    `attributes` maps column name → sourced value.
    The key 'name' is always present as a plain field.
    """
    name: str
    attributes: dict[str, SourcedValue] = Field(default_factory=dict)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class ScrapedPage(BaseModel):
    url: str
    title: str
    text: str           # clean plain text, truncated
    char_count: int = 0


class PipelineResult(BaseModel):
    query: str
    columns: list[str]          # ordered attribute names (includes 'name')
    entities: list[Entity]
    total_pages_scraped: int
    total_sources: int
    llm_tokens: int = 0
    llm_cost: float = 0.0


class ExpandResult(BaseModel):
    query: str
    new_columns: list[str]          # columns just added
    columns: list[str]              # full updated column list
    entities: list[Entity]          # all entities with new attrs filled in
    cells_filled: int
    llm_tokens: int = 0
    llm_cost: float = 0.0


class PhaseEvent(BaseModel):
    phase: str      # planning | searching | scraping | extracting | merging | done
    message: str
    detail: str = ""
