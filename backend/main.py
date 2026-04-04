"""
Agentic Search — FastAPI Backend
=================================
Run with:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from models import PipelineResult, ExpandResult, Entity, SourcedValue
from utils.cost_tracker import start_tracking, get_report
from pipeline.planner import plan_queries, plan_schema
from pipeline.extractor import extract_all
from pipeline.merger import merge_entities
from pipeline.gap_filler import gap_fill
from services.search_service import search
from services.scraper_service import scrape_urls

# ── Config ────────────────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "8"))
MAX_PAGES_TO_SCRAPE = int(os.getenv("MAX_PAGES_TO_SCRAPE", "10"))

# ── App ───────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agentic_search")

app = FastAPI(title="Agentic Search", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# ── Result cache (LRU, in-memory) ───────────────────────────────────────────────
_CACHE: OrderedDict[str, dict] = OrderedDict()
_PAGES_CACHE: OrderedDict[str, list] = OrderedDict()   # query → list[ScrapedPage]
_CACHE_MAX = 20


def _cache_get(query: str) -> dict | None:
    key = query.strip().lower()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return None


def _cache_set(query: str, value: dict) -> None:
    key = query.strip().lower()
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def _pages_cache_get(query: str):
    key = query.strip().lower()
    if key in _PAGES_CACHE:
        _PAGES_CACHE.move_to_end(key)
        return _PAGES_CACHE[key]
    return None


def _pages_cache_set(query: str, pages: list) -> None:
    key = query.strip().lower()
    _PAGES_CACHE[key] = pages
    _PAGES_CACHE.move_to_end(key)
    while len(_PAGES_CACHE) > _CACHE_MAX:
        _PAGES_CACHE.popitem(last=False)


# ── Static frontend ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(content=(FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "search_provider": "brave",
    }


# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Agentic Search endpoint (SSE stream) ──────────────────────────────────────

@app.get("/api/search")
async def agentic_search(q: str = Query(..., min_length=2, description="Topic query")):
    """
    Streams pipeline progress as SSE events, then emits a final 'result'
    event with the full structured table.

    SSE event types:
        phase  → {phase, message, detail}
        result → PipelineResult (full JSON)
        error  → {message}
        done   → {}
    """
    async def pipeline():
        start_tracking()
        try:
            # ── 1. Query Planning ─────────────────────────────────────────────
            cached = _cache_get(q)
            if cached:
                logger.info("Cache hit for query: %r", q)
                yield _sse("phase", {"phase": "planning",
                                      "message": "⚡ Returning cached result",
                                      "detail": "This query was already processed"})
                yield _sse("result", cached)
                yield _sse("done", {})
                return

            logger.info("Pipeline start — query: %r", q)
            yield _sse("phase", {"phase": "planning",
                                  "message": "Planning search queries…",
                                  "detail": f'Topic: "{q}"'})

            sub_queries = await plan_queries(q)

            yield _sse("phase", {"phase": "planning",
                                  "message": f"Generated {len(sub_queries)} search queries",
                                  "detail": " · ".join(sub_queries)})

            # ── 2. Search ─────────────────────────────────────────────────────
            yield _sse("phase", {"phase": "searching",
                                  "message": f"Searching the web across {len(sub_queries)} queries…",
                                  "detail": ""})

            search_batches = await asyncio.gather(
                *[search(sq, MAX_SEARCH_RESULTS) for sq in sub_queries]
            )

            # Deduplicate URLs while preserving order
            seen: set[str] = set()
            all_urls: list[str] = []
            for batch in search_batches:
                for r in batch:
                    if r.url not in seen:
                        seen.add(r.url)
                        all_urls.append(r.url)

            yield _sse("phase", {"phase": "searching",
                                  "message": f"Found {len(all_urls)} unique URLs",
                                  "detail": ""})

            # ── 2b. Schema Planning ───────────────────────────────────────────
            yield _sse("phase", {"phase": "planning",
                                  "message": "Inferring table schema from search results…",
                                  "detail": ""})

            all_snippets = [r.snippet for batch in search_batches for r in batch]
            target_columns = await plan_schema(q, all_snippets)

            yield _sse("phase", {"phase": "planning",
                                  "message": f"Schema fixed: {len(target_columns)} columns",
                                  "detail": ", ".join(target_columns)})

            # ── 3. Scraping ───────────────────────────────────────────────────
            yield _sse("phase", {"phase": "scraping",
                                  "message": f"Scraping up to {MAX_PAGES_TO_SCRAPE} pages…",
                                  "detail": ""})

            pages = await scrape_urls(all_urls, max_pages=MAX_PAGES_TO_SCRAPE)
            _pages_cache_set(q, pages)  # persist for /api/expand

            yield _sse("phase", {"phase": "scraping",
                                  "message": f"Scraped {len(pages)} pages successfully",
                                  "detail": " · ".join(p.title[:50] for p in pages[:4])})

            if not pages:
                yield _sse("error", {"message": "Could not scrape any pages. Try a different query."})
                return

            # ── 4. Entity Extraction ──────────────────────────────────────────
            yield _sse("phase", {"phase": "extracting",
                                  "message": f"Extracting entities from {len(pages)} pages with LLM…",
                                  "detail": "Running in parallel"})

            raw_batches = await extract_all(pages, topic=q, target_columns=target_columns)
            total_raw = sum(len(b) for b in raw_batches)

            yield _sse("phase", {"phase": "extracting",
                                  "message": f"Extracted {total_raw} raw entity mentions",
                                  "detail": ""})

            # ── 5. Merge + Schema Inference ───────────────────────────────────
            yield _sse("phase", {"phase": "merging",
                                  "message": "Deduplicating and merging entities…",
                                  "detail": ""})

            merged = merge_entities(raw_batches)
            columns = ["name"] + target_columns if target_columns else ["name"]

            yield _sse("phase", {"phase": "merging",
                                  "message": f"Merged to {len(merged)} unique entities · {len(columns)-1} attributes",
                                  "detail": "Columns: " + ", ".join(columns)})

            # ── 6. Gap Fill ──────────────────────────────────────────────────────────────
            yield _sse("phase", {"phase": "merging",
                                  "message": "Filling empty cells with targeted re-extraction…",
                                  "detail": f"Checking {len(merged)} entities for gaps"})

            filled = await gap_fill(merged, target_columns, pages, topic=q)

            if filled:
                yield _sse("phase", {"phase": "merging",
                                      "message": f"Gap fill: recovered {filled} additional cell values",
                                      "detail": ""})
            else:
                yield _sse("phase", {"phase": "merging",
                                      "message": "Gap fill: no additional values found",
                                      "detail": ""})

            # ── 6. Result ─────────────────────────────────────────────────────
            report = get_report()
            result = PipelineResult(
                query=q,
                columns=columns,
                entities=merged,
                total_pages_scraped=len(pages),
                total_sources=len(all_urls),
                llm_tokens=report.total_tokens if report else 0,
                llm_cost=report.estimated_cost if report else 0.0,
            )
            result_dict = result.model_dump()
            _cache_set(q, result_dict)
            logger.info(
                "Pipeline done — %d entities, %d pages, cache size %d",
                len(merged), len(pages), len(_CACHE),
            )
            yield _sse("result", result_dict)
            yield _sse("done", {})

        except Exception as exc:
            logger.exception("Pipeline error for query %r", q)
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        pipeline(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Expand Columns endpoint (SSE stream) ──────────────────────────────────────

@app.get("/api/expand")
async def expand_columns(
    q: str = Query(..., min_length=2, description="Original search query"),
    new_columns: str = Query(..., description="Comma-separated new column names"),
):
    """
    Adds new columns to an existing result table by running targeted gap-fill
    and optionally a fresh search+scrape pass if coverage is low.

    SSE event types:
        phase         → {phase, message, detail}
        expand_result → ExpandResult JSON
        error         → {message}
        done          → {}
    """
    async def expand_pipeline():
        start_tracking()
        try:
            # ── Validate & parse ──────────────────────────────────────────────
            col_names = [
                c.strip().lower().replace(" ", "_")
                for c in new_columns.split(",")
                if c.strip()
            ]
            if not col_names:
                yield _sse("error", {"message": "No valid column names provided."})
                return

            cached_result = _cache_get(q)
            if not cached_result:
                yield _sse("error", {"message": f'No cached result for "{q}". Run a search first.'})
                return

            # Deserialise entities from cache dict
            entities = [
                Entity(
                    name=e["name"],
                    attributes={
                        k: SourcedValue(**v)
                        for k, v in e.get("attributes", {}).items()
                    },
                )
                for e in cached_result["entities"]
            ]

            existing_columns: list[str] = cached_result["columns"]
            truly_new = [c for c in col_names if c not in existing_columns]
            if not truly_new:
                yield _sse("error", {"message": "All requested columns already exist in the table."})
                return

            yield _sse("phase", {
                "phase": "expanding",
                "message": f"Expanding table with {len(truly_new)} new column(s)…",
                "detail": ", ".join(truly_new),
            })

            # ── Per-Entity Targeted Expansion ─────────────────────────────────
            # The user requested accurate per-entity searches instead of relying on
            # previously scraped listicles to avoid pulling random matching names.
            
            cells_filled = 0
            # Limit to top 15 entities to avoid overwhelming search APIs on huge tables
            top_entities = entities[:15]
            
            from pipeline.gap_filler import _fill_from_page
            import asyncio
            
            # Shared semaphore for LLM calls across all entities
            llm_sem = asyncio.Semaphore(10)

            async def _process_entity(entity):
                filled_here = 0
                
                # 1. New Brave search for this entity + new attributes
                col_str = " ".join(c.replace("_", " ") for c in truly_new)
                sq = f'"{entity.name}" {col_str}'
                search_results = await search(sq, max_results=3)
                
                if not search_results:
                    return 0
                    
                urls = [r.url for r in search_results]
                
                # 2. Scrape those fresh pages
                pages = await scrape_urls(urls, max_pages=3)
                if not pages:
                    return 0
                
                # 3. Extraction for just these new columns
                known_attrs = {k: v.value for k, v in entity.attributes.items() if v.value}
                
                tasks = [
                    _fill_from_page(
                        entity_name=entity.name, 
                        missing_cols=truly_new, 
                        page=p, 
                        sem=llm_sem, 
                        topic=q, 
                        known_attrs=known_attrs
                    )
                    for p in pages
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Find valid values and attach SourcedValue
                for page, found in zip(pages, results):
                    for attr, val in found.items():
                        if attr in truly_new and attr not in entity.attributes and val:
                            entity.attributes[attr] = SourcedValue(
                                value=val["value"],
                                confidence=val.get("confidence", 1.0),
                                source_url=page.url,
                                source_title=page.title,
                            )
                            filled_here += 1
                return filled_here

            yield _sse("phase", {
                "phase": "searching",
                "message": f"Running targeted web searches for {len(top_entities)} entities...",
                "detail": "Searching e.g., 'Entity Name " + " ".join(truly_new) + "'"
            })
            
            # Execute per-entity search and extract concurrently
            counts = await asyncio.gather(*[_process_entity(e) for e in top_entities])
            cells_filled = sum(counts)
            
            yield _sse("phase", {
                "phase": "expanding",
                "message": f"Completed targeted searches and extracted {cells_filled} new cells",
                "detail": ""
            })

            # ── Update result cache with new columns ──────────────────────────
            updated_columns = existing_columns + truly_new
            
            report = get_report()
            
            updated_result = dict(cached_result)
            updated_result["columns"] = updated_columns
            updated_result["entities"] = [e.model_dump() for e in entities]
            
            # Optionally add to previously accumulated cost, but for ExpandResult we return the marginal cost.
            # We must accumulate it for the total cache block though.
            updated_result["llm_tokens"] = updated_result.get("llm_tokens", 0) + (report.total_tokens if report else 0)
            updated_result["llm_cost"] = updated_result.get("llm_cost", 0.0) + (report.estimated_cost if report else 0.0)
            
            _cache_set(q, updated_result)

            expand_result = ExpandResult(
                query=q,
                new_columns=truly_new,
                columns=updated_columns,
                entities=entities,
                cells_filled=cells_filled,
                llm_tokens=report.total_tokens if report else 0,
                llm_cost=report.estimated_cost if report else 0.0,
            )
            yield _sse("expand_result", expand_result.model_dump())
            yield _sse("done", {})

        except Exception as exc:
            logger.exception("Expand error for query %r", q)
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        expand_pipeline(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
