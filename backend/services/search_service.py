"""
Search service — uses the Brave Search API.

Requires BRAVE_SEARCH_API_KEY to be set in the environment.
"""
from __future__ import annotations

import os
import httpx

from models import SearchResult


async def search(query: str, max_results: int = 8) -> list[SearchResult]:
    """Run a Brave Search query and return structured results."""
    return await _brave_search(query, max_results)


# ── Brave Search API ──────────────────────────────────────────────────────────

async def _brave_search(query: str, max_results: int) -> list[SearchResult]:
    api_key = os.environ["BRAVE_SEARCH_API_KEY"]
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {
        "q": query,
        "count": min(max_results, 20),   # Brave caps at 20
        "text_decorations": "false",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("web", {}).get("results", []):
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("description", ""),
        ))
    return results


