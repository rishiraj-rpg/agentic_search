"""
Async web scraper.

Fetches pages concurrently with httpx, strips boilerplate HTML with
BeautifulSoup, and returns clean plain text capped at MAX_TEXT_CHARS
to keep LLM token usage predictable.
"""
from __future__ import annotations

import asyncio
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from models import ScrapedPage

# Domains that are auth-walled or need JavaScript — skip them
_SKIP_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "linkedin.com", "tiktok.com", "youtube.com", "reddit.com",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# Max characters of text kept per page (controls LLM cost)
_MAX_TEXT_CHARS = 12_000


def _should_skip(url: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lstrip("www.")
    if domain in _SKIP_DOMAINS:
        return True
    if parsed.path.lower().endswith(".pdf"):
        return True
    return False


def _extract_text(html: str) -> str:
    """Strip tags and boilerplate; return collapsed plain text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "noscript", "svg", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s{2,}", " ", text)
    return text[:_MAX_TEXT_CHARS]


async def _fetch_one(client: httpx.AsyncClient, url: str) -> ScrapedPage | None:
    if _should_skip(url):
        return None
    try:
        resp = await client.get(url, headers=_HEADERS, follow_redirects=True, timeout=10)
        if resp.status_code >= 400:
            return None
        if "text/html" not in resp.headers.get("content-type", ""):
            return None

        html = resp.text
        text = _extract_text(html)
        if len(text) < 200:   # skip near-empty pages
            return None

        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string or url).strip() if soup.title else url
        return ScrapedPage(url=url, title=title, text=text, char_count=len(text))
    except Exception:
        return None


async def scrape_urls(urls: list[str], max_pages: int = 10) -> list[ScrapedPage]:
    """Fetch up to `max_pages` URLs concurrently; return successful results."""
    unique_urls = list(dict.fromkeys(urls))[:max_pages]
    async with httpx.AsyncClient() as client:
        pages = await asyncio.gather(*[_fetch_one(client, u) for u in unique_urls])
    return [p for p in pages if p is not None]
