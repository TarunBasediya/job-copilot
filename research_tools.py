import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.tools import tool

MAX_PAGE_CHARS = 6000  # ek page se itna hi text agent ko denge (context bachane ke liye)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo) and return titles, URLs and short snippets.
    Use this to find recent information about a company, its products,
    culture, funding or news. Returns up to max_results results."""
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"Search failed: {e}. Try a different query."

    if not results:
        return "No results found. Try a different query."

    lines = []
    for r in results:
        lines.append(f"- {r.get('title')}\n  URL: {r.get('href')}\n  {r.get('body')}")
    return "\n".join(lines)


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


@tool
def fetch_page(url: str) -> str:
    """Download a web page and return its readable text (truncated).
    Use this AFTER web_search, to read a promising result in more detail."""
    if not url.startswith(("http://", "https://")):
        return "Invalid URL: must start with http:// or https://"
    try:
        resp = httpx.get(
            url,
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (JobCopilot learning project)"},
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Could not fetch {url}: {e}"

    text = clean_html(resp.text)
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n...[truncated]"
    return text