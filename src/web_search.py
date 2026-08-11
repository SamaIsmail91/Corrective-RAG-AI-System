"""
Optional corrective fallback: if the document knowledge base has nothing
relevant even after query rewriting, the original Corrective-RAG paper's
answer is to fall back to a web search rather than hallucinate. This is
opt-in (off by default) and best-effort: no API key is used, it degrades
silently if the `duckduckgo-search` package isn't installed or there's no
network access, and results are graded exactly like document chunks before
they're ever allowed into the final answer.
"""

from __future__ import annotations

import importlib.util
from typing import Dict, List


def is_available() -> bool:
    return importlib.util.find_spec("duckduckgo_search") is not None


def web_search(query: str, max_results: int = 3) -> List[Dict]:
    """Returns chunk-shaped dicts so they can flow through the same grading
    and generation code path as document chunks. Returns [] on any failure
    (no network, package missing, rate limited, etc.) — callers should treat
    an empty list as "fallback unavailable", not as an error."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return []

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []

    out = []
    for r in raw_results:
        body = (r.get("body") or "").strip()
        if not body:
            continue
        out.append(
            {
                "id": r.get("href", "web-result"),
                "text": body,
                "metadata": {
                    "source": r.get("title", "Web result"),
                    "page": 0,
                    "url": r.get("href", ""),
                },
                "similarity": None,
            }
        )
    return out
