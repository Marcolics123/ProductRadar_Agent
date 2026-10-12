import os
from typing import Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from usage import ResearchPhase, record_search_results, reserve_search_call


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_QUERY_LENGTH = 500
DEFAULT_SEARCH_RESULTS = 8
MAX_SEARCH_RESULTS = 20
SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


def select_diverse_results(
    results: list[SearchResult],
    *,
    excluded_domains: set[str] | None = None,
    max_per_domain: int = 2,
) -> list[SearchResult]:
    """按域名控制结果集中度；补强证据时可排除已有来源。"""

    excluded = {
        domain.casefold().removeprefix("www.")
        for domain in (excluded_domains or set())
    }
    per_domain_count: dict[str, int] = {}
    selected: list[SearchResult] = []
    seen_urls: set[str] = set()
    for result in results:
        domain = (urlparse(result.url).hostname or "").casefold().removeprefix("www.")
        if not domain or domain in excluded or result.url in seen_urls:
            continue
        if per_domain_count.get(domain, 0) >= max(max_per_domain, 1):
            continue
        seen_urls.add(result.url)
        per_domain_count[domain] = per_domain_count.get(domain, 0) + 1
        selected.append(result)
    return selected


def search(
    query: str,
    *,
    max_results: int = DEFAULT_SEARCH_RESULTS,
    search_depth: SearchDepth = "basic",
    include_domains: list[str] | None = None,
    include_domains_mode: Literal["prefer", "restrict"] = "prefer",
    phase: ResearchPhase = "initial",
) -> list[SearchResult]:
    if not isinstance(query, str) or not query.strip():
        return []
    query = query.strip()
    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH]
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
    checked_domains = list(
        dict.fromkeys(
            domain.strip().lower()
            for domain in (include_domains or [])
            if isinstance(domain, str) and domain.strip()
        )
    )

    # advanced 搜索按 Tavily 当前计费计作两个 credits，其余深度计作一个。
    reserve_search_call(
        phase=phase,
        estimated_credits=2 if search_depth == "advanced" else 1,
    )
    request_body: dict[str, object] = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "chunks_per_source": 3,
        "topic": "general",
    }
    if checked_domains:
        request_body["include_domains"] = checked_domains
        request_body["include_domains_mode"] = include_domains_mode
    response = httpx.post(
        TAVILY_SEARCH_URL,
        headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
        json=request_body,
        timeout=30.0,
    )
    response.raise_for_status()

    try:
        response_data = response.json()
    except ValueError:
        return []
    if not isinstance(response_data, dict):
        return []
    raw_results = response_data.get("results", [])
    if not isinstance(raw_results, list):
        return []

    results: list[SearchResult] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue
        title = raw_result.get("title")
        url = raw_result.get("url")
        snippet = raw_result.get("content")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.strip():
            continue
        title = title.strip()
        url = url.strip()
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            continue
        if not isinstance(snippet, str):
            snippet = ""
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
            )
        )

    record_search_results(len(results))
    return results
