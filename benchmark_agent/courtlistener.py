"""CourtListener API client: unified search, opinion fetch, and citation lookup.

Functions return plain dicts/lists and raise on failure; the environment turns
results and errors into observations. `NonRetryableError` marks client errors
(malformed query, auth, not found) that a retry would not fix.
"""

import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import backoff
import requests as requests_original
import requests_cache
from dateutil import parser as date_parser

from .actions import SEARCH_TYPES
from .tracing import capture_http

logger = logging.getLogger(__name__)


class NonRetryableError(Exception):
    """Client error (400/401/403/404) that should not be retried."""


# Cache lives under the repo's .cache/ (gitignored) rather than the current
# working directory, so it lands in the same place no matter where a run starts.
CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache" / "courtlistener_cache.sqlite"

requests = requests_cache.CachedSession(
    str(CACHE_PATH),
    backend="sqlite",
    expire_after=timedelta(days=10),
)

COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
CITATION_LOOKUP_URL = f"{COURTLISTENER_BASE_URL}/citation-lookup/"
RATE_LIMIT_DELAY = 1  # seconds between requests
NON_RETRYABLE_STATUSES = (400, 401, 403, 404)


def _headers(user_agent: str) -> dict[str, str]:
    headers = {"User-Agent": user_agent}
    api_key = os.getenv("COURTLISTENER_API_KEY") or os.getenv("COURTLISTENER_API_TOKEN")
    if api_key:
        headers["Authorization"] = f"Token {api_key}"
    return headers


@backoff.on_exception(
    backoff.expo,
    requests_original.exceptions.RequestException,
    max_time=300,
)
def make_courtlistener_request(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET a V4 endpoint (e.g. "search/", "opinions/123/") with rate limiting and retries."""
    url = f"{COURTLISTENER_BASE_URL}/{endpoint}"
    headers = _headers(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    logger.debug(f"Making request to {url} with params: {params}")
    time.sleep(RATE_LIMIT_DELAY)
    with capture_http("request-courtlistener", "GET", url, params=params) as capture:
        response = requests.get(url, params=params, headers=headers)
        capture(response)
    if not response.ok:
        logger.error(f"CourtListener V4 API request failed: {response.status_code} - {response.text}")
        if response.status_code in NON_RETRYABLE_STATUSES:
            raise NonRetryableError(f"Semantic error: {response.status_code} - {response.text}")
        response.raise_for_status()  # 5xx: let backoff retry
    return response.json()


def fetch_opinion(opinion_id: str) -> dict[str, Any]:
    """Fetch a single opinion by CourtListener opinion ID."""
    return make_courtlistener_request(f"opinions/{opinion_id}/", {})


def search_courtlistener(query: str, search_type: str = "opinions", max_snippets: int = 5, **kwargs) -> dict[str, Any]:
    """Search CourtListener and return a summary of the top results.

    Returns {"count", "results": [...], "total_results", "snippets_shown", "search_type",
    "api_type"}; each result carries normalized fields plus the raw hit under "metadata".
    """
    if not query:
        raise ValueError("Query parameter is required for CourtListener search")
    if search_type not in SEARCH_TYPES:
        raise ValueError(f"Unsupported search type: {search_type}. Supported types: {list(SEARCH_TYPES.keys())}")
    params = {
        "q": query,
        "type": SEARCH_TYPES[search_type],
        "order_by": "score desc",
        "available_only": "on",
        **kwargs,
    }
    full_results = make_courtlistener_request("search/", params)

    hits = full_results.get("results", [])
    results = []
    for i, hit in enumerate(hits[:max_snippets]):
        # Opinion clusters may only carry the snippet on the nested opinion.
        snippet = hit.get("snippet") or (hit.get("opinions") or [{}])[0].get("snippet", "")
        results.append(
            {
                "id": hit.get("cluster_id") or hit.get("docket_id") or hit.get("id"),
                "case_name": hit.get("caseName", f"Case {i}"),
                "case_name_full": hit.get("caseNameFull"),
                "name": hit.get("name", ""),
                "court": hit.get("court", "Unknown Court"),
                "date_filed": hit.get("dateFiled", "Unknown Date"),
                "date_argued": hit.get("dateArgued", ""),
                "url": hit.get("absolute_url", ""),
                "snippet": snippet,
                "citations": "; ".join(hit.get("citation") or []),
                "docket_number": hit.get("docketNumber", ""),
                "document_type": hit.get("document_type", ""),
                "metadata": hit,
            }
        )
    return {
        "count": full_results.get("count", 0),
        "results": results,
        "total_results": len(hits),
        "snippets_shown": len(results),
        "search_type": search_type,
        "api_type": SEARCH_TYPES[search_type],
    }


def _sleep_until_iso(iso_timestamp: str) -> None:
    try:
        until = date_parser.isoparse(iso_timestamp)
    except Exception:
        return
    if until.tzinfo is None:
        until = until.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    delta = (until - now).total_seconds()
    if delta > 0:
        time.sleep(min(delta, 60))


def lookup_citation(cite: str, max_attempts: int = 3) -> list[dict[str, Any]]:
    """Resolve a reporter citation (e.g. '934 F.3d 53') via the citation-lookup API (POST).

    Returns the API's list of matches (possibly empty). Honors 429 `wait_until`.
    """
    headers = _headers("Mozilla/5.0 (compatible; courtlistener-citation-lookup/1.0)")
    headers["Accept"] = "application/json"
    for attempt in range(max_attempts):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            with capture_http("lookup-citation", "POST", CITATION_LOOKUP_URL, json={"text": cite}) as capture:
                resp = requests.post(
                    CITATION_LOOKUP_URL,
                    headers=headers,
                    json={"text": cite},
                    timeout=30,
                )
                capture(resp)
            if resp.status_code == 429:
                wait_until = (resp.json() if resp.text else {}).get("wait_until")
                if wait_until:
                    _sleep_until_iso(wait_until)
                else:
                    time.sleep(5)
                continue
            resp.raise_for_status()
            raw = resp.json()
            return raw if isinstance(raw, list) else []
        except requests_original.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in NON_RETRYABLE_STATUSES:
                raise NonRetryableError(str(e)) from e
            if attempt == max_attempts - 1:
                raise
            time.sleep(5)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            logger.warning(f"Citation lookup attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    return []
