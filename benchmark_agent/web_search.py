"""Open web search via SerpAPI: date parsing, the client, and the search entry point.

The paper (Table A4) describes OPEN_WEB_SEARCH as "Searches the web with SerpAPI";
its trajectories show Google web search and Google Scholar. We also keep SerpAPI's
Google News engine because the action schema advertises `search_type='news'`.
The alternative, closer to what the paper's trajectories show, is to drop `news`
and remove it from the OpenWebSearch action description. Revisit if our action
usage or results diverge from the paper's Table A7.

Errors propagate to the environment, which turns them into an error observation;
a failed search must not look like "0 results" to the agent.
"""

import logging
import os
import random
import re
import string
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests
from dateutil import parser as dateutil_parser
from dateutil.tz import tzutc
from dotenv import load_dotenv

from .tracing import capture_http

load_dotenv()

logger = logging.getLogger(__name__)


# --- Date parsing for search result timestamps ---


_RELATIVE_UNITS = {
    "minute": lambda n: timedelta(minutes=n),
    "hour": lambda n: timedelta(hours=n),
    "day": lambda n: timedelta(days=n),
    "week": lambda n: timedelta(weeks=n),
    "month": lambda n: timedelta(days=30 * n),  # Approximate
    "year": lambda n: timedelta(days=365 * n),  # Approximate
}


def parse_date_string(date_str: str | None) -> datetime | None:
    """Parse SerpAPI date strings ("2 days ago", "07/22/2025, 06:39 PM, +0000 UTC",
    ISO 8601, ...) into a timezone-aware datetime, or None if unparseable."""
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.strip()
    try:
        parsed = _parse_relative_date(date_str) or _parse_serpapi_news_format(date_str)
        if parsed is None:
            parsed = dateutil_parser.parse(date_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tzutc())
        return parsed
    except Exception as e:
        logger.warning(f"Failed to parse date string '{date_str}': {e}")
        return None


def _parse_relative_date(date_str: str) -> datetime | None:
    """Parse "14 hours ago", "yesterday", "just now" relative to now."""
    lower = date_str.lower()
    now = datetime.now(tzutc())
    match = re.search(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", lower)
    if match:
        return now - _RELATIVE_UNITS[match.group(2)](int(match.group(1)))
    if "yesterday" in lower:
        return now - timedelta(days=1)
    if "today" in lower or "just now" in lower or "moments ago" in lower:
        return now
    return None


def _parse_serpapi_news_format(date_str: str) -> datetime | None:
    """Parse SerpAPI news format: "07/22/2025, 06:39 PM, +0000 UTC"."""
    match = re.match(
        r"(\d{2}/\d{2}/\d{4}),\s+(\d{1,2}:\d{2}\s+[AP]M),\s+([+-]\d{4})\s+UTC", date_str
    )
    if not match:
        return None
    date_part, time_part, tz_offset = match.groups()
    dt = datetime.strptime(f"{date_part} {time_part}", "%m/%d/%Y %I:%M %p")
    sign = -1 if tz_offset.startswith("-") else 1
    offset = timedelta(hours=int(tz_offset[1:3]), minutes=int(tz_offset[3:]))
    return dt.replace(tzinfo=tzutc()) - sign * offset


# --- SerpAPI client ---


class SerpApiClient:
    """Client for SerpApi search engines with exponential backoff on rate limits."""

    BASE_URL = "https://serpapi.com/search"

    def __init__(
        self, api_key: str | None = None, max_retries: int = 3, base_delay: float = 1.0
    ):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SerpApi API key is required. Set SERPAPI_API_KEY environment variable or pass api_key parameter."
            )
        self.base_url = self.BASE_URL
        self.max_retries = max_retries
        self.base_delay = base_delay

    def search(
        self,
        query: str,
        engine: str = "google",
        cutoff_date: date | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if cutoff_date:
            query = f"{query} before:{cutoff_date.strftime('%Y-%m-%d')}"
        params = {"q": query, "engine": engine, "api_key": self.api_key, **kwargs}
        return self._make_request_with_retry(self.base_url, params)

    def _make_request_with_retry(
        self, url: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """GET with retries on 429 and connection errors; other HTTP errors raise immediately."""
        for attempt in range(self.max_retries + 1):
            try:
                with capture_http(
                    "request-serpapi", "GET", url, params=params
                ) as capture:
                    response = requests.get(url, params=params)
                    capture(response)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as e:
                if response.status_code != 429:
                    logger.error(
                        f"SerpApi request failed with status {response.status_code}: {e}"
                    )
                    raise
                error, reason = e, "rate limit hit"
            except requests.RequestException as e:
                error, reason = e, f"connection error: {e}"
            if attempt == self.max_retries:
                logger.error(
                    f"SerpApi request failed after {self.max_retries + 1} attempts ({reason})"
                )
                raise error
            delay = self.base_delay * (2**attempt) + random.uniform(0, 1)
            logger.warning(
                f"SerpApi {reason} (attempt {attempt + 1}/{self.max_retries + 1}). Retrying in {delay:.2f}s"
            )
            time.sleep(delay)

    def google_search(
        self,
        query: str,
        num_results: int = 10,
        cutoff_date: date | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return self.search(
            query, engine="google", num=num_results, cutoff_date=cutoff_date, **kwargs
        )

    def google_scholar_search(self, query: str, **kwargs) -> dict[str, Any]:
        return self.search(query, engine="google_scholar", **kwargs)

    def news_search(
        self, query: str, cutoff_date: date | None = None, **kwargs
    ) -> dict[str, Any]:
        return self.search(
            query, engine="google_news", cutoff_date=cutoff_date, **kwargs
        )


# --- Search results and search entry point ---


class SearchResult:
    """Standardized search result format."""

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source: str,
        published_date: str | None = None,
        metadata: dict[str, Any] | None = None,
        result_id: str | None = None,
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.published_date_raw = published_date  # Store original string
        self.published_date = parse_date_string(published_date)  # Parse to datetime
        self.metadata = metadata or {}
        self.result_id = result_id or "".join(
            random.choices(string.ascii_letters + string.digits, k=5)
        )

    def __repr__(self) -> str:
        date_str = f" ({self.published_date_raw})" if self.published_date_raw else ""
        return f"{self.title}{date_str}\n{self.snippet}\n{self.url}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_date": self.published_date.isoformat()
            if self.published_date
            else None,
            "published_date_raw": self.published_date_raw,
            "metadata": self.metadata,
        }


def _filter_by_cutoff_date(
    results: list[SearchResult], cutoff_date: date | None, exclude_undated: bool = True
) -> list[SearchResult]:
    """Drop results published on or after cutoff_date; drop undated results if exclude_undated."""
    kept = []
    for result in results:
        if not result.published_date:
            if not exclude_undated:
                kept.append(result)
            continue
        if cutoff_date:
            tz = result.published_date.tzinfo or datetime.now().astimezone().tzinfo
            if result.published_date >= datetime.combine(
                cutoff_date, datetime.min.time(), tzinfo=tz
            ):
                continue
        kept.append(result)
    return kept


def search(
    query: str,
    search_type: str = "web",
    num_results: int = 10,
    cutoff_date: date | None = None,
    exclude_undated: bool = False,
    **kwargs,
) -> list[SearchResult]:
    """Search via SerpAPI. search_type is "web", "news", or "google_scholar".

    Raises on client/API failure so the caller can report the error to the agent.
    """
    client = SerpApiClient()
    if search_type == "web":
        raw = client.google_search(
            query, num_results=num_results, cutoff_date=cutoff_date, **kwargs
        )
        results = [_web_result(r) for r in raw.get("organic_results", [])]
    elif search_type == "news":
        raw = client.news_search(query, cutoff_date=cutoff_date, **kwargs)
        results = [_news_result(a) for a in raw.get("news_results", [])]
    elif search_type == "google_scholar":
        params = {
            "as_ylo": "1900",
            "as_yhi": "2099",
            "num": num_results,
            "hl": kwargs.get("hl", "en"),
        }
        raw = client.google_scholar_search(query, **params)
        results = [_scholar_result(a) for a in raw.get("organic_results", [])]
    else:
        raise ValueError(f"Unsupported search type: {search_type}")
    return _filter_by_cutoff_date(results, cutoff_date, exclude_undated)[:num_results]


def _web_result(result: dict[str, Any]) -> SearchResult:
    return SearchResult(
        title=result.get("title", ""),
        url=result.get("link", ""),
        snippet=result.get("snippet", ""),
        source="google",
        published_date=result.get("date")
        or result.get("publication_date")
        or result.get("published_at"),
        metadata={
            "position": result.get("position"),
            "displayed_link": result.get("displayed_link"),
            "favicon": result.get("favicon"),
            "source_info": result.get("source"),
            "snippet_highlighted_words": result.get("snippet_highlighted_words", []),
        },
    )


def _news_result(article: dict[str, Any]) -> SearchResult:
    source_info = article.get("source", {})
    is_dict = isinstance(source_info, dict)
    return SearchResult(
        title=article.get("title", ""),
        url=article.get("link", ""),
        # SerpAPI news rarely provides a separate snippet; fall back to the title.
        snippet=article.get("snippet", "") or article.get("title", ""),
        source=source_info.get("name", "") if is_dict else str(source_info),
        published_date=article.get("date"),  # "07/22/2025, 01:10 PM, +0000 UTC"
        metadata={
            "position": article.get("position"),
            "thumbnail": article.get("thumbnail"),
            "thumbnail_small": article.get("thumbnail_small"),
            "authors": source_info.get("authors", []) if is_dict else [],
            "source_icon": source_info.get("icon", "") if is_dict else "",
        },
    )


def _scholar_result(article: dict[str, Any]) -> SearchResult:
    publication_info = article.get("publication_info", {})
    pdf_link = next(
        (
            r.get("link")
            for r in article.get("resources", [])
            if r.get("file_format") == "PDF"
        ),
        None,
    )
    return SearchResult(
        title=article.get("title", "N/A"),
        url=article.get("link", "N/A"),
        snippet=article.get("snippet", "N/A"),
        source="google_scholar",
        published_date=None,
        metadata={
            "authors": [
                a.get("name", "Unknown") for a in publication_info.get("authors", [])
            ],
            "pdf_link": pdf_link or "",
            "cited_by": article.get("inline_links", {})
            .get("cited_by", {})
            .get("total", 0),
            "position": article.get("position", 0),
            "journal_venue": publication_info.get("summary", "N/A"),
        },
    )
