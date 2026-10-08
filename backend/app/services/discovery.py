"""M8.1 — GitHub Repository Discovery.

Pipeline (metadata only — never analysis evidence):

    User query
        ↓
    Validation
        ↓
    TTL cache (in-process, 5 min)
        ↓
    GitHub REST API (GET /search/repositories)
        ↓
    Normalization (GitHubRepository)
        ↓
    RepositorySearchResult / TrendingResult

Trending is Northern Star's API-derived discovery ranking, NOT an official
GitHub ranking. It merges two curated GitHub searches (recently created
popular repos + recently pushed active repos), dedupes, and orders by a
deterministic trend score documented on ``trend_score``.

Trend score formula (each component in [0, 1], weights sum to 1):

    S = log10(stars + 1) / log10(max_stars + 1)   (popularity)
    F = log10(forks + 1) / log10(max_forks + 1)   (adoption)
    R = max(0, 1 - days_since_push / 365)           (recency)

    trend_score = round(0.5 * S + 0.25 * F + 0.25 * R, 4)

``rank_change`` is always None: M8.1 keeps no historical snapshots, so it
must not pretend to know history (that belongs to M8.2).
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from ..config import Settings
from ..models.schemas import (
    GitHubRepository,
    RepositorySearchResult,
    TrendingRepository,
    TrendingResult,
)

SEARCH_PATH = "/search/repositories"
GITHUB_API_VERSION = "2022-11-28"

MAX_QUERY_LENGTH = 256
MAX_PER_PAGE = 30
MAX_TRENDING_LIMIT = 100
CACHE_TTL_SECONDS = 300

VALID_SORTS = {"best-match", "stars", "forks", "updated"}
VALID_ORDERS = {"asc", "desc"}


# ---------------------------------------------------------------------------
# Errors (typed so the API layer maps them to clean HTTP codes)
# ---------------------------------------------------------------------------

class DiscoveryError(Exception):
    """Base class for every discovery failure."""


class DiscoveryValidationError(DiscoveryError, ValueError):
    """The user's request parameters are malformed."""


class DiscoveryRateLimitedError(DiscoveryError):
    """GitHub rate limit hit (HTTP 403 with exhausted quota, or 429)."""


class DiscoveryUpstreamError(DiscoveryError):
    """GitHub returned 5xx (after one bounded retry) or an unusable payload."""


class DiscoveryTimeoutError(DiscoveryError):
    """The GitHub request exceeded the configured timeout."""


class DiscoveryConnectionError(DiscoveryError):
    """Could not reach GitHub at all (DNS, connection refused, ...)."""


# ---------------------------------------------------------------------------
# Validation (pure; raises DiscoveryValidationError)
# ---------------------------------------------------------------------------

def validate_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise DiscoveryValidationError("Query must not be empty.")
    if len(query) > MAX_QUERY_LENGTH:
        raise DiscoveryValidationError(
            f"Query must be at most {MAX_QUERY_LENGTH} characters."
        )
    return query  # preserved verbatim (only surrounding whitespace trimmed below)


def validate_search_params(
    query: str,
    page: int = 1,
    per_page: int = 10,
    sort: str = "best-match",
    order: str = "desc",
) -> tuple[str, int, int, str, str]:
    q = validate_query(query).strip()
    if not q:
        raise DiscoveryValidationError("Query must not be empty.")
    if not isinstance(page, int) or page < 1:
        raise DiscoveryValidationError("page must be an integer >= 1.")
    if not isinstance(per_page, int) or not 1 <= per_page <= MAX_PER_PAGE:
        raise DiscoveryValidationError(
            f"per_page must be an integer between 1 and {MAX_PER_PAGE}."
        )
    if sort not in VALID_SORTS:
        raise DiscoveryValidationError(
            f"sort must be one of: {', '.join(sorted(VALID_SORTS))}."
        )
    if order not in VALID_ORDERS:
        raise DiscoveryValidationError("order must be 'asc' or 'desc'.")
    return q, page, per_page, sort, order


def validate_trending_limit(limit: int) -> int:
    if not isinstance(limit, int) or not 1 <= limit <= MAX_TRENDING_LIMIT:
        raise DiscoveryValidationError(
            f"limit must be an integer between 1 and {MAX_TRENDING_LIMIT}."
        )
    return limit


# ---------------------------------------------------------------------------
# Normalization (GitHub raw item → GitHubRepository; never leaks raw fields)
# ---------------------------------------------------------------------------

def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_repo(item: dict[str, Any]) -> GitHubRepository:
    """Convert one GitHub search item to Northern Star's curated schema."""
    if not isinstance(item, dict):
        raise DiscoveryUpstreamError("GitHub returned a malformed repository entry.")
    owner_info = item.get("owner") or {}
    license_info = item.get("license") or {}
    topics = item.get("topics") or []
    full_name = item.get("full_name") or ""
    fallback_owner = full_name.split("/")[0] if "/" in full_name else ""
    return GitHubRepository(
        id=_as_int(item.get("id")),
        full_name=full_name,
        name=item.get("name") or full_name.split("/")[-1],
        owner=owner_info.get("login") or fallback_owner,
        html_url=item.get("html_url") or "",
        description=item.get("description"),
        language=item.get("language"),
        stars=_as_int(item.get("stargazers_count")),
        forks=_as_int(item.get("forks_count")),
        open_issues=_as_int(item.get("open_issues_count")),
        watchers=_as_int(item.get("watchers_count")),
        topics=[t for t in topics if isinstance(t, str)][:20],
        default_branch=item.get("default_branch"),
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
        pushed_at=item.get("pushed_at"),
        license=license_info.get("name") if isinstance(license_info, dict) else None,
        archived=bool(item.get("archived", False)),
        fork=bool(item.get("fork", False)),
    )


# ---------------------------------------------------------------------------
# GitHub API client (httpx; transport injectable for offline tests)
# ---------------------------------------------------------------------------

class GitHubClient:
    """Minimal client for GitHub's repository search endpoint."""

    def __init__(
        self,
        settings: Settings,
        transport: Optional[Any] = None,
    ) -> None:
        self.base_url = settings.github_api_base_url.rstrip("/")
        self.timeout = settings.github_timeout_seconds
        # Read once: the token value never leaves this object except inside
        # the Authorization header. It is never logged, stored, or returned.
        self._token = settings.github_token
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "northern-star-discovery",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def search_repositories(
        self,
        q: str,
        *,
        sort: Optional[str] = None,
        order: str = "desc",
        page: int = 1,
        per_page: int = 10,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Return (total_count, items) from GET /search/repositories."""
        params: dict[str, Any] = {
            "q": q, "order": order, "page": page, "per_page": per_page,
        }
        if sort:  # None → GitHub best-match relevance
            params["sort"] = sort
        return self._get_with_retry(SEARCH_PATH, params)

    def _get_with_retry(
        self, path: str, params: dict[str, Any]
    ) -> tuple[int, list[dict[str, Any]]]:
        """One bounded retry for transient 5xx; rate limits are never retried."""
        payload, status = self._get(path, params)
        if status is not None and 500 <= status <= 599:
            payload, status = self._get(path, params)
            if status is not None and 500 <= status <= 599:
                raise DiscoveryUpstreamError(
                    "GitHub is temporarily unavailable. Try again later."
                )
        total, items = self._parse_search_payload(payload)
        return total, items

    def _get(
        self, path: str, params: dict[str, Any]
    ) -> tuple[dict[str, Any], Optional[int]]:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self._transport,
            ) as client:
                response = client.get(path, params=params, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise DiscoveryTimeoutError(
                "GitHub request timed out. Try again later."
            ) from exc
        except httpx.RequestError as exc:
            raise DiscoveryConnectionError(
                "Could not reach GitHub. Check network connectivity."
            ) from exc
        if response.status_code == 429 or _is_rate_limited(response):
            raise DiscoveryRateLimitedError(_rate_limit_message(response))
        if 500 <= response.status_code <= 599:
            return {}, response.status_code
        if response.status_code != 200:
            raise DiscoveryUpstreamError(
                f"GitHub request failed (HTTP {response.status_code})."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise DiscoveryUpstreamError(
                "GitHub returned an unreadable response."
            ) from exc
        if not isinstance(payload, dict):
            raise DiscoveryUpstreamError("GitHub returned a malformed response.")
        return payload, response.status_code

    @staticmethod
    def _parse_search_payload(
        payload: dict[str, Any],
    ) -> tuple[int, list[dict[str, Any]]]:
        total = payload.get("total_count", 0)
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise DiscoveryUpstreamError("GitHub returned a malformed response.")
        return _as_int(total), [i for i in items if isinstance(i, dict)]


def _is_rate_limited(response: httpx.Response) -> bool:
    if response.status_code != 403:
        return False
    if response.headers.get("X-RateLimit-Remaining") == "0":
        return True
    try:
        body = response.json()
        message = str(body.get("message", "")).lower() if isinstance(body, dict) else ""
    except ValueError:
        return False
    return "rate limit" in message or "abuse" in message


def _rate_limit_message(response: httpx.Response) -> str:
    reset = response.headers.get("X-RateLimit-Reset")
    if reset and reset.isdigit():
        when = datetime.fromtimestamp(int(reset), tz=timezone.utc).isoformat()
        return f"GitHub rate limit exceeded. Quota resets at {when}."
    return "GitHub rate limit exceeded. Try again later."


# ---------------------------------------------------------------------------
# In-process TTL cache (discovery metadata only — never analysis evidence)
# ---------------------------------------------------------------------------

_CACHE: dict[tuple, tuple[float, Any]] = {}


def clear_discovery_cache() -> None:
    _CACHE.clear()


def _cache_get(key: tuple, ttl: int) -> Optional[Any]:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.monotonic() > expires_at:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_put(key: tuple, value: Any, ttl: int) -> None:
    _CACHE[key] = (time.monotonic() + ttl, value)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _github_sort(sort: str) -> Optional[str]:
    return None if sort == "best-match" else sort


def search_discovery(
    query: str,
    *,
    settings: Settings,
    page: int = 1,
    per_page: int = 10,
    language: Optional[str] = None,
    sort: str = "best-match",
    order: str = "desc",
    transport: Optional[Any] = None,
) -> RepositorySearchResult:
    """Search GitHub repositories; results are cached for 5 minutes."""
    q, page, per_page, sort, order = validate_search_params(
        query, page, per_page, sort, order)
    lang = language.strip() if isinstance(language, str) and language.strip() else None
    ttl = settings.discovery_cache_ttl_seconds or CACHE_TTL_SECONDS
    key = ("search", q, page, per_page, lang, sort, order)
    cached = _cache_get(key, ttl)
    if cached is not None:
        return cached

    github_q = f"{q} language:{lang}" if lang else q
    client = GitHubClient(settings, transport=transport)
    total, items = client.search_repositories(
        github_q, sort=_github_sort(sort), order=order,
        page=page, per_page=per_page,
    )
    repos = [normalize_repo(item) for item in items]
    result = RepositorySearchResult(
        query=q,
        repositories=repos,
        total_count=total,
        page=page,
        per_page=per_page,
        has_more=page * per_page < total,
    )
    _cache_put(key, result, ttl)
    return result


def trend_score(
    stars: int, forks: int, pushed_at: Optional[str],
    max_stars: int, max_forks: int, now: datetime,
) -> float:
    """Deterministic score in [0, 1]; see module docstring for the formula."""
    def _norm(value: int, maximum: int) -> float:
        if maximum <= 0:
            return 0.0
        return math.log10(value + 1) / math.log10(maximum + 1)

    recency = 0.0
    if pushed_at:
        try:
            pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
            days = (now - pushed).total_seconds() / 86400
            recency = max(0.0, 1.0 - max(0.0, days) / 365.0)
        except (ValueError, TypeError):
            recency = 0.0
    return round(
        0.5 * _norm(stars, max_stars)
        + 0.25 * _norm(forks, max_forks)
        + 0.25 * recency, 4)


def get_trending(
    *,
    settings: Settings,
    limit: int = 100,
    language: Optional[str] = None,
    transport: Optional[Any] = None,
    now: Optional[datetime] = None,
    fresh: bool = False,
) -> TrendingResult:
    """Northern Star's API-derived trending ranking (not official GitHub).

    Merges two curated searches — recently created popular repos and
    recently pushed active repos — dedupes by full_name, scores with
    ``trend_score``, and returns the top ``limit``. No cloning, no indexing.
    """
    limit = validate_trending_limit(limit)
    lang = language.strip() if isinstance(language, str) and language.strip() else None
    ttl = settings.discovery_cache_ttl_seconds or CACHE_TTL_SECONDS
    key = ("trending", limit, lang)
    if not fresh:
        cached = _cache_get(key, ttl)
        if cached is not None:
            return cached

    moment = now or datetime.now(timezone.utc)
    from datetime import timedelta
    recent_created = (moment - timedelta(days=180)).strftime("%Y-%m-%d")
    recent_pushed = (moment - timedelta(days=30)).strftime("%Y-%m-%d")
    lang_q = f" language:{lang}" if lang else ""
    queries = [
        f"created:>{recent_created} stars:>500{lang_q}",
        f"pushed:>{recent_pushed} stars:>1000{lang_q}",
    ]

    client = GitHubClient(settings, transport=transport)
    seen: dict[str, dict[str, Any]] = {}
    for github_q in queries:
        _total, items = client.search_repositories(
            github_q, sort="stars", order="desc", page=1, per_page=100)
        for item in items:
            name = item.get("full_name") if isinstance(item, dict) else None
            if isinstance(name, str) and name and name not in seen:
                seen[name] = item

    repos = [normalize_repo(item) for item in seen.values()]
    max_stars = max([r.stars for r in repos] + [0])
    max_forks = max([r.forks for r in repos] + [0])
    scored = [
        (trend_score(r.stars, r.forks, r.pushed_at, max_stars, max_forks, moment), r)
        for r in repos
    ]
    # Deterministic order: score desc, then stars desc, then full_name asc.
    scored.sort(key=lambda t: (-t[0], -t[1].stars, t[1].full_name))
    trending = [
        TrendingRepository(**r.model_dump(), rank=i, trend_score=s, rank_change=None)
        for i, (s, r) in enumerate(scored[:limit], start=1)
    ]
    result = TrendingResult(
        repositories=trending,
        total=len(trending),
        limit=limit,
        generated_at=moment.isoformat(),
    )
    _cache_put(key, result, ttl)
    return result
