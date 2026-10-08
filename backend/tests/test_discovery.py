"""Tests for M8.1 GitHub repository discovery.

Fully offline: the GitHub HTTP layer is replaced with httpx.MockTransport,
so no test ever touches the live GitHub API.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from app.config import get_settings
from app.models.schemas import (
    GitHubRepository,
    RepositorySearchResult,
    TrendingResult,
)
from app.services import discovery as discovery_service
from app.services.discovery import (
    DiscoveryConnectionError,
    DiscoveryRateLimitedError,
    DiscoveryTimeoutError,
    DiscoveryUpstreamError,
    DiscoveryValidationError,
    GitHubClient,
    clear_discovery_cache,
    get_trending,
    normalize_repo,
    search_discovery,
    trend_score,
    validate_search_params,
    validate_trending_limit,
)


def _settings():
    return get_settings()


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_discovery_cache()
    yield
    clear_discovery_cache()


def _item(name="acme/widget", stars=42000, **over):
    owner, repo = name.split("/")
    item = {
        "id": 123,
        "full_name": name,
        "name": repo,
        "owner": {"login": owner},
        "html_url": f"https://github.com/{name}",
        "description": "A widget.",
        "language": "Python",
        "stargazers_count": stars,
        "forks_count": 1000,
        "open_issues_count": 12,
        "watchers_count": stars,
        "topics": ["ai", "agents"],
        "default_branch": "main",
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2026-10-07T00:00:00Z",
        "pushed_at": "2026-10-07T00:00:00Z",
        "license": {"name": "MIT"},
        "archived": False,
        "fork": False,
    }
    item.update(over)
    return item


def _transport(handler):
    return httpx.MockTransport(handler)


def _search_transport(items, total=1234, requests_log=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if requests_log is not None:
            requests_log.append(request)
        return httpx.Response(
            200, json={"total_count": total, "items": items})
    return _transport(handler)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_repository_round_trip(self):
        repo = normalize_repo(_item())
        assert repo.full_name == "acme/widget"
        assert repo.owner == "acme"
        assert repo.stars == 42000
        assert repo.license == "MIT"
        dumped = repo.model_dump(mode="json")
        assert GitHubRepository.model_validate(dumped) == repo
        json.dumps(dumped)
        assert "stargazers_count" not in dumped  # raw fields never leak
        assert "token" not in json.dumps(dumped).lower()

    def test_nullable_fields(self):
        repo = normalize_repo({"id": 1, "full_name": "a/b", "name": "b"})
        assert repo.description is None
        assert repo.language is None
        assert repo.license is None
        assert repo.topics == []
        assert repo.owner == "a"

    def test_search_result_shape(self):
        result = RepositorySearchResult(
            query="local ai", repositories=[], total_count=0,
            page=1, per_page=10, has_more=False)
        assert result.query == "local ai"
        json.dumps(result.model_dump(mode="json"))

    def test_trending_rank_change_defaults_none(self):
        repo = normalize_repo(_item())
        from app.models.schemas import TrendingRepository
        t = TrendingRepository(**repo.model_dump(), rank=1, trend_score=0.9)
        assert t.rank_change is None


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_queries_rejected(self):
        for bad in ("", "   ", "\t\n"):
            with pytest.raises(DiscoveryValidationError):
                validate_search_params(bad)

    def test_long_query_rejected(self):
        with pytest.raises(DiscoveryValidationError):
            validate_search_params("x" * 257)

    def test_query_preserved(self):
        q, _, _, _, _ = validate_search_params("  local AI  ")
        assert q == "local AI"

    def test_bad_page(self):
        for bad in (0, -1):
            with pytest.raises(DiscoveryValidationError):
                validate_search_params("ai", page=bad)

    def test_bad_per_page(self):
        for bad in (0, 31, 100):
            with pytest.raises(DiscoveryValidationError):
                validate_search_params("ai", per_page=bad)

    def test_bad_sort_order(self):
        with pytest.raises(DiscoveryValidationError):
            validate_search_params("ai", sort="nope")
        with pytest.raises(DiscoveryValidationError):
            validate_search_params("ai", order="sideways")

    def test_trending_limit(self):
        assert validate_trending_limit(100) == 100
        for bad in (0, 101, -5):
            with pytest.raises(DiscoveryValidationError):
                validate_trending_limit(bad)


# ---------------------------------------------------------------------------
# GitHub client tests
# ---------------------------------------------------------------------------

class TestClient:
    def test_success_and_auth_header(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            seen["url"] = str(request.url)
            return httpx.Response(
                200, json={"total_count": 1, "items": [_item()]})

        client = GitHubClient(_settings(), transport=_transport(handler))
        total, items = client.search_repositories(
            "ai", sort="stars", order="desc", page=1, per_page=10)
        assert total == 1 and len(items) == 1
        assert "sort=stars" in seen["url"]
        # No token configured in test env → no auth header; header shape
        # verified in the token test below.
        assert seen["auth"] is None

    def test_bearer_token_header(self, monkeypatch):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            return httpx.Response(200, json={"total_count": 0, "items": []})

        monkeypatch.setenv("GITHUB_TOKEN", "secret-token-123")
        client = GitHubClient(_settings(), transport=_transport(handler))
        client.search_repositories("ai")
        assert seen["auth"] == "Bearer secret-token-123"

    def test_no_token_no_header(self, monkeypatch):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            return httpx.Response(200, json={"total_count": 0, "items": []})

        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        client = GitHubClient(_settings(), transport=_transport(handler))
        client.search_repositories("ai")
        assert seen["auth"] is None

    def test_403_rate_limit(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403, headers={"X-RateLimit-Remaining": "0",
                              "X-RateLimit-Reset": "1790000000"},
                json={"message": "API rate limit exceeded"})
        client = GitHubClient(_settings(), transport=_transport(handler))
        with pytest.raises(DiscoveryRateLimitedError):
            client.search_repositories("ai")

    def test_429(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"message": "too many"})
        client = GitHubClient(_settings(), transport=_transport(handler))
        with pytest.raises(DiscoveryRateLimitedError):
            client.search_repositories("ai")

    def test_5xx_retries_once_then_raises(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(503, json={})
        client = GitHubClient(_settings(), transport=_transport(handler))
        with pytest.raises(DiscoveryUpstreamError):
            client.search_repositories("ai")
        assert len(calls) == 2  # bounded: initial + exactly one retry

    def test_5xx_recovers_on_retry(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(502, json={})
            return httpx.Response(200, json={"total_count": 0, "items": []})
        client = GitHubClient(_settings(), transport=_transport(handler))
        total, items = client.search_repositories("ai")
        assert (total, items) == (0, [])

    def test_timeout(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("slow")
        client = GitHubClient(_settings(), transport=_transport(handler))
        with pytest.raises(DiscoveryTimeoutError):
            client.search_repositories("ai")

    def test_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns")
        client = GitHubClient(_settings(), transport=_transport(handler))
        with pytest.raises(DiscoveryConnectionError):
            client.search_repositories("ai")

    def test_malformed_payload(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"items": "nope"})
        client = GitHubClient(_settings(), transport=_transport(handler))
        with pytest.raises(DiscoveryUpstreamError):
            client.search_repositories("ai")


# ---------------------------------------------------------------------------
# Search service tests
# ---------------------------------------------------------------------------

class TestSearch:
    def test_params_and_normalization(self):
        log = []
        transport = _search_transport([_item()], requests_log=log)
        result = search_discovery(
            "local ai", settings=_settings(), page=1, per_page=10,
            transport=transport)
        assert result.query == "local ai"
        assert result.total_count == 1234
        assert result.has_more is True
        assert result.repositories[0].full_name == "acme/widget"
        url = str(log[0].url)
        assert "q=local" in url and "page=1" in url

    def test_has_more_false_on_last_page(self):
        transport = _search_transport([_item()], total=10)
        result = search_discovery(
            "ai", settings=_settings(), page=1, per_page=10, transport=transport)
        assert result.has_more is False

    def test_language_filter_in_query(self):
        log = []
        transport = _search_transport([], total=0, requests_log=log)
        search_discovery("ai", settings=_settings(), language="Python",
                         transport=transport)
        assert "language" in str(log[0].url)

    def test_best_match_omits_sort(self):
        log = []
        transport = _search_transport([], total=0, requests_log=log)
        search_discovery("ai", settings=_settings(), sort="best-match",
                         transport=transport)
        assert "sort=" not in str(log[0].url)

    def test_caching_avoids_second_call(self):
        calls = []
        transport = _search_transport([_item()], requests_log=calls)
        first = search_discovery("ai", settings=_settings(), transport=transport)
        second = search_discovery("ai", settings=_settings(), transport=transport)
        assert len(calls) == 1
        assert first is second

    def test_cache_key_uniqueness(self):
        calls = []
        transport = _search_transport([_item()], requests_log=calls)
        search_discovery("ai", settings=_settings(), page=1, transport=transport)
        search_discovery("ai", settings=_settings(), page=2, transport=transport)
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Trending tests
# ---------------------------------------------------------------------------

class TestTrending:
    NOW = datetime(2026, 10, 8, tzinfo=timezone.utc)

    def _trending_transport(self, items_lists):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            idx = min(len(calls) - 1, len(items_lists) - 1)
            items = items_lists[idx]
            return httpx.Response(
                200, json={"total_count": len(items), "items": items})
        return _transport(handler), calls

    def test_deterministic_ranking(self):
        items = [
            _item("big/popular", stars=90000, forks=5000,
                  pushed_at="2026-10-07T00:00:00Z"),
            _item("small/fresh", stars=2000, forks=100,
                  pushed_at="2026-10-07T00:00:00Z"),
        ]
        transport, _ = self._trending_transport([items, []])
        first = get_trending(settings=_settings(), limit=10,
                             transport=transport, now=self.NOW)
        clear_discovery_cache()
        transport2, _ = self._trending_transport([items, []])
        second = get_trending(settings=_settings(), limit=10,
                              transport=transport2, now=self.NOW)
        assert [r.full_name for r in first.repositories] == \
               [r.full_name for r in second.repositories]
        assert first.repositories[0].full_name == "big/popular"
        assert first.repositories[0].rank == 1

    def test_limit_and_no_duplicates(self):
        items = [_item(f"o/r{i}", stars=1000 + i) for i in range(5)]
        transport, _ = self._trending_transport([items, items[:2]])
        result = get_trending(settings=_settings(), limit=3,
                              transport=transport, now=self.NOW)
        assert len(result.repositories) == 3
        names = [r.full_name for r in result.repositories]
        assert len(set(names)) == 3

    def test_max_100(self):
        items = [_item(f"o/r{i}", stars=1000 + i) for i in range(3)]
        transport, _ = self._trending_transport([items, []])
        result = get_trending(settings=_settings(), limit=100,
                              transport=transport, now=self.NOW)
        assert result.total <= 100

    def test_no_fake_rank_change(self):
        transport, _ = self._trending_transport(
            [[_item("a/b", stars=5000)], []])
        result = get_trending(settings=_settings(), limit=10,
                              transport=transport, now=self.NOW)
        assert all(r.rank_change is None for r in result.repositories)

    def test_score_formula(self):
        # Max repo scores 1.0 when freshly pushed.
        s = trend_score(90000, 5000, "2026-10-07T00:00:00Z",
                        90000, 5000, self.NOW)
        assert s == pytest.approx(1.0, abs=0.01)
        # Stale push loses the recency quarter.
        stale = trend_score(90000, 5000, "2020-01-01T00:00:00Z",
                            90000, 5000, self.NOW)
        assert stale == pytest.approx(0.75, abs=0.01)
        assert 0.0 <= trend_score(0, 0, None, 0, 0, self.NOW) <= 1.0


# ---------------------------------------------------------------------------
# Security tests
# ---------------------------------------------------------------------------

class TestSecurity:
    def test_token_never_in_output(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "super-secret-xyz")
        transport = _search_transport([_item()])
        result = search_discovery("ai", settings=_settings(), transport=transport)
        assert "super-secret-xyz" not in result.model_dump_json()

    def test_token_never_in_errors(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "super-secret-xyz")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, headers={"X-RateLimit-Remaining": "0"},
                                  json={"message": "rate limit"})
        transport = _transport(handler)
        try:
            search_discovery("ai", settings=_settings(), transport=transport)
            raise AssertionError("expected rate-limit error")
        except DiscoveryRateLimitedError as exc:
            assert "super-secret-xyz" not in str(exc)


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

class TestApi:
    def test_search_200(self, monkeypatch):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.models.schemas import RepositorySearchResult

        result = RepositorySearchResult(
            query="ai", repositories=[normalize_repo(_item())],
            total_count=1, page=1, per_page=10, has_more=False)
        monkeypatch.setattr(
            "app.services.discovery.search_discovery", lambda *a, **k: result)
        client = TestClient(app)
        resp = client.get("/api/v1/discover/search?q=ai")
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "ai"
        assert body["repositories"][0]["full_name"] == "acme/widget"

    def test_search_empty_query_422(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        assert client.get("/api/v1/discover/search?q=%20%20").status_code == 422

    def test_search_bad_sort_422(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        assert client.get("/api/v1/discover/search?q=ai&sort=nope").status_code == 422

    def test_search_bad_per_page_422(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        assert client.get("/api/v1/discover/search?q=ai&per_page=99").status_code == 422

    def test_rate_limit_maps_429(self, monkeypatch):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.services.discovery import DiscoveryRateLimitedError

        def boom(*a, **k):
            raise DiscoveryRateLimitedError("GitHub rate limit exceeded.")
        monkeypatch.setattr("app.services.discovery.search_discovery", boom)
        client = TestClient(app)
        resp = client.get("/api/v1/discover/search?q=ai")
        assert resp.status_code == 429

    def test_trending_200(self, monkeypatch):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.models.schemas import TrendingRepository, TrendingResult

        repo = TrendingRepository(**normalize_repo(_item()).model_dump(),
                                  rank=1, trend_score=0.95)
        result = TrendingResult(repositories=[repo], total=1, limit=10,
                                generated_at="2026-10-08T00:00:00+00:00")
        monkeypatch.setattr(
            "app.services.discovery.get_trending", lambda *a, **k: result)
        client = TestClient(app)
        resp = client.get("/api/v1/discover/trending?limit=10")
        assert resp.status_code == 200
        assert resp.json()["repositories"][0]["rank"] == 1

    def test_trending_bad_limit_422(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        assert client.get("/api/v1/discover/trending?limit=500").status_code == 422


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCli:
    def test_discover_parser(self):
        from app.cli import build_parser
        args = build_parser().parse_args(["discover", "AI coding agents"])
        assert args.query == "AI coding agents"

    def test_trending_parser(self):
        from app.cli import build_parser
        args = build_parser().parse_args(["trending", "--limit", "20"])
        assert args.limit == 20

    def test_discover_json(self, monkeypatch, capsys):
        from app.cli import main
        result = RepositorySearchResult(
            query="ai", repositories=[normalize_repo(_item())],
            total_count=1, page=1, per_page=10, has_more=False)
        monkeypatch.setattr(
            "app.services.discovery.search_discovery", lambda *a, **k: result)
        assert main(["discover", "ai", "--json"]) == 0
        body = json.loads(capsys.readouterr().out)
        assert body["repositories"][0]["stars"] == 42000

    def test_discover_human(self, monkeypatch, capsys):
        from app.cli import main
        result = RepositorySearchResult(
            query="ai", repositories=[normalize_repo(_item())],
            total_count=1, page=1, per_page=10, has_more=False)
        monkeypatch.setattr(
            "app.services.discovery.search_discovery", lambda *a, **k: result)
        assert main(["discover", "ai"]) == 0
        out = capsys.readouterr().out
        assert "GitHub Discovery" in out and "acme/widget" in out

    def test_trending_json(self, monkeypatch, capsys):
        from app.cli import main
        from app.models.schemas import TrendingRepository, TrendingResult
        repo = TrendingRepository(**normalize_repo(_item()).model_dump(),
                                  rank=1, trend_score=0.9)
        result = TrendingResult(repositories=[repo], total=1, limit=20,
                                generated_at="2026-10-08T00:00:00+00:00")
        monkeypatch.setattr(
            "app.services.discovery.get_trending", lambda *a, **k: result)
        assert main(["trending", "--limit", "20", "--json"]) == 0
        body = json.loads(capsys.readouterr().out)
        assert body["repositories"][0]["trend_score"] == 0.9
