"""Tests for M8.2 historical trend intelligence.

Fully offline: snapshots are seeded directly into the test SQLite database
at deterministic timestamps; GitHub capture paths use httpx.MockTransport.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.config import get_settings
from app.services import trends as trends_service
from app.services.trends import (
    TrendValidationError,
    capture_trending_snapshot,
    compare_window,
    emerging_scores,
    get_repository_history,
    list_snapshot_times,
    store_snapshot_rows,
    validate_limit,
    validate_window,
)


def _settings():
    return get_settings()


@pytest.fixture(autouse=True)
def _isolated_db(no_default_storage):
    # no_default_storage already redirects storage; nothing else needed.
    yield


def _row(full_name, stars=1000, forks=100, rank=1, pushed_at=None,
         trend_score=0.5, language="Python"):
    return {
        "full_name": full_name, "rank": rank, "stars": stars, "forks": forks,
        "open_issues": 5, "watchers": stars, "pushed_at": pushed_at,
        "trend_score": trend_score, "language": language,
        "topics": ["ai"], "html_url": f"https://github.com/{full_name}",
    }


def _seed(settings, entries: dict[str, list[tuple]]):
    """entries: full_name → list of (snapshot_at, stars, forks, rank)."""
    for full_name, points in entries.items():
        for snap_at, stars, forks, rank in points:
            store_snapshot_rows(settings, snap_at, [
                _row(full_name, stars=stars, forks=forks, rank=rank,
                     pushed_at="2026-10-07T00:00:00Z")])

T0 = "2026-10-01T00:00:00+00:00"
T1 = "2026-10-08T00:00:00+00:00"  # exactly 7d after T0


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_schema_and_indexes(self):
        from app.services.indexing import connect_evidence_db, evidence_db_path
        settings = _settings()
        conn = connect_evidence_db(
            evidence_db_path(settings.storage_root, settings.db_filename))
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "discovery_snapshots" in tables
            # M1-M8.1 tables untouched.
            assert {"repositories", "files", "chunks"} <= tables
            indexes = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
            assert {"idx_snapshots_full_name", "idx_snapshots_snapshot_at",
                    "idx_snapshots_repo_time"} <= indexes
        finally:
            conn.close()

    def test_duplicate_protection(self):
        settings = _settings()
        new, skipped = store_snapshot_rows(settings, T0, [_row("a/b")])
        assert (new, skipped) == (1, 0)
        new, skipped = store_snapshot_rows(settings, T0, [_row("a/b")])
        assert (new, skipped) == (0, 1)
        assert list_snapshot_times(settings) == [T0]

    def test_persistence(self):
        settings = _settings()
        store_snapshot_rows(settings, T0, [_row("a/b", stars=42)])
        rows = trends_service._rows_for_repo(settings, "a/b")
        assert rows[0]["stars"] == 42


# ---------------------------------------------------------------------------
# Snapshot capture tests
# ---------------------------------------------------------------------------

def _trending_transport(items):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"total_count": len(items), "items": items})
    return httpx.MockTransport(handler)


def _gh_item(name, stars=1000):
    owner, repo = name.split("/")
    return {
        "id": 1, "full_name": name, "name": repo, "owner": {"login": owner},
        "html_url": f"https://github.com/{name}", "description": "d",
        "language": "Python", "stargazers_count": stars, "forks_count": 50,
        "open_issues_count": 1, "watchers_count": stars, "topics": [],
        "default_branch": "main", "created_at": "2020-01-01T00:00:00Z",
        "updated_at": T1, "pushed_at": T1,
        "license": {"name": "MIT"}, "archived": False, "fork": False,
    }


class TestCapture:
    def test_capture_success(self):
        settings = _settings()
        transport = _trending_transport([_gh_item("a/b"), _gh_item("c/d")])
        result = capture_trending_snapshot(
            settings=settings, limit=10, transport=transport)
        assert result.repositories_captured == 2
        assert result.new_rows == 2
        assert result.skipped_duplicates == 0
        assert result.snapshot_at

    def test_capture_default_limit_is_100(self):
        import inspect
        assert inspect.signature(capture_trending_snapshot).parameters[
            "limit"].default == 100

    def test_capture_bad_limit(self):
        from app.services.discovery import DiscoveryValidationError
        with pytest.raises(DiscoveryValidationError):
            capture_trending_snapshot(settings=_settings(), limit=500,
                                      transport=_trending_transport([]))

    def test_capture_github_failure_propagates(self):
        from app.services.discovery import DiscoveryUpstreamError

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={})
        with pytest.raises(DiscoveryUpstreamError):
            capture_trending_snapshot(
                settings=_settings(), limit=10,
                transport=httpx.MockTransport(handler))

    def test_malformed_items_skipped(self):
        settings = _settings()
        transport = _trending_transport([_gh_item("a/b"), "nonsense", None])
        result = capture_trending_snapshot(
            settings=settings, limit=10, transport=transport)
        assert result.repositories_captured == 1
        assert result.new_rows == 1


# ---------------------------------------------------------------------------
# History / window comparison tests
# ---------------------------------------------------------------------------

class TestHistory:
    def test_no_history(self):
        result = compare_window(settings=_settings(), window="7d")
        assert result.has_history is False
        assert result.repositories == []
        assert result.history_reason

    def test_one_snapshot_no_comparison(self):
        settings = _settings()
        _seed(settings, {"a/b": [(T1, 1000, 100, 1)]})
        result = compare_window(settings=settings, window="7d")
        assert result.has_history is False
        assert len(result.repositories) == 1
        assert result.repositories[0].history_available is False
        assert result.repositories[0].star_delta is None

    def test_two_snapshots_7d(self):
        settings = _settings()
        _seed(settings, {"a/b": [(T0, 1000, 100, 2), (T1, 1500, 120, 1)]})
        result = compare_window(settings=settings, window="7d")
        assert result.has_history is True
        assert result.previous_snapshot_at == T0
        r = result.repositories[0]
        assert r.history_available is True
        assert r.star_delta == 500
        assert r.star_growth_percent == pytest.approx(50.0)
        assert r.fork_delta == 20
        assert r.rank_change == 1  # 2 → 1 moved UP
        assert r.emerging_score is not None

    def test_rank_decline_negative(self):
        settings = _settings()
        _seed(settings, {"a/b": [(T0, 1000, 100, 1), (T1, 900, 90, 5)]})
        result = compare_window(settings=settings, window="7d")
        r = result.repositories[0]
        assert r.rank_change == -4
        assert r.star_delta == -100

    def test_closest_snapshot_selection(self):
        settings = _settings()
        # T0 exact 7d; an extra snapshot 1h off target must lose to T0.
        _seed(settings, {"a/b": [
            (T0, 1000, 100, 1),
            ("2026-10-01T01:00:00+00:00", 1100, 110, 1),
            (T1, 1500, 120, 1),
        ]})
        result = compare_window(settings=settings, window="7d")
        assert result.previous_snapshot_at == T0

    def test_insufficient_distance(self):
        settings = _settings()
        # Only 1 day apart; 7d window (±33.6h tolerance) has no match.
        _seed(settings, {"a/b": [
            ("2026-10-07T00:00:00+00:00", 1000, 100, 1), (T1, 1100, 110, 1)]})
        result = compare_window(settings=settings, window="7d")
        assert result.has_history is False
        assert result.previous_snapshot_at is None

    def test_24h_and_30d_windows(self):
        settings = _settings()
        _seed(settings, {"a/b": [
            ("2026-10-07T00:00:00+00:00", 1000, 100, 1), (T1, 1100, 110, 1)]})
        assert compare_window(settings=settings, window="24h").has_history is True
        assert compare_window(settings=settings, window="30d").has_history is False

    def test_zero_previous_stars(self):
        settings = _settings()
        _seed(settings, {"a/b": [(T0, 0, 0, 1), (T1, 50, 5, 1)]})
        r = compare_window(settings=settings, window="7d").repositories[0]
        assert r.history_available is True
        assert r.star_delta == 50
        assert r.star_growth_percent is None
        assert r.fork_growth_percent is None

    def test_missing_pushed_at(self):
        settings = _settings()
        store_snapshot_rows(settings, T0, [_row("a/b", pushed_at=None)])
        store_snapshot_rows(settings, T1, [_row("a/b", stars=1100, pushed_at=None)])
        r = compare_window(settings=settings, window="7d").repositories[0]
        assert r.history_available is True  # no crash; activity contributes 0
        assert r.emerging_score is not None

    def test_appearing_repo_has_no_history(self):
        settings = _settings()
        _seed(settings, {
            "a/b": [(T0, 1000, 100, 1), (T1, 1100, 110, 1)],
            "c/d": [(T1, 500, 50, 2)],
        })
        result = compare_window(settings=settings, window="7d")
        by_name = {r.full_name: r for r in result.repositories}
        assert by_name["c/d"].history_available is False
        assert by_name["c/d"].emerging_score is None

    def test_invalid_window_and_limit(self):
        with pytest.raises(TrendValidationError):
            validate_window("90d")
        with pytest.raises(TrendValidationError):
            validate_limit(0)
        with pytest.raises(TrendValidationError):
            validate_limit(101)
        assert validate_window("24h") == "24h"


# ---------------------------------------------------------------------------
# Emerging score tests
# ---------------------------------------------------------------------------

class TestEmerging:
    NOW = datetime(2026, 10, 8, tzinfo=timezone.utc)

    def test_growth_beats_size(self):
        curr_small = {"full_name": "new/hot", "stars": 2000, "forks": 200,
                      "rank": 2, "pushed_at": "2026-10-07T00:00:00Z"}
        prev_small = {"full_name": "new/hot", "stars": 1000, "forks": 100,
                      "rank": 10, "pushed_at": "2026-09-30T00:00:00Z"}
        curr_big = {"full_name": "old/giant", "stars": 505000, "forks": 50000,
                    "rank": 1, "pushed_at": "2026-10-07T00:00:00Z"}
        prev_big = {"full_name": "old/giant", "stars": 500000, "forks": 49900,
                    "rank": 1, "pushed_at": "2026-09-30T00:00:00Z"}
        scores = emerging_scores(
            [(curr_small, prev_small), (curr_big, prev_big)], self.NOW)
        assert scores["new/hot"] > scores["old/giant"]

    def test_no_history_no_score(self):
        curr = {"full_name": "x/y", "stars": 10, "forks": 1, "rank": 1,
                "pushed_at": None}
        assert emerging_scores([(curr, None)], self.NOW) == {"x/y": None}

    def test_bounded(self):
        curr = {"full_name": "x/y", "stars": 10**7, "forks": 10**6, "rank": 1,
                "pushed_at": "2026-10-07T00:00:00Z"}
        prev = {"full_name": "x/y", "stars": 1, "forks": 1, "rank": 100,
                "pushed_at": None}
        score = emerging_scores([(curr, prev)], self.NOW)["x/y"]
        assert 0.0 <= score <= 1.0

    def test_deterministic_tie_break_in_compare(self):
        settings = _settings()
        _seed(settings, {
            "a/b": [(T0, 1000, 100, 1), (T1, 1500, 150, 1)],
            "c/d": [(T0, 1000, 100, 2), (T1, 1500, 150, 2)],
        })
        first = [r.full_name for r in compare_window(
            settings=settings, window="7d").repositories]
        second = [r.full_name for r in compare_window(
            settings=settings, window="7d").repositories]
        assert first == second == ["a/b", "c/d"]

    def test_popularity_vs_emergence_ordering(self):
        settings = _settings()
        _seed(settings, {
            "old/giant": [(T0, 500000, 50000, 1), (T1, 505000, 50100, 1)],
            "new/hot": [(T0, 1000, 100, 5), (T1, 3000, 400, 2)],
        })
        repos = compare_window(settings=settings, window="7d").repositories
        assert repos[0].full_name == "new/hot"  # growth-heavy ranking
        assert repos[0].emerging_score > repos[1].emerging_score


# ---------------------------------------------------------------------------
# Repository history endpoint logic
# ---------------------------------------------------------------------------

class TestRepoHistory:
    def test_unknown_repo_empty(self):
        result = get_repository_history(
            settings=_settings(), owner="acme", repo="ghost", window="30d")
        assert result.snapshots == []
        assert result.has_history is False
        assert result.comparison is None

    def test_history_points_and_comparison(self):
        settings = _settings()
        _seed(settings, {"a/b": [(T0, 1000, 100, 2), (T1, 1500, 120, 1)]})
        result = get_repository_history(
            settings=settings, owner="a", repo="b", window="7d")
        assert result.total_snapshots == 2
        assert result.has_history is True
        assert result.comparison is not None
        assert result.comparison.star_delta == 500
        assert result.comparison.rank_change == 1
        # Only actual stored snapshots are returned.
        assert [s.snapshot_at for s in result.snapshots] == [T0, T1]

    def test_single_snapshot_no_comparison(self):
        settings = _settings()
        _seed(settings, {"a/b": [(T1, 1000, 100, 1)]})
        result = get_repository_history(
            settings=settings, owner="a", repo="b", window="30d")
        assert result.total_snapshots == 1
        assert result.has_history is False

    def test_bad_slug(self):
        with pytest.raises(TrendValidationError):
            get_repository_history(
                settings=_settings(), owner="..", repo="x", window="7d")


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

class TestApi:
    def test_trends_no_history(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/api/v1/discover/trends?window=7d&limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_history"] is False
        assert body["window"] == "7d"

    def test_trends_invalid_window_422(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        assert client.get("/api/v1/discover/trends?window=90d").status_code == 422

    def test_trends_invalid_limit_422(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        assert client.get("/api/v1/discover/trends?limit=500").status_code == 422

    def test_history_unknown_repo_200_empty(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/api/v1/discover/repositories/acme/ghost/history")
        assert resp.status_code == 200
        assert resp.json()["has_history"] is False

    def test_snapshot_endpoint(self, monkeypatch):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.models.schemas import TrendSnapshotResult
        result = TrendSnapshotResult(
            snapshot_at=T1, limit=10, repositories_captured=2,
            new_rows=2, skipped_duplicates=0)
        monkeypatch.setattr(
            "app.services.trends.capture_trending_snapshot",
            lambda *a, **k: result)
        client = TestClient(app)
        resp = client.post("/api/v1/discover/snapshots?limit=10")
        assert resp.status_code == 200
        assert resp.json()["new_rows"] == 2

    def test_snapshot_upstream_failure_502(self, monkeypatch):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.services.discovery import DiscoveryUpstreamError

        def boom(*a, **k):
            raise DiscoveryUpstreamError("GitHub is temporarily unavailable.")
        monkeypatch.setattr(
            "app.services.trends.capture_trending_snapshot", boom)
        client = TestClient(app)
        assert client.post("/api/v1/discover/snapshots").status_code == 502

    def test_snapshot_rate_limit_429(self, monkeypatch):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.services.discovery import DiscoveryRateLimitedError

        def boom(*a, **k):
            raise DiscoveryRateLimitedError("GitHub rate limit exceeded.")
        monkeypatch.setattr(
            "app.services.trends.capture_trending_snapshot", boom)
        client = TestClient(app)
        assert client.post("/api/v1/discover/snapshots").status_code == 429


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCli:
    def test_parsers(self):
        from app.cli import build_parser
        parser = build_parser()
        assert parser.parse_args(["snapshot-trending"]).limit == 100
        assert parser.parse_args(["trends", "--window", "24h"]).window == "24h"
        args = parser.parse_args(["history", "acme/widget"])
        assert args.repo == "acme/widget"

    def test_snapshot_json(self, monkeypatch, capsys):
        from app.cli import main
        from app.models.schemas import TrendSnapshotResult
        result = TrendSnapshotResult(
            snapshot_at=T1, limit=10, repositories_captured=1,
            new_rows=1, skipped_duplicates=0)
        monkeypatch.setattr(
            "app.services.trends.capture_trending_snapshot",
            lambda *a, **k: result)
        assert main(["snapshot-trending", "--limit", "10", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["new_rows"] == 1

    def test_trends_json(self, monkeypatch, capsys):
        from app.cli import main
        settings = _settings()
        _seed(settings, {"a/b": [(T0, 1000, 100, 2), (T1, 1500, 120, 1)]})
        assert main(["trends", "--window", "7d", "--json"]) == 0
        body = json.loads(capsys.readouterr().out)
        assert body["has_history"] is True
        assert body["repositories"][0]["rank_change"] == 1

    def test_history_json(self, monkeypatch, capsys):
        from app.cli import main
        settings = _settings()
        _seed(settings, {"a/b": [(T0, 1000, 100, 2), (T1, 1500, 120, 1)]})
        assert main(["history", "a/b", "--window", "30d", "--json"]) == 0
        body = json.loads(capsys.readouterr().out)
        assert body["total_snapshots"] == 2

    def test_history_no_data(self, capsys):
        from app.cli import main
        assert main(["history", "acme/ghost"]) == 0
        assert "No snapshots" in capsys.readouterr().out

    def test_trends_invalid_window_exits_1(self):
        from app.cli import main
        # argparse choices reject before the handler runs (exit code 2).
        assert main(["trends", "--window", "90d"]) == 2
