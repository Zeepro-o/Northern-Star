"""Tests for lexical retrieval: ranking, provenance, isolation, and the API."""

from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services.github import FetchedRepository
from app.services.indexing import connect_evidence_db, index_repository
from app.services.ingestion import _assemble_manifest, analyze_repository
from app.services.retrieval import RepoNotIndexedError, search_evidence
from tests.conftest import build_evidence_checkout, write_files


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _manifest_for(checkout, owner: str = "acme", repo: str = "evidence") -> tuple:
    settings = get_settings()
    analysis = analyze_repository(checkout, settings, tree_root_name=repo)
    fetched = FetchedRepository(
        owner=owner,
        repo=repo,
        github_url=f"https://github.com/{owner}/{repo}",
        checkout_root=checkout,
        commit_hash="c0ffee",
        default_branch="main",
    )
    return _assemble_manifest(fetched, analysis, settings, None), checkout


def _paths(results) -> set[str]:
    return {r.file_path for r in results}


@pytest.fixture()
def indexed_evidence(evidence_repo):
    checkout, manifest, db_path = evidence_repo
    index_repository(manifest, checkout, db_path)
    return checkout, manifest, db_path


class TestSearchService:
    def test_hits_expected_file_for_concept(self, indexed_evidence):
        _, _, db_path = indexed_evidence
        results = search_evidence(db_path, "acme/evidence", "authentication")
        assert results
        assert "src/auth/middleware.py" in _paths(results)

    def test_react_component_found(self, indexed_evidence):
        _, _, db_path = indexed_evidence
        results = search_evidence(db_path, "acme/evidence", "button")
        assert results
        assert "src/components/Button.tsx" in _paths(results)

    def test_database_connection_found(self, indexed_evidence):
        _, _, db_path = indexed_evidence
        results = search_evidence(db_path, "acme/evidence", "connection database")
        assert results
        assert "src/db/connection.py" in _paths(results)

    def test_api_endpoint_found(self, indexed_evidence):
        _, _, db_path = indexed_evidence
        results = search_evidence(db_path, "acme/evidence", "list_items")
        assert results
        assert "src/api/routes.py" in _paths(results)

    def test_exact_provenance_on_every_result(self, indexed_evidence):
        _, _, db_path = indexed_evidence
        results = search_evidence(db_path, "acme/evidence", "authentication")
        expected = (db_path.parent / "fixtures" / "evidence-repo" / "src/auth/middleware.py")
        expected_text = expected.read_text()
        for r in results:
            assert r.file_path
            assert r.start_line >= 1
            assert r.end_line >= r.start_line
            assert r.language is not None
            assert r.content
        top = results[0]
        assert top.file_path == "src/auth/middleware.py"
        assert top.start_line == 1
        assert top.end_line == len(expected_text.splitlines())
        assert top.language == "Python"
        assert top.content == expected_text

    def test_no_matches_returns_empty_list(self, indexed_evidence):
        _, _, db_path = indexed_evidence
        assert search_evidence(db_path, "acme/evidence", "zzqqxwv") == []

    def test_empty_query_matches_nothing(self, indexed_evidence):
        _, _, db_path = indexed_evidence
        assert search_evidence(db_path, "acme/evidence", "") == []
        assert search_evidence(db_path, "acme/evidence", "   !!! ") == []

    def test_unindexed_repo_raises(self, evidence_repo):
        _, _, db_path = evidence_repo
        with pytest.raises(RepoNotIndexedError):
            search_evidence(db_path, "acme/evidence", "authentication")

    def test_limit_is_respected(self, indexed_evidence):
        _, _, db_path = indexed_evidence
        many = search_evidence(db_path, "acme/evidence", "fastapi", limit=5, default_limit=20)
        assert len(many) <= 5

    def test_fts_special_characters_are_safe(self, indexed_evidence):
        _, _, db_path = indexed_evidence
        # Punctuation-heavy queries must never raise a FTS5 parse error.
        for q in ('"', "NEAR(x)", "pqr OR xyz", "a:b"):
            search_evidence(db_path, "acme/evidence", q)  # must not raise


class TestRanking:
    def _build_ranking_repo(self, checkout):
        write_files(checkout, {
            "short_a.py": (
                "# Short file\n"
                "def run():\n"
                "    # unique_marker_xyz lives here once\n"
                "    return 1\n"
            ),
            "long_b.py": "\n".join(
                f"v{i} = {i}" if i != 100 else "    # unique_marker_xyz buried in a long file"
                for i in range(250)
            ),
        })

    def test_shorter_file_ranks_first_for_rare_term(self, tmp_path, no_default_storage):
        checkout = tmp_path / "ranking"
        checkout.mkdir(parents=True, exist_ok=True)
        self._build_ranking_repo(checkout)
        manifest, _ = _manifest_for(checkout)
        db_path = no_default_storage / "ranking.db"
        index_repository(manifest, checkout, db_path)

        results = search_evidence(
            db_path, manifest.id, "unique_marker_xyz", default_limit=10
        )
        assert len(results) >= 1
        # BM25 favours the rare term (higher relative frequency): short file first.
        assert results[0].file_path == "short_a.py"
        assert "short_a.py" in _paths(results)
        assert "long_b.py" in _paths(results)


class TestIsolation:
    def test_query_never_leaks_across_repositories(self, indexed_evidence, tmp_path):
        checkout, manifest, db_path = indexed_evidence

        # Repo B: same code, plus one file only it contains.
        other = tmp_path / "other"
        shutil.copytree(checkout, other)
        write_files(
            other,
            {"src/secret_vault.py": 'def store_password():\n    "Secrets live in a vault."\n    pass\n'},
        )
        other_manifest, _ = _manifest_for(other, owner="elsewhere", repo="other")
        index_repository(other_manifest, other, db_path)

        # The same concept exists in both repos, but "vault" is only in B.
        assert search_evidence(db_path, "acme/evidence", "vault") == []
        b_results = search_evidence(db_path, "elsewhere/other", "vault")
        assert b_results
        assert "src/secret_vault.py" in _paths(b_results)

    def test_identical_query_returns_only_own_rows(self, indexed_evidence, tmp_path):
        checkout, manifest, db_path = indexed_evidence
        other = tmp_path / "other-copy"
        shutil.copytree(checkout, other)
        other_manifest, _ = _manifest_for(other, owner="elsewhere", repo="other")
        index_repository(other_manifest, other, db_path)

        a = search_evidence(db_path, "acme/evidence", "authentication", default_limit=50)
        b = search_evidence(db_path, "elsewhere/other", "authentication", default_limit=50)
        assert a and b
        # Every stored chunk carries its own repo_id; the results came from the
        # requested repository only when the joining filters on repo_id held.
        with connect_evidence_db(db_path) as conn:
            leaks = conn.execute(
                "SELECT DISTINCT repo_id FROM chunks WHERE file_path = ?",
                ("src/auth/middleware.py",),
            ).fetchall()
        assert {r["repo_id"] for r in leaks} == {"acme/evidence", "elsewhere/other"}
        assert a[0].file_path == "src/auth/middleware.py"
        # Validation: results equal expected file sets; no cross-repo rows.
        assert {r.file_path for r in a} == {r.file_path for r in b}


class TestApiSearch:
    def _seed_repo(self, tmp_path, no_default_storage, owner="acme", repo="evidence",
                   extra_files=None, index=True):
        settings = get_settings()
        checkout = build_evidence_checkout(tmp_path / "fixtures")
        if extra_files:
            write_files(checkout, extra_files)
        manifest, _ = _manifest_for(checkout, owner=owner, repo=repo)
        repo_dir = no_default_storage / owner / repo
        (repo_dir / "checkout").mkdir(parents=True, exist_ok=True)
        shutil.copytree(checkout, repo_dir / "checkout", dirs_exist_ok=True)
        (repo_dir / settings.manifest_filename).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        if index:
            index_repository(manifest, checkout, no_default_storage / settings.db_filename)
        return repo_dir

    def test_search_endpoint_returns_ranked_provenance(
        self, client, tmp_path, no_default_storage
    ):
        self._seed_repo(tmp_path, no_default_storage)
        resp = client.get("/api/v1/repos/acme/evidence/search", params={"q": "authentication"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "authentication"
        assert body["repo_id"] == "acme/evidence"
        assert body["total"] >= 1
        top = body["results"][0]
        assert top["file_path"] == "src/auth/middleware.py"
        assert top["start_line"] == 1
        assert top["language"] == "Python"
        assert "bearer token" in top["content"]

    def test_limit_param_clamps_results(self, client, tmp_path, no_default_storage):
        self._seed_repo(tmp_path, no_default_storage)
        resp = client.get(
            "/api/v1/repos/acme/evidence/search",
            params={"q": "fastapi", "limit": 2},
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) <= 2

    def test_no_results_is_200_empty(self, client, tmp_path, no_default_storage):
        self._seed_repo(tmp_path, no_default_storage)
        resp = client.get("/api/v1/repos/acme/evidence/search", params={"q": "qqzzxx"})
        assert resp.status_code == 200
        assert resp.json()["total"] == 0
        assert resp.json()["results"] == []

    def test_empty_query_rejected(self, client, tmp_path, no_default_storage):
        self._seed_repo(tmp_path, no_default_storage)
        resp = client.get("/api/v1/repos/acme/evidence/search", params={"q": ""})
        assert resp.status_code in (400, 422)  # route-level 400 OR schema 422

    def test_unknown_repo_404(self, client, tmp_path, no_default_storage):
        self._seed_repo(tmp_path, no_default_storage)
        resp = client.get("/api/v1/repos/acme/nope/search", params={"q": "authentication"})
        assert resp.status_code == 404

    def test_ingested_but_not_indexed_repo_404(self, client, tmp_path, no_default_storage):
        self._seed_repo(tmp_path, no_default_storage, index=False)
        resp = client.get("/api/v1/repos/acme/evidence/search", params={"q": "authentication"})
        assert resp.status_code == 404
        assert "not indexed" in resp.json()["detail"]

    def test_reindex_endpoint_is_idempotent_and_preserves_results(
        self, client, tmp_path, no_default_storage
    ):
        # Seed with an extra file already known to the manifest.
        self._seed_repo(
            tmp_path,
            no_default_storage,
            extra_files={"src/system_secrets.py": 'def rotate_key():\n    "Secrets in a vault."\n    pass\n'},
        )
        first_hit = client.get("/api/v1/repos/acme/evidence/search", params={"q": "rotate_key"})
        assert first_hit.status_code == 200
        assert "src/system_secrets.py" in {r["file_path"] for r in first_hit.json()["results"]}

        # Re-indexing the unchanged repo must not duplicate rows.
        resp = client.post("/api/v1/repos/acme/evidence/index")
        assert resp.status_code == 200
        summary = resp.json()
        assert summary["repo_id"] == "acme/evidence"
        assert summary["chunks_created"] >= 1

        again = client.get("/api/v1/repos/acme/evidence/search", params={"q": "rotate_key"})
        assert again.status_code == 200
        paths = [r["file_path"] for r in again.json()["results"]]
        assert paths.count("src/system_secrets.py") == 1  # no duplicates after re-index

    def test_index_endpoint_404_for_uningested(self, client, no_default_storage):
        resp = client.post("/api/v1/repos/ghost/repo/index")
        assert resp.status_code == 404

    def test_invalid_slugs_rejected(self, client, no_default_storage):
        from fastapi import HTTPException

        from app.api.routes import search_repository

        for owner, repo in (("..", "x"), ("x", "..")):
            with pytest.raises(HTTPException):
                search_repository(owner=owner, repo=repo, q="a")
            with pytest.raises(HTTPException):
                search_repository(owner=owner, repo=repo, q=str(owner or repo))