"""API-level tests for the ingestion endpoints.

Network is not required: ``fetch_repository`` is replaced with a fake that
copies the sample repository fixture into the storage layout, so the whole
route → service → manifest → persist pipeline is exercised locally.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import github as github_service
from app.services.github import FetchedRepository, RepoNotFoundError


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _fake_fetch_factory(sample_repo: Path):
    def fake_fetch(url: str, base_dir: Path, timeout: int):
        repo = github_service.parse_github_url(url)
        if repo.repo.lower() == "missing":
            raise RepoNotFoundError(f"Repository not found or not accessible: {repo.canonical_url}")
        repo_dir = base_dir / repo.owner.lower() / repo.repo.lower()
        checkout = repo_dir / "checkout"
        checkout.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sample_repo, checkout, dirs_exist_ok=True)
        return FetchedRepository(
            owner=repo.owner, repo=repo.repo, github_url=repo.canonical_url, checkout_root=checkout
        )

    return fake_fetch


class TestIngestEndpoint:
    def test_ingest_returns_manifest(self, client, sample_repo, monkeypatch, no_default_storage):
        monkeypatch.setattr("app.services.ingestion.fetch_repository", _fake_fetch_factory(sample_repo))

        response = client.post(
            "/api/v1/repos", json={"url": "https://github.com/acme/widget"}
        )
        assert response.status_code == 201
        manifest = response.json()
        assert manifest["id"] == "acme/widget"
        assert manifest["owner"] == "acme"
        assert manifest["repo"] == "widget"
        assert manifest["readme_present"] is True
        assert manifest["package_manager"] == "npm"
        names = {f["name"] for f in manifest["frameworks"]}
        assert "React" in names
        assert manifest["total_files"] == 6
        assert any(e["path"] == "src/main.ts" for e in manifest["file_inventory"])

    def test_ingest_persists_manifest_file(self, client, sample_repo, monkeypatch, no_default_storage):
        monkeypatch.setattr("app.services.ingestion.fetch_repository", _fake_fetch_factory(sample_repo))

        client.post("/api/v1/repos", json={"url": "owner/repo-short"})
        manifest_file = no_default_storage / "owner" / "repo-short" / "manifest.json"
        assert manifest_file.exists()
        assert "repo-short" in manifest_file.read_text()

    def test_invalid_url_returns_400(self, client, sample_repo, monkeypatch):
        monkeypatch.setattr("app.services.ingestion.fetch_repository", _fake_fetch_factory(sample_repo))
        response = client.post("/api/v1/repos", json={"url": "https://gitlab.com/a/b"})
        assert response.status_code == 400
        assert "Invalid repository URL" in response.json()["detail"]

    def test_unparseable_url_returns_400(self, client, sample_repo, monkeypatch):
        monkeypatch.setattr("app.services.ingestion.fetch_repository", _fake_fetch_factory(sample_repo))
        response = client.post("/api/v1/repos", json={"url": "not a url"})
        assert response.status_code == 400

    def test_missing_repo_returns_404(self, client, sample_repo, monkeypatch):
        monkeypatch.setattr("app.services.ingestion.fetch_repository", _fake_fetch_factory(sample_repo))
        response = client.post("/api/v1/repos", json={"url": "https://github.com/acme/missing"})
        assert response.status_code == 404


class TestGetManifestEndpoint:
    def test_get_ingested_manifest(self, client, sample_repo, monkeypatch, no_default_storage):
        monkeypatch.setattr("app.services.ingestion.fetch_repository", _fake_fetch_factory(sample_repo))
        client.post("/api/v1/repos", json={"url": "https://github.com/acme/widget"})

        response = client.get("/api/v1/repos/acme/widget")
        assert response.status_code == 200
        assert response.json()["id"] == "acme/widget"

    def test_get_unknown_returns_404(self, client, no_default_storage):
        response = client.get("/api/v1/repos/acme/does-not-exist")
        assert response.status_code == 404

    def test_traversal_slug_rejected(self, client):
        # Direct call (HTTP clients normalise '/' + '..', so exercise the guard
        # at the route-function boundary): a traversal slug must not resolve.
        from fastapi import HTTPException

        from app.api.routes import get_repository_manifest

        for owner, repo in (("..", ".."), (".", "widget"), ("acme", ".."), ("..", ".")):
            with pytest.raises(HTTPException) as excinfo:
                get_repository_manifest(owner, repo)
            assert excinfo.value.status_code == 400


class TestHealth:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"