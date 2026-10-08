"""Tests for the SQLite evidence index (schema, insertion, idempotency)."""

from __future__ import annotations

import shutil

from app.config import get_settings
from app.services.github import FetchedRepository
from app.services.indexing import (
    INDEXABLE_CATEGORIES,
    connect_evidence_db,
    evidence_db_path,
    index_repository,
    repo_is_indexed,
)
from app.services.ingestion import _assemble_manifest, analyze_repository


def _write_files(root, files: dict[str, str]) -> None:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _manifest_for(checkout, owner: str = "acme", repo: str = "evidence") -> tuple:
    """Analyse a checkout and assemble a manifest the indexer can consume."""
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


def _table_names(db_path) -> set[str]:
    with connect_evidence_db(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        return {r[0] for r in rows}


def _count(db_path, table: str) -> int:
    with connect_evidence_db(db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _fts_count(db_path) -> int:
    with connect_evidence_db(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]


class TestSchema:
    def test_all_tables_created(self, evidence_repo):
        _, _, db_path = evidence_repo
        tables = _table_names(db_path)
        assert {"repositories", "files", "chunks", "chunks_fts"} <= tables

    def test_evidence_db_path_resolution(self, evidence_repo):
        _, _, db_path = evidence_repo
        settings = get_settings()
        assert evidence_db_path(settings.storage_root, settings.db_filename) == db_path


class TestIndexRepository:
    def test_index_populates_all_tables(self, evidence_repo):
        checkout, manifest, db_path = evidence_repo
        stats = index_repository(manifest, checkout, db_path)
        assert stats.repo_id == "acme/evidence"
        assert stats.files_indexed == len(manifest.file_inventory)
        assert stats.chunks_created > 0
        assert _count(db_path, "repositories") == 1
        assert _count(db_path, "files") == stats.files_indexed
        assert _count(db_path, "chunks") == stats.chunks_created

    def test_every_chunk_has_full_provenance(self, evidence_repo):
        checkout, manifest, db_path = evidence_repo
        index_repository(manifest, checkout, db_path)
        with connect_evidence_db(db_path) as conn:
            rows = conn.execute(
                "SELECT repo_id, file_id, file_path, language, start_line, end_line, content"
                " FROM chunks"
            ).fetchall()
        assert rows
        for r in rows:
            assert r["repo_id"] == "acme/evidence"
            assert r["file_id"] > 0
            assert r["file_path"]
            assert r["start_line"] >= 1 and r["end_line"] >= r["start_line"]
            assert r["content"]

    def test_repository_row_records_git_provenance(self, evidence_repo):
        checkout, manifest, db_path = evidence_repo
        index_repository(manifest, checkout, db_path)
        with connect_evidence_db(db_path) as conn:
            row = conn.execute(
                "SELECT id, commit_hash, default_branch FROM repositories"
            ).fetchone()
        assert row["id"] == "acme/evidence"
        assert row["commit_hash"] == "deadbeef"
        assert row["default_branch"] == "main"

    def test_large_file_is_split(self, evidence_repo):
        checkout, manifest, db_path = evidence_repo
        index_repository(manifest, checkout, db_path)
        with connect_evidence_db(db_path) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE file_path = 'src/generated_big.py'"
            ).fetchone()[0]
        assert n > 1  # >150-line file must be split into sequential chunks

    def test_content_lines_match_provenance(self, evidence_repo):
        checkout, manifest, db_path = evidence_repo
        index_repository(manifest, checkout, db_path)
        with connect_evidence_db(db_path) as conn:
            row = conn.execute(
                "SELECT start_line, end_line, content FROM chunks"
                " WHERE file_path = 'src/auth/middleware.py' ORDER BY start_line"
            ).fetchone()
        expected = (checkout / "src/auth/middleware.py").read_text().splitlines()
        assert row["start_line"] == 1
        assert row["end_line"] == len(expected)
        assert len(row["content"].splitlines()) == len(expected)


class TestReindexIdempotent:
    def test_reindex_does_not_duplicate(self, evidence_repo):
        checkout, manifest, db_path = evidence_repo
        first = index_repository(manifest, checkout, db_path)
        second = index_repository(manifest, checkout, db_path)
        assert second.files_indexed == first.files_indexed
        assert second.chunks_created == first.chunks_created
        assert _count(db_path, "files") == first.files_indexed
        assert _count(db_path, "chunks") == first.chunks_created

    def test_reindex_updates_content_without_fiss_residue(self, evidence_repo):
        checkout, manifest, db_path = evidence_repo
        index_repository(manifest, checkout, db_path)
        original_fts = _fts_count(db_path)

        # Change a file's content, re-analyse, re-index: old chunk gone.
        (checkout / "src/db/connection.py").write_text(
            "def connect_database(path):  # renamed signature now\n    pass\n"
        )
        new_manifest, _ = _manifest_for(checkout)
        index_repository(new_manifest, checkout, db_path)

        with connect_evidence_db(db_path) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE file_path = 'src/db/connection.py'"
            ).fetchone()[0]
        assert n == 1  # whole small file, one chunk
        assert _count(db_path, "repositories") == 1


class TestFiltering:
    def test_only_indexable_categories_stored(self, tmp_path, no_default_storage):
        checkout = tmp_path / "mixed"
        checkout.mkdir(parents=True, exist_ok=True)
        _write_files(checkout, {
            "app.py": "print('hi')\n",                          # SOURCE
            "data.csv": "a,b\n1,2\n",                           # DATA
            "logo.png": b"\x89PNG\r\n\x1a\nfake".decode("latin-1"),  # BINARY
            "docs/readme.md": "# docs\n",                       # DOCUMENTATION
            "notes.frob": "something unusual\n",                # OTHER (no mapping)
        })
        manifest, _ = _manifest_for(checkout)
        assert len(manifest.file_inventory) == 5

        db_path = no_default_storage / "mixed.db"
        stats = index_repository(manifest, checkout, db_path)
        assert stats.files_indexed == 3  # app.py, data.csv, docs/readme.md
        with connect_evidence_db(db_path) as conn:
            indexed = {r["path"] for r in conn.execute("SELECT path FROM files")}
        assert indexed == {"app.py", "data.csv", "docs/readme.md"}

    def test_binary_and_secret_files_never_reach_index(self, sample_repo, tmp_path):
        index_repository(
            _manifest_for(sample_repo)[0], sample_repo, tmp_path / "idx.db"
        )
        with connect_evidence_db(tmp_path / "idx.db") as conn:
            paths = {r["path"] for r in conn.execute("SELECT path FROM files")}
        assert ".env" not in paths             # secrets skipped by M1
        assert "public/logo.png" not in paths  # binary excluded
        assert "dist/bundle.js" not in paths   # build output excluded


class TestRepoIsolation:
    def test_two_repositories_share_db_without_cross_contamination(
        self, evidence_repo, tmp_path
    ):
        checkout, manifest, db_path = evidence_repo
        index_repository(manifest, checkout, db_path)

        other_checkout = tmp_path / "other-checkout"
        shutil.copytree(checkout, other_checkout)
        other_manifest, _ = _manifest_for(other_checkout, owner="elsewhere", repo="other")
        assert other_manifest.id != manifest.id
        index_repository(other_manifest, other_checkout, db_path)

        assert _count(db_path, "repositories") == 2
        with connect_evidence_db(db_path) as conn:
            repos = {r["id"] for r in conn.execute("SELECT id FROM repositories")}
            assert repos == {"acme/evidence", "elsewhere/other"}
            acme_chunks = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE repo_id = 'acme/evidence'"
            ).fetchone()["n"]
            other_chunks = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE repo_id = 'elsewhere/other'"
            ).fetchone()["n"]
        assert acme_chunks > 0 and other_chunks > 0
        assert _count(db_path, "chunks") == acme_chunks + other_chunks

    def test_repo_is_indexed_flag(self, evidence_repo):
        checkout, manifest, db_path = evidence_repo
        assert repo_is_indexed(db_path, "acme/evidence") is False
        index_repository(manifest, checkout, db_path)
        assert repo_is_indexed(db_path, "acme/evidence") is True


def test_indexable_categories_are_expected_subset():
    values = {c.value for c in INDEXABLE_CATEGORIES}
    assert "binary" not in values
    assert "build" not in values
    assert {"source", "config", "documentation", "data"} <= values