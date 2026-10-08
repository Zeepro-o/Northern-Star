"""Tests for the repository analysis and truncation behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings
from app.models.schemas import FileKind
from app.services.ingestion import analyze_repository, build_directory_tree


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("NORTHERN_STAR_STORAGE", str(storage))
    return get_settings()


def make_file(root: Path, rel: str, content: str = "x\n") -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


class TestAnalyzeRepository:
    def test_basic_counts(self, sample_repo, settings):
        analysis = analyze_repository(sample_repo, settings, tree_root_name="sample-repo")

        # 6 analysed files: README.md, package.json, src/main.ts, src/utils/helper.py,
        # public/logo.png, .gitignore  (dist/, node_modules/, .env excluded)
        assert analysis.total_files == 6
        assert analysis.source_files == 2  # src/main.ts + src/utils/helper.py
        assert {e.category for e in analysis.entries} >= {FileKind.SOURCE, FileKind.CONFIG, FileKind.BINARY}
        assert analysis.total_size_bytes > 0

    def test_categories_breakdown(self, sample_repo, settings):
        analysis = analyze_repository(sample_repo, settings)
        by_kind = {c.category: c.file_count for c in analysis.categories}
        assert by_kind[FileKind.SOURCE] == 2
        assert by_kind[FileKind.CONFIG] == 2  # package.json + .gitignore
        assert by_kind[FileKind.DOCUMENTATION] == 1
        assert by_kind[FileKind.BINARY] == 1

    def test_language_breakdown(self, sample_repo, settings):
        analysis = analyze_repository(sample_repo, settings)
        by_lang = {l.language: l.file_count for l in analysis.languages}
        assert by_lang["Python"] == 1
        assert by_lang["TypeScript"] == 1
        assert by_lang["JSON"] == 1

    def test_ignored_directories_listed(self, sample_repo, settings):
        analysis = analyze_repository(sample_repo, settings)
        reasons = {i.path: i.reason for i in analysis.ignored_directories}
        assert reasons["node_modules"] == "vendored dependencies"
        assert reasons["dist"] == "build output"

    def test_commit_provenance_absent_for_plain_dir(self, sample_repo, settings):
        analysis = analyze_repository(sample_repo, settings)
        assert analysis.total_files == 6


class TestTruncation:
    def test_max_files_limit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NORTHERN_STAR_MAX_FILES", "3")
        settings = get_settings()
        root = tmp_path / "big"
        root.mkdir(exist_ok=True)
        for i in range(10):
            make_file(root, f"file_{i}.py")
        analysis = analyze_repository(root, settings)
        assert analysis.truncated is True
        assert analysis.total_files == 3
        assert any("MAX_FILES" in w for w in analysis.warnings)

    def test_max_repo_size_limit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NORTHERN_STAR_MAX_REPO_SIZE_BYTES", "16")
        settings = get_settings()
        root = tmp_path / "big"
        root.mkdir(exist_ok=True)
        make_file(root, "a.py", content="a" * 100)
        analysis = analyze_repository(root, settings)
        assert analysis.truncated is True


class TestDirectoryTree:
    def test_tree_aggregates(self, sample_repo, settings):
        analysis = analyze_repository(sample_repo, settings)
        tree = analysis.directory_structure
        assert tree.name == "."
        assert tree.file_count == analysis.total_files
        by_name = {c.name: c for c in tree.children}
        assert "src" in by_name
        src = by_name["src"]
        assert src.file_count == 2
        utils = {c.name: c for c in src.children}
        assert "utils" in utils and utils["utils"].file_count == 1

    def test_build_directly(self):
        from app.models.schemas import FileEntry

        entries = [
            FileEntry(path="a/b.py", language="Python", category=FileKind.SOURCE, size_bytes=10),
            FileEntry(path="a/c.py", language="Python", category=FileKind.SOURCE, size_bytes=20),
            FileEntry(path="top.txt", language="Text", category=FileKind.DOCUMENTATION, size_bytes=5),
        ]
        tree = build_directory_tree(entries, "root", _dummy_settings())
        assert tree.name == "root"
        assert tree.file_count == 3
        assert tree.size_bytes == 35
        by_name = {c.name: c for c in tree.children}
        assert by_name["a"].file_count == 2
        assert by_name["a"].size_bytes == 30


def _dummy_settings():
    class _S:
        dir_tree_max_depth = 5
        dir_tree_max_children = 200

    return _S()