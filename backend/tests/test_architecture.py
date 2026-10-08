"""Tests for M7.1 deterministic architecture graph.

No LLM, no network. Fake repositories are materialized on disk, run through
the real ingestion analysis + indexing pipeline, then graphed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import get_settings
from app.models.schemas import ArchitectureResult
from app.services.architecture import (
    build_architecture_graph,
    build_module_index,
    classify_directory,
    classify_file,
    extract_js_imports,
    extract_python_imports,
    is_config_file,
    is_test_file,
    resolve_js_import,
    resolve_python_import,
)
from app.services.architecture import RawImport


def _settings():
    return get_settings()


def _setup_indexed_repo(storage: Path, files: dict[str, str],
                         owner: str = "acme", repo: str = "widget"):
    """Materialize checkout + manifest + index under the test storage root."""
    from app.services.github import FetchedRepository
    from app.services.indexing import evidence_db_path, index_repository
    from app.services.ingestion import _assemble_manifest, analyze_repository

    settings = _settings()
    repo_dir = storage / owner / repo
    checkout = repo_dir / "checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        target = checkout / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    analysis = analyze_repository(checkout, settings, tree_root_name=repo)
    fetched = FetchedRepository(
        owner=owner, repo=repo,
        github_url=f"https://github.com/{owner}/{repo}",
        checkout_root=checkout, commit_hash="abc123", default_branch="main",
    )
    manifest = _assemble_manifest(fetched, analysis, settings, None)
    (repo_dir / settings.manifest_filename).write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8")
    db_path = evidence_db_path(settings.storage_root, settings.db_filename)
    index_repository(manifest, checkout, db_path)
    return settings, f"{owner}/{repo}"


DEMO_FILES = {
    "README.md": "# Widget\nA widget app.\n",
    "pyproject.toml": '[project]\nname = "widget"\n',
    "app/api/routes.py": (
        "from fastapi import APIRouter\n"
        "from app.services.judge import judge\n"
        "from app.models.schemas import JudgeResult\n"
        "\n"
        "router = APIRouter()\n"
    ),
    "app/services/judge.py": (
        "from app.models.schemas import JudgeResult\n"
        "\n"
        "\n"
        "def judge() -> JudgeResult:\n"
        "    raise NotImplementedError\n"
    ),
    "app/models/schemas.py": "class JudgeResult:\n    pass\n",
    "app/services/external.py": "import flask\nimport requests\n",
    "tests/test_judge.py": "from app.services.judge import judge\n",
    "frontend/components/Button.tsx": (
        "import { helper } from '../utils/helper';\n"
        "export function Button() { return helper(); }\n"
    ),
    "frontend/utils/helper.ts": "export function helper() { return 1; }\n",
}


@pytest.fixture()
def demo_repo(no_default_storage: Path):
    return _setup_indexed_repo(no_default_storage, DEMO_FILES)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_valid_result_serializes(self, demo_repo):
        settings, repo_id = demo_repo
        result = build_architecture_graph(repo_id, settings=settings)
        dumped = result.model_dump(mode="json")
        assert dumped["repo_id"] == repo_id
        assert dumped["root"] == "project"
        assert dumped["total_nodes"] == len(dumped["nodes"])
        assert dumped["total_edges"] == len(dumped["edges"])
        # Round-trip through the schema.
        assert ArchitectureResult.model_validate(dumped).summary == result.summary
        json.dumps(dumped)  # must be JSON-serializable

    def test_node_edge_fields(self, demo_repo):
        settings, repo_id = demo_repo
        result = build_architecture_graph(repo_id, settings=settings)
        node_ids = {n.id for n in result.nodes}
        assert "project" in node_ids
        for n in result.nodes:
            assert n.id and n.label and n.type
            assert isinstance(n.files, list)
        for e in result.edges:
            assert e.source in node_ids
            assert e.target in node_ids
            assert e.relationship


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassification:
    def test_tests_detected(self):
        assert is_test_file("tests/test_judge.py")
        assert is_test_file("src/__tests__/a.test.ts")
        assert is_test_file("web/app.spec.js")
        assert classify_file("tests/test_judge.py") == "test"
        assert not is_test_file("app/services/judge.py")

    def test_config_detected(self):
        assert is_config_file("pyproject.toml")
        assert is_config_file("deploy/app.yaml")
        assert classify_file("package.json") == "config"
        assert not is_config_file("app/main.py")

    def test_source_roles(self):
        assert classify_file("app/api/routes.py") == "api"
        assert classify_file("app/services/judge.py") == "service"
        assert classify_file("app/models/schemas.py") == "model"
        assert classify_file("src/db/conn.py") == "database"
        assert classify_file("app/utils/x.py") == "utility"
        assert classify_file("app/main.py") == "module"

    def test_ambiguous_is_conservative(self):
        assert classify_file("notes.md") == "unknown"
        assert classify_directory("mystuff") == "directory"

    def test_directory_roles(self):
        assert classify_directory("tests") == "test"
        assert classify_directory("server") == "backend"


# ---------------------------------------------------------------------------
# Import extraction / resolution tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_python_absolute(self):
        raws = extract_python_imports(
            "import os\nfrom app.services.judge import judge\n")
        assert any(r.specifier == "app.services.judge" for r in raws)

    def test_python_relative(self):
        raws = extract_python_imports("from .sibling import thing\n")
        assert raws and raws[0].relative_level == 1
        assert raws[0].specifier == "sibling"

    def test_python_broken_syntax_falls_back(self):
        raws = extract_python_imports("import broken (((\nfrom x import y\n")
        assert any(r.specifier == "x" for r in raws)

    def test_js_relative_only(self):
        raws = extract_js_imports(
            "import React from 'react';\nimport { h } from '../utils/helper';\n")
        assert [r.specifier for r in raws] == ["../utils/helper"]

    def test_resolve_python_src_layout(self):
        files = {"src/flask/app.py", "src/flask/__init__.py"}
        index = build_module_index(files)
        target = resolve_python_import(
            "src/flask/cli.py", RawImport(specifier="flask.app", line=1), index)
        assert target == "src/flask/app.py"

    def test_resolve_js(self):
        files = {"a/b.ts", "a/c.ts"}
        target = resolve_js_import(
            "a/b.ts", RawImport(specifier="./c", line=1), files)
        assert target == "a/c.ts"

    def test_external_never_resolves(self):
        index = build_module_index({"app/main.py"})
        assert resolve_python_import(
            "app/main.py", RawImport(specifier="flask", line=1), index) is None


# ---------------------------------------------------------------------------
# Structure / relationship / evidence tests
# ---------------------------------------------------------------------------

class TestGraph:
    def test_project_root_and_hierarchy(self, demo_repo):
        settings, repo_id = demo_repo
        result = build_architecture_graph(repo_id, settings=settings)
        node_ids = {n.id for n in result.nodes}
        assert "project" in node_ids
        assert "dir:app" in node_ids
        assert "dir:app/services" in node_ids
        assert "dir:tests" in node_ids
        contains = {(e.source, e.target) for e in result.edges
                    if e.relationship == "contains"}
        assert ("project", "dir:app") in contains
        assert ("dir:app", "dir:app/services") in contains
        assert ("dir:app/services", "file:app/services/judge.py") in contains

    def test_internal_imports_resolved(self, demo_repo):
        settings, repo_id = demo_repo
        result = build_architecture_graph(repo_id, settings=settings)
        pairs = {(e.source, e.target, e.relationship) for e in result.edges}
        assert ("file:app/api/routes.py", "file:app/services/judge.py", "imports") in pairs
        assert ("file:app/api/routes.py", "file:app/models/schemas.py", "imports") in pairs
        assert ("file:frontend/components/Button.tsx",
                "file:frontend/utils/helper.ts", "imports") in pairs

    def test_test_files_use_tests_relationship(self, demo_repo):
        settings, repo_id = demo_repo
        result = build_architecture_graph(repo_id, settings=settings)
        pairs = {(e.source, e.target, e.relationship) for e in result.edges}
        assert ("file:tests/test_judge.py", "file:app/services/judge.py", "tests") in pairs

    def test_external_imports_create_nothing(self, demo_repo):
        settings, repo_id = demo_repo
        result = build_architecture_graph(repo_id, settings=settings)
        node_ids = {n.id for n in result.nodes}
        assert not any("flask" in nid for nid in node_ids)
        assert not any("requests" in nid for nid in node_ids)
        for e in result.edges:
            assert e.source in node_ids and e.target in node_ids

    def test_no_duplicate_edges(self, demo_repo):
        settings, repo_id = demo_repo
        result = build_architecture_graph(repo_id, settings=settings)
        keys = [(e.source, e.target, e.relationship) for e in result.edges]
        assert len(keys) == len(set(keys))

    def test_evidence_grounded(self, demo_repo):
        settings, repo_id = demo_repo
        result = build_architecture_graph(repo_id, settings=settings)
        valid = {c.id for c in result.evidence_citations}
        assert valid, "expected evidence citations"
        by_id = {c.id: c for c in result.evidence_citations}
        for n in result.nodes:
            for eid in n.evidence_ids:
                assert eid in valid
        for e in result.edges:
            for eid in e.evidence_ids:
                assert eid in valid
        # Provenance resolves to real files with line ranges.
        for c in result.evidence_citations:
            assert c.file_path and c.start_line >= 1 and c.end_line >= c.start_line
        # Import edge evidence points at the importing file.
        edge = next(e for e in result.edges
                    if e.source == "file:app/api/routes.py"
                    and e.target == "file:app/services/judge.py")
        assert edge.evidence_ids
        assert by_id[edge.evidence_ids[0]].file_path == "app/api/routes.py"

    def test_deterministic(self, demo_repo):
        settings, repo_id = demo_repo
        first = build_architecture_graph(repo_id, settings=settings)
        second = build_architecture_graph(repo_id, settings=settings)
        assert first.model_dump(mode="json") == second.model_dump(mode="json")

    def test_max_nodes_caps_files(self, demo_repo):
        settings, repo_id = demo_repo
        result = build_architecture_graph(repo_id, settings=settings, max_nodes=5)
        assert result.total_nodes <= 5
        assert "project" in {n.id for n in result.nodes}

    def test_docs_only_repo(self, no_default_storage: Path):
        settings, repo_id = _setup_indexed_repo(
            no_default_storage, {"README.md": "# Hi\n", "docs/guide.md": "text\n"})
        result = build_architecture_graph(repo_id, settings=settings)
        assert result.total_nodes >= 1
        assert not any(e.relationship in ("imports", "tests") for e in result.edges)
        assert "source modules" in result.summary

    def test_unindexed_raises(self, no_default_storage: Path):
        from app.services.retrieval import RepoNotIndexedError
        with pytest.raises(RepoNotIndexedError):
            build_architecture_graph("acme/ghost", settings=_settings())

    def test_missing_manifest_raises(self, no_default_storage: Path):
        from app.services.retrieval import RepoNotIndexedError
        with pytest.raises(RepoNotIndexedError):
            build_architecture_graph("acme/nothing", settings=_settings())


# ---------------------------------------------------------------------------
# API + CLI tests
# ---------------------------------------------------------------------------

class TestApi:
    def test_architecture_200(self, demo_repo):
        from fastapi.testclient import TestClient
        from app.main import app
        settings, repo_id = demo_repo
        owner, repo = repo_id.split("/")
        client = TestClient(app)
        resp = client.get(f"/api/v1/repos/{owner}/{repo}/architecture")
        assert resp.status_code == 200
        body = resp.json()
        assert body["repo_id"] == repo_id
        assert body["root"] == "project"
        assert body["total_nodes"] == len(body["nodes"])
        assert isinstance(body["summary"], str)

    def test_architecture_unknown_404(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/api/v1/repos/acme/does-not-exist-xyz/architecture")
        assert resp.status_code == 404


class TestCli:
    def test_parser(self):
        from app.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["architecture", "acme/widget"])
        assert args.command == "architecture"
        assert args.repo == "acme/widget"

    def test_json_output(self, demo_repo, capsys):
        from app.cli import main
        settings, repo_id = demo_repo
        assert main(["architecture", repo_id, "--json"]) == 0
        out = capsys.readouterr().out
        body = json.loads(out)
        assert body["repo_id"] == repo_id
        assert body["total_nodes"] > 0

    def test_human_output(self, demo_repo, capsys):
        from app.cli import main
        settings, repo_id = demo_repo
        assert main(["architecture", repo_id]) == 0
        out = capsys.readouterr().out
        assert "Repository Architecture" in out
        assert repo_id in out
