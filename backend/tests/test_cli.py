"""Tests for the in-process CLI — offline: no network, no server, no Ollama."""

from __future__ import annotations

import json
import shutil

import pytest

from app.cli import main
from app.config import get_settings
from app.models.schemas import Claim
from app.services.github import FetchedRepository, InvalidGitHubUrlError
from app.services.indexing import index_repository
from app.services.ingestion import _assemble_manifest, analyze_repository
from tests.conftest import build_evidence_checkout


def _seed_repo(tmp_path, no_default_storage, owner="acme", repo="evidence",
               index=True):
    """Persist an ingested fixture repo into the test storage."""
    settings = get_settings()
    checkout = build_evidence_checkout(tmp_path / "fixtures")
    analysis = analyze_repository(checkout, settings, tree_root_name="evidence")
    fetched = FetchedRepository(
        owner=owner,
        repo=repo,
        github_url=f"https://github.com/{owner}/{repo}",
        checkout_root=checkout,
        commit_hash="deadbeef",
        default_branch="main",
    )
    manifest = _assemble_manifest(fetched, analysis, settings, None)
    repo_dir = no_default_storage / owner / repo
    (repo_dir / "checkout").mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkout, repo_dir / "checkout", dirs_exist_ok=True)
    (repo_dir / settings.manifest_filename).write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    if index:
        index_repository(manifest, checkout, no_default_storage / settings.db_filename)
    return repo_dir


class TestArgumentParsing:
    def test_no_command_prints_help_exit_2(self, capsys):
        assert main([]) == 2
        assert "usage:" in capsys.readouterr().err

    def test_unknown_command_exit_2(self, capsys):
        assert main(["frobnicate"]) == 2

    def test_bad_owner_repo_exits_1(self, no_default_storage, capsys):
        for bad in ("norepo", "..//x", "a//b", "/"):
            assert main(["manifest", bad]) == 1
            assert "owner/repo" in capsys.readouterr().err

    def test_missing_question_exit_2(self):
        assert main(["ask", "acme/evidence"]) == 2


class TestManifest:
    def test_manifest_exit_0_prints_id(self, tmp_path, no_default_storage, capsys):
        _seed_repo(tmp_path, no_default_storage)
        code = main(["manifest", "acme/evidence"])
        out = capsys.readouterr().out
        assert code == 0
        assert "acme/evidence" in out
        assert "languages:" in out

    def test_manifest_not_ingested_exit_1(self, no_default_storage, capsys):
        assert main(["manifest", "ghost/nope"]) == 1
        assert "not been ingested" in capsys.readouterr().err

    def test_manifest_json(self, tmp_path, no_default_storage, capsys):
        _seed_repo(tmp_path, no_default_storage)
        assert main(["manifest", "acme/evidence", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["id"] == "acme/evidence"
        assert payload["total_files"] > 0


class TestSearch:
    def test_search_finds_provenance(self, tmp_path, no_default_storage, capsys):
        _seed_repo(tmp_path, no_default_storage)
        code = main(["search", "acme/evidence", "authentication", "--limit", "5"])
        out = capsys.readouterr().out
        assert code == 0
        assert "src/auth/middleware.py" in out
        assert "1-10" in out  # exact line-range provenance rendered

    def test_search_json_parses(self, tmp_path, no_default_storage, capsys):
        _seed_repo(tmp_path, no_default_storage)
        assert main(["search", "acme/evidence", "authentication", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["repo_id"] == "acme/evidence"
        assert payload["results"]
        top = payload["results"][0]
        assert top["file_path"] == "src/auth/middleware.py"
        assert top["start_line"] >= 1

    def test_search_not_indexed_exit_1(self, tmp_path, no_default_storage, capsys):
        _seed_repo(tmp_path, no_default_storage, index=False)
        assert main(["search", "acme/evidence", "q"]) == 1
        assert "not indexed" in capsys.readouterr().err


class TestIndex:
    def test_index_reports_counts(self, tmp_path, no_default_storage, capsys):
        _seed_repo(tmp_path, no_default_storage, index=False)
        code = main(["index", "acme/evidence"])
        out = capsys.readouterr().out
        assert code == 0
        assert "chunks:" in out

    def test_index_manifests_absent_exit_1(self, no_default_storage, capsys):
        assert main(["index", "ghost/nope"]) == 1


class TestIngest:
    def test_ingest_offline_success(self, monkeypatch, no_default_storage, capsys):
        fake = FetchedRepository(
            owner="acme", repo="widget", github_url="https://github.com/acme/widget",
            checkout_root=no_default_storage, commit_hash="abc123", default_branch="main",
        )
        analysis = analyze_repository(build_evidence_checkout(no_default_storage / "fx"),
                                      get_settings(), tree_root_name="widget")
        manifest = _assemble_manifest(fake, analysis, get_settings(), None)

        monkeypatch.setattr("app.cli.ingest_github_repo",
                            lambda url, settings, ref=None: manifest)
        code = main(["ingest", "https://github.com/acme/widget"])
        out = capsys.readouterr().out
        assert code == 0
        assert "acme/widget" in out

    def test_ingest_bad_url_exit_1(self, monkeypatch, no_default_storage, capsys):
        def boom(url, settings, ref=None):
            raise InvalidGitHubUrlError("needs http")

        monkeypatch.setattr("app.cli.ingest_github_repo", boom)
        assert main(["ingest", "git@github.com:acme/x.git"]) == 1
        assert "needs http" in capsys.readouterr().err


class TestAsk:
    def _stub_llm(self, monkeypatch, content: str):
        """Route qa's Ollama call to a canned JSON answer (offline)."""

        class Stub:
            def complete(self, messages):
                return content

        monkeypatch.setattr("app.services.qa.OllamaClient", lambda *a, **kw: Stub())
        return Stub

    def test_ask_exit_0_prints_answer_and_citations(
        self, tmp_path, no_default_storage, monkeypatch, capsys
    ):
        _seed_repo(tmp_path, no_default_storage)
        self._stub_llm(
            monkeypatch,
            '{"answer": "Auth is enforced in [E1].", "citations": ["E1"], '
            '"confidence": "high", "evidence_sufficient": true}',
        )
        # "authentication" reliably retrieves src/auth/middleware.py (M2 fixture).
        code = main(["ask", "acme/evidence", "authentication", "--top-k", "5"])
        out = capsys.readouterr().out
        assert code == 0
        assert "Auth is enforced in [E1]." in out
        # M3.1: the rendering must label confidence as model-reported — never
        # imply Northern Star proved the answer — and show grounding + count.
        assert "confidence: high (model-reported)" in out
        assert "evidence_sufficient: True" in out
        assert "evidence_grounding: cited" in out
        assert "evidence_blocks: 1" in out
        assert "src/auth/middleware.py" in out
        # Deterministic kind label from the manifest (a Python file → source).
        assert "[code]" in out

    def test_ask_json_parses_with_citation_provenance(
        self, tmp_path, no_default_storage, monkeypatch, capsys
    ):
        _seed_repo(tmp_path, no_default_storage)
        # The answer must anchor [E1] inline: under the AND rule a citation the
        # text never references is dropped, so a bracketless stub would yield
        # zero citations and fail the assertions below.
        self._stub_llm(
            monkeypatch,
            '{"answer": "In [E1].", "citations": ["E1"], "confidence": "medium", '
            '"evidence_sufficient": true}',
        )
        assert main(["ask", "acme/evidence", "authentication", "--json", "--top-k", "3"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["evidence_sufficient"] is True
        assert payload["citations"][0]["file_path"] == "src/auth/middleware.py"
        assert payload["citations"][0]["start_line"] >= 1
        # M3.1 honesty fields are exposed to --json consumers.
        assert payload["confidence_source"] == "model"
        assert payload["evidence_grounding"] == "cited"

    def test_ask_model_flag_replaces_settings(self, tmp_path, no_default_storage, monkeypatch, capsys):
        _seed_repo(tmp_path, no_default_storage)
        seen = {}

        def fake_answer(repo_id, question, **kw):
            seen.update(kw)
            from app.models.schemas import AnswerResponse
            return AnswerResponse(question=question, repo_id=repo_id,
                                  answer="x", citations=[], confidence="low",
                                  evidence_sufficient=False,
                                  confidence_source="model",
                                  evidence_grounding="none")

        monkeypatch.setattr("app.services.qa.answer_question", fake_answer)
        assert main(["ask", "acme/evidence", "q", "--model", "qwen2.5:3b"]) == 0
        assert seen["settings"].ollama_model == "qwen2.5:3b"
        assert seen["top_k"] is None  # default QA_TOP_K applies

    def test_ask_renders_evidence_kind_tags(
        self, tmp_path, no_default_storage, monkeypatch, capsys
    ):
        """Deterministic [code] / [documentation] labels from the manifest."""
        _seed_repo(tmp_path, no_default_storage)
        # "authentication" retrieves src/auth/middleware.py → a Python source
        # file, whose FileKind is "source" → rendered as [code].
        self._stub_llm(
            monkeypatch,
            '{"answer": "Auth is enforced in [E1].", "citations": ["E1"], '
            '"confidence": "high", "evidence_sufficient": true}',
        )
        assert main(["ask", "acme/evidence", "authentication", "--top-k", "5"]) == 0
        out = capsys.readouterr().out
        assert "src/auth/middleware.py:" in out
        assert "[code]" in out
        # "demonstration" retrieves README.md → documentation → [documentation].
        self._stub_llm(
            monkeypatch,
            '{"answer": "The purpose is documented in [E1].", "citations": ["E1"], '
            '"confidence": "medium", "evidence_sufficient": true}',
        )
        assert main(["ask", "acme/evidence", "demonstration", "--top-k", "5"]) == 0
        out = capsys.readouterr().out
        assert "README.md:" in out
        assert "[documentation]" in out

    def test_ask_unindexed_exit_1(self, tmp_path, no_default_storage, capsys):
        _seed_repo(tmp_path, no_default_storage, index=False)
        assert main(["ask", "acme/evidence", "q"]) == 1
        assert "not indexed" in capsys.readouterr().err


class TestVersion:
    def test_version_prints_app_version(self, capsys):
        assert main(["version"]) == 0
        assert capsys.readouterr().out.strip() == "0.3.0"


class TestClaims:
    def test_claims_exit_0_prints_claims(self, tmp_path, no_default_storage, monkeypatch, capsys):
        _seed_repo(tmp_path, no_default_storage)
        assert main(["claims", "acme/evidence"]) == 0
        out = capsys.readouterr().out
        assert "Repository: acme/evidence" in out
        assert "README claims found:" in out
        # The evidence repo's README has no claim keywords, so count is 0

    def test_claims_json_parses(self, tmp_path, no_default_storage, monkeypatch):
        _seed_repo(tmp_path, no_default_storage)
        assert main(["claims", "acme/evidence", "--json"]) == 0

    def test_claims_not_ingested_exit_1(self, no_default_storage, capsys):
        assert main(["claims", "ghost/nope"]) == 1
        assert "not been ingested" in capsys.readouterr().err


class TestVerify:
    def test_verify_exit_0_prints_verdict(self, tmp_path, no_default_storage, monkeypatch, capsys):
        _seed_repo(tmp_path, no_default_storage)
        self._stub_llm(
            monkeypatch,
            '{"verdict": "supported", "explanation": "Auth is enforced in [E1].", "citations": ["E1"]}',
        )
        # "authentication" matches src/auth/middleware.py in the evidence fixture
        assert main(["verify", "acme/evidence", "authentication is required"]) == 0
        out = capsys.readouterr().out
        assert "verdict: supported" in out
        assert "E1" in out

    def test_verify_unclear_when_no_evidence(self, tmp_path, no_default_storage, monkeypatch, capsys):
        _seed_repo(tmp_path, no_default_storage)
        self._stub_llm(monkeypatch, "should not be called")
        # The claim "uses Cassandra" won't match any evidence in the test repo
        assert main(["verify", "acme/evidence", "uses Cassandra"]) == 0
        out = capsys.readouterr().out
        assert "verdict: unclear" in out

    def test_verify_json_parses(self, tmp_path, no_default_storage, monkeypatch):
        _seed_repo(tmp_path, no_default_storage)
        self._stub_llm(
            monkeypatch,
            '{"verdict": "contradicted", "explanation": "Uses MongoDB [E1].", "citations": ["E1"]}',
        )
        assert main(["verify", "acme/evidence", "uses Redis", "--json"]) == 0

    def test_verify_model_flag_replaces_settings(self, tmp_path, no_default_storage, monkeypatch):
        _seed_repo(tmp_path, no_default_storage)
        seen = {}

        def fake_verify(repo_id, claim, **kw):
            seen.update(kw)
            return Claim(
                id="claim_0",
                text=claim.text,
                source=claim.source,
                kind=claim.kind,
                category=claim.category,
                verdict="unclear",
                verdict_explanation="",
                evidence_ids=[],
                repo_id=repo_id,
            )

        monkeypatch.setattr("app.services.claims.verify_claim", fake_verify)
        from app.cli import main
        assert main(["verify", "acme/evidence", "test claim", "--model", "qwen2.5:3b"]) == 0
        assert seen["settings"].ollama_model == "qwen2.5:3b"

    def test_verify_unindexed_exit_1(self, tmp_path, no_default_storage, capsys):
        _seed_repo(tmp_path, no_default_storage, index=False)
        assert main(["verify", "acme/evidence", "test claim"]) == 1
        assert "not indexed" in capsys.readouterr().err

    def _stub_llm(self, monkeypatch, response_json: str):
        """Patch the LLM client used by the claims service."""

        class Stub:
            def complete(self, messages):
                return response_json

        monkeypatch.setattr("app.services.claims.OllamaClient", lambda *a, **kw: Stub())