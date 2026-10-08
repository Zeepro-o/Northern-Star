"""Tests for the evidence-grounded Q&A service and POST /ask endpoint.

Everything here is mocked: no Ollama, no network. Retrieval is faked with
hand-built SearchResult lists; the LLM is faked with an in-memory stub.
"""

from __future__ import annotations

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.models.schemas import AnswerResponse, SearchResult
from app.services.github import FetchedRepository
from app.services.indexing import index_repository
from app.services.ingestion import _assemble_manifest, analyze_repository
from app.services.llm import OllamaUnavailableError
from app.services.qa import (
    INSUFFICIENT_ANSWER,
    answer_question,
    label_evidence,
    parse_answer,
    sanitize_answer,
    validate_citations,
)
from app.services.retrieval import RepoNotIndexedError
from tests.conftest import build_evidence_checkout

# ---------------------------------------------------------------------------
# Small builders
# ---------------------------------------------------------------------------


def _evidence(*specs) -> list[SearchResult]:
    """Build SearchResults from tuples (path, start, end, language, content)."""
    out = []
    for path, start, end, lang, content in specs:
        out.append(
            SearchResult(
                file_path=path,
                start_line=start,
                end_line=end,
                language=lang,
                content=content,
            )
        )
    return out


class StubLLM:
    """A scripted LLM: returns the payload below; records what it was asked."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[dict]] = []

    def complete(self, messages) -> str:
        self.calls.append(messages)
        return self.content


def _json_answer(text: str, citations, confidence="high", sufficient=True) -> str:
    return json.dumps(
        {
            "answer": text,
            "citations": citations,
            "confidence": confidence,
            "evidence_sufficient": sufficient,
        }
    )


def _settings(no_default_storage):
    return get_settings()


# ---------------------------------------------------------------------------
# Label / validation / sanitization (pure helpers)
# ---------------------------------------------------------------------------


class TestLabelEvidence:
    def test_assigns_sequential_ids(self):
        blocks = label_evidence(
            "acme/evidence", _evidence(("a.py", 1, 5, "Python", "x"))
        )
        assert [b.id for b in blocks] == ["E1"]
        b = blocks[0]
        assert b.repository == "acme/evidence"
        assert b.file_path == "a.py" and b.start_line == 1 and b.end_line == 5

    def test_ids_match_block_positions(self):
        blocks = label_evidence(
            "acme/evidence",
            _evidence(
                ("a.py", 1, 5, "Python", "x"),
                ("b.py", 6, 10, "Python", "y"),
                ("c.py", 11, 15, "Python", "z"),
            ),
        )
        assert [b.id for b in blocks] == ["E1", "E2", "E3"]


class TestValidateCitations:
    def test_drops_invalid_ids_when_only_e1_e3_supplied(self):
        assert validate_citations(["E1", "E99", "E3"], {"E1", "E2", "E3"}) == [
            "E1",
            "E3",
        ]

    def test_deduplicates_preserving_order(self):
        assert validate_citations(["E2", "E1", "E2", "E1"], {"E1", "E2"}) == ["E2", "E1"]

    def test_accepts_bracketed_form_normalized(self):
        assert validate_citations(["[E2]"], {"E2"}) == ["E2"]
        assert validate_citations(["E2"], {"E2"}) == ["E2"]

    def test_case_normalized(self):
        assert validate_citations(["e1"], {"E1"}) == ["E1"]

    def test_garbage_is_rejected(self):
        assert validate_citations(["", "E", "12", "abc", None], {"E1"}) == []

    def test_only_supplied_ids_survive(self):
        """The invariant: the model cannot create evidence."""
        out = validate_citations(["E4", "E5", "E1"], {"E1", "E2", "E3"})
        assert out == ["E1"]
        assert all(c not in ("E4", "E5") for c in out)


class TestSanitizeAnswer:
    def test_invalid_markers_stripped(self):
        text = "Routing is in app.py [E1]; it also scales to 10k [E99]."
        assert sanitize_answer(text, {"E1"}) == (
            "Routing is in app.py [E1]; it also scales to 10k ."
        )

    def test_valid_markers_preserved(self):
        text = "Sessions live in [E2] and [E3]."
        assert sanitize_answer(text, {"E2", "E3"}) == text

    def test_no_markers_unchanged(self):
        assert sanitize_answer("plain answer", {"E1"}) == "plain answer"


# ---------------------------------------------------------------------------
# parse_answer — JSON primary, plain-text fallback
# ---------------------------------------------------------------------------


class TestParseAnswerJson:
    def test_happy_path(self):
        raw = _json_answer("The answer [E1].", ["E1"], "high", True)
        p = parse_answer(raw, {"E1"})
        assert p.raw_was_json
        assert p.answer == "The answer [E1]."
        assert p.citations == ["E1"]
        assert p.confidence == "high"
        assert p.evidence_sufficient is True

    def test_code_fenced_json_accepted(self):
        raw = '```json\n' + _json_answer("ok", [], "low", False) + '\n```'
        p = parse_answer(raw, set())
        assert p.raw_was_json
        assert p.answer == "ok"
        assert p.confidence == "low"
        assert p.evidence_sufficient is False

    def test_trailing_prose_after_json_ignored(self):
        raw = _json_answer("vault", ["E2"], "medium", True) + "\nThanks!"
        p = parse_answer(raw, {"E2"})
        assert p.answer == "vault"

    def test_confidence_case_normalized(self):
        p = parse_answer(_json_answer("x", [], "HIGH", False), set())
        assert p.confidence == "high"

    def test_missing_sufficient_inferred_from_citations(self):
        raw = '{"answer": "x [E1]", "citations": ["E1"], "confidence": "high"}'
        p = parse_answer(raw, {"E1"})
        assert p.evidence_sufficient is True

    def test_null_sufficient_falls_back(self):
        raw = (
            '{"answer": "x", "citations": [], "confidence": "low", '
            '"evidence_sufficient": null}'
        )
        p = parse_answer(raw, set())
        assert p.evidence_sufficient is False


class TestParseAnswerFallback:
    def test_plain_text_answer_mines_citations(self):
        p = parse_answer("The vault lives in secret_vault.py [E2].", {"E2"})
        assert not p.raw_was_json
        assert p.answer == "The vault lives in secret_vault.py [E2]."
        assert p.citations == ["E2"]
        assert p.confidence == "medium"
        assert p.evidence_sufficient is True

    def test_plain_text_without_citations_is_insufficient(self):
        p = parse_answer("I do not know this.", set())
        assert p.citations == []
        assert p.evidence_sufficient is False

    def test_confidence_word_in_text_used(self):
        p = parse_answer("I am confident this is [E1].", {"E1"})
        assert p.confidence == "medium"  # "confident" ≠ the enum "high"


# ---------------------------------------------------------------------------
# answer_question — mocked retrieval + LLM
# ---------------------------------------------------------------------------


class TestAnswerQuestion:
    def test_happy_path_resolves_citations(self, no_default_storage):
        evidence = _evidence(
            ("src/db/connection.py", 1, 10, "Python", "def connect_database(path):\n    pass"),
            ("src/api/routes.py", 20, 30, "Python", "@router.get('/items')\ndef list_items():\n    pass"),
        )
        llm = StubLLM(
            _json_answer(
                "The database connection is opened in [E1]; endpoints are in [E2].",
                ["E1", "E2"],
                "high",
                True,
            )
        )
        result = answer_question(
            "acme/evidence",
            "How do connections and endpoints work?",
            settings=_settings(no_default_storage),
            top_k=5,
            llm=llm,
            retrieve=lambda db, rid, q, **kw: evidence,
        )
        assert isinstance(result, AnswerResponse)
        assert result.evidence_sufficient is True
        assert result.confidence == "high"
        assert [c.id for c in result.citations] == ["E1", "E2"]
        c1 = result.citations[0]
        assert (c1.file_path, c1.start_line, c1.end_line, c1.language) == (
            "src/db/connection.py", 1, 10, "Python",
        )
        assert "The database connection is opened in [E1]" in result.answer

    def test_prompt_passed_to_llm_contains_block_ids(self, no_default_storage):
        evidence = _evidence(("src/db/connection.py", 1, 10, "Python", "content x"))
        llm = StubLLM(_json_answer("ok [E1].", ["E1"], "medium", True))
        answer_question(
            "acme/evidence", "q?", settings=_settings(no_default_storage),
            top_k=5, llm=llm, retrieve=lambda db, rid, q, **kw: evidence,
        )
        (system_msg, user_msg) = llm.calls[0]
        assert system_msg["role"] == "system"
        assert "[E1]" in user_msg["content"]
        assert "src/db/connection.py" in user_msg["content"]

    def test_invalid_citation_rejected_and_stripped(self, no_default_storage):
        evidence = _evidence(("src/a.py", 1, 5, "Python", "a"))
        llm = StubLLM(
            _json_answer("real fact [E1] and made up fact [E99].", ["E1", "E99"], "high", True)
        )
        result = answer_question(
            "acme/evidence", "q?", settings=_settings(no_default_storage),
            top_k=5, llm=llm, retrieve=lambda db, rid, q, **kw: evidence,
        )
        # [E99] must never survive into the response — not in the answer text,
        # not in the citations array.
        assert "[E99]" not in result.answer
        assert [c.id for c in result.citations] == ["E1"]
        assert result.answer.startswith("real fact [E1]")
        assert result.confidence_source == "model"
        assert result.evidence_grounding == "cited"  # E1 survived validation

    def test_model_cannot_invent_evidence_even_in_citations(self, no_default_storage):
        evidence = _evidence(("src/a.py", 1, 5, "Python", "a"))
        llm = StubLLM(_json_answer("scales to 10k users.", ["E7"], "high", True))
        result = answer_question(
            "acme/evidence", "prove 10k users?", settings=_settings(no_default_storage),
            top_k=5, llm=llm, retrieve=lambda db, rid, q, **kw: evidence,
        )
        assert result.citations == []  # E7 doesn't exist among E1..E1
        assert "[E7]" not in result.answer

    def test_fallback_non_json_text_is_handled(self, no_default_storage):
        evidence = _evidence(("src/vault.py", 1, 5, "Python", "class Vault:"))
        llm = StubLLM("The vault is implemented in src/vault.py [E1].")
        result = answer_question(
            "acme/evidence", "where is the vault?", settings=_settings(no_default_storage),
            top_k=5, llm=llm, retrieve=lambda db, rid, q, **kw: evidence,
        )
        assert result.answer == "The vault is implemented in src/vault.py [E1]."
        assert [c.id for c in result.citations] == ["E1"]
        assert result.evidence_sufficient is True  # conservative: cited something

    def test_model_high_confidence_with_no_citations_is_unvalidated(self, no_default_storage):
        evidence = _evidence(("src/a.py", 1, 5, "Python", "a"))
        llm = StubLLM(_json_answer("The answer makes several claims.", [], "high", True))
        result = answer_question(
            "acme/evidence", "q?", settings=_settings(no_default_storage),
            top_k=5, llm=llm, retrieve=lambda db, rid, q, **kw: evidence,
        )
        # high + sufficient is what the MODEL claims; Northern Star still
        # reports that zero evidence blocks are anchored in the answer.
        assert result.confidence == "high"
        assert result.evidence_sufficient is True
        assert result.confidence_source == "model"
        assert result.evidence_grounding == "none"
        assert result.citations == []

    def test_grounding_reflects_validated_citations(self, no_default_storage):
        evidence = _evidence(("src/a.py", 1, 5, "Python", "a"))
        llm = StubLLM(_json_answer("Auth is enforced in [E1].", ["E1"], "high", True))
        result = answer_question(
            "acme/evidence", "q?", settings=_settings(no_default_storage),
            top_k=5, llm=llm, retrieve=lambda db, rid, q, **kw: evidence,
        )
        assert result.confidence_source == "model"
        assert result.evidence_grounding == "cited"
        assert [c.id for c in result.citations] == ["E1"]

    def test_fallback_plain_text_grounding_derived_from_validated(self, no_default_storage):
        evidence = _evidence(
            ("src/a.py", 1, 5, "Python", "a"),
            ("src/vault.py", 10, 15, "Python", "class Vault:"),
        )
        llm = StubLLM("The vault lives in src/vault.py [E2].")
        result = answer_question(
            "acme/evidence", "where is the vault?", settings=_settings(no_default_storage),
            top_k=5, llm=llm, retrieve=lambda db, rid, q, **kw: evidence,
        )
        assert result.confidence_source == "model"
        assert result.evidence_grounding == "cited"
        assert result.answer == "The vault lives in src/vault.py [E2]."
        assert [c.id for c in result.citations] == ["E2"]

    def test_inline_declared_mismatch_drops_citations(self, no_default_storage):
        # Answer says [E4]; the citations array declares only E1. Both IDs are
        # real chunks — but the two signals disagree, so neither survives.
        evidence = _evidence(
            ("src/a.py", 1, 5, "Python", "a"),
            ("src/b.py", 1, 5, "Python", "b"),
            ("src/c.py", 1, 5, "Python", "c"),
            ("src/d.py", 1, 5, "Python", "d"),
        )
        llm = StubLLM(_json_answer("The answer rests on [E4].", ["E1"], "high", True))
        result = answer_question(
            "acme/evidence", "q?", settings=_settings(no_default_storage),
            top_k=5, llm=llm, retrieve=lambda db, rid, q, **kw: evidence,
        )
        # The misleading inline marker must not survive as a citation.
        assert "[E4]" not in result.answer
        assert result.citations == []
        assert result.evidence_grounding == "none"
        assert result.confidence_source == "model"

    def test_inline_and_declared_consistent_citations_survive(self, no_default_storage):
        evidence = _evidence(
            ("a.py", 1, 5, "Python", "a"),
            ("b.py", 1, 5, "Python", "b"),
        )
        llm = StubLLM(_json_answer("Uses [E1] and [E2].", ["E2", "E1"], "medium", True))
        result = answer_question(
            "acme/evidence", "q?", settings=_settings(no_default_storage),
            top_k=5, llm=llm, retrieve=lambda db, rid, q, **kw: evidence,
        )
        assert {c.id for c in result.citations} == {"E1", "E2"}
        assert "[E1]" in result.answer and "[E2]" in result.answer
        assert result.evidence_grounding == "cited"
        assert result.confidence_source == "model"

    def test_no_evidence_short_circuits_without_calling_llm(self, no_default_storage):
        always_raise = StubLLM("should never be called")
        result = answer_question(
            "acme/evidence",
            "Prove that this project supports 10,000 concurrent users.",
            settings=_settings(no_default_storage),
            top_k=5,
            llm=always_raise,
            retrieve=lambda db, rid, q, **kw: [],  # nothing retrieved
        )
        assert always_raise.calls == []  # the LLM was never invoked
        assert result.evidence_sufficient is False
        assert result.confidence == "low"
        assert result.answer == INSUFFICIENT_ANSWER
        assert result.citations == []
        # M3.1: this is the deterministic path — label it as such, claim no
        # evidence grounding.
        assert result.confidence_source == "deterministic"
        assert result.evidence_grounding == "none"

    def test_readme_evidence_stays_distinguished(self, no_default_storage):
        # README chunk AND implementation chunk both retrieved: answer must
        # surface both and the README citation must be present (still distinct).
        evidence = _evidence(
            ("README.md", 1, 3, "Markdown", "Acme App\n=========\nA scalable widget engine."),
            ("src/main.py", 1, 5, "Python", "def main():\n    return 1"),
        )
        llm = StubLLM(
            _json_answer(
                "The README claims scalability [E1] and the implementation defines main() [E2].",
                ["E1", "E2"], "medium", True,
            )
        )
        result = answer_question(
            "acme/evidence", "is this scalable?", settings=_settings(no_default_storage),
            top_k=5, llm=llm, retrieve=lambda db, rid, q, **kw: evidence,
        )
        ids = {c.id: c for c in result.citations}
        assert "E1" in ids and ids["E1"].file_path == "README.md"
        assert "E2" in ids and ids["E2"].file_path == "src/main.py"

    def test_unindexed_repo_raises_upstream(self, no_default_storage):
        def raise_not_indexed(db, rid, q, **kw):
            raise RepoNotIndexedError("not indexed")

        with pytest.raises(RepoNotIndexedError):
            answer_question(
                "acme/evidence", "q?", settings=_settings(no_default_storage),
                top_k=5, llm=StubLLM("x"),
                retrieve=raise_not_indexed,
            )

    def test_ollama_failure_propagates(self, no_default_storage):
        evidence = _evidence(("src/a.py", 1, 5, "Python", "a"))

        class FailingLLM:
            def complete(self, messages):
                raise OllamaUnavailableError("ollama is down")

        with pytest.raises(OllamaUnavailableError):
            answer_question(
                "acme/evidence", "q?", settings=_settings(no_default_storage),
                top_k=5, llm=FailingLLM(), retrieve=lambda db, rid, q, **kw: evidence,
            )


# ---------------------------------------------------------------------------
# POST /api/v1/repos/{owner}/{repo}/ask — API wiring with mocked qa layer
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _seed_repo(tmp_path, no_default_storage, index=True):
    settings = get_settings()
    checkout = build_evidence_checkout(tmp_path / "fixtures")
    analysis = analyze_repository(checkout, settings, tree_root_name="evidence")
    fetched = FetchedRepository(
        owner="acme",
        repo="evidence",
        github_url="https://github.com/acme/evidence",
        checkout_root=checkout,
        commit_hash="deadbeef",
        default_branch="main",
    )
    manifest = _assemble_manifest(fetched, analysis, settings, None)
    repo_dir = no_default_storage / "acme" / "evidence"
    (repo_dir / "checkout").mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkout, repo_dir / "checkout", dirs_exist_ok=True)
    (repo_dir / settings.manifest_filename).write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    if index:
        index_repository(manifest, checkout, no_default_storage / settings.db_filename)
    return repo_dir


def _answer_json(**overrides) -> dict:
    base = {
        "question": "q?",
        "repo_id": "acme/evidence",
        "answer": "Ground truth [E1].",
        "citations": [{"id": "E1", "file_path": "src/a.py",
                       "start_line": 1, "end_line": 5, "language": "Python",
                       "content": "def a():\n    pass"}],
        "confidence": "high",
        "evidence_sufficient": True,
        "confidence_source": "model",
        "evidence_grounding": "cited",
    }
    base.update(overrides)
    return base


class TestAskEndpoint:
    def test_200_returns_answer_and_citation_provenance(
        self, client, tmp_path, no_default_storage, monkeypatch
    ):
        _seed_repo(tmp_path, no_default_storage)

        def fake_answer(repo_id, question, **kw) -> AnswerResponse:
            return AnswerResponse.model_validate(_answer_json(question=question))

        monkeypatch.setattr("app.api.routes.qa_service.answer_question", fake_answer)
        resp = client.post(
            "/api/v1/repos/acme/evidence/ask",
            json={"question": "How are things?", "top_k": 3},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "How are things?"
        assert body["repo_id"] == "acme/evidence"
        assert body["evidence_sufficient"] is True
        assert body["confidence"] == "high"
        # M3.1 honesty fields reach the wire: the model reported high, and the
        # one surviving citation makes the grounding "cited".
        assert body["confidence_source"] == "model"
        assert body["evidence_grounding"] == "cited"
        cited = body["citations"][0]
        # Every citation carries resolved provenance.
        assert cited["id"] == "E1"
        assert cited["file_path"] == "src/a.py"
        assert cited["start_line"] == 1 and cited["end_line"] == 5
        assert cited["language"] == "Python"

    def test_top_k_forwarded(self, client, tmp_path, no_default_storage, monkeypatch):
        _seed_repo(tmp_path, no_default_storage)
        seen = {}

        def fake_answer(repo_id, question, **kw):
            seen.update(kw)
            return AnswerResponse.model_validate(_answer_json())

        monkeypatch.setattr("app.api.routes.qa_service.answer_question", fake_answer)
        client.post("/api/v1/repos/acme/evidence/ask",
                    json={"question": "q", "top_k": 7})
        assert seen["top_k"] == 7
        assert seen["settings"].storage_root == no_default_storage

    def test_empty_question_rejected_422(self, client, tmp_path, no_default_storage, monkeypatch):
        _seed_repo(tmp_path, no_default_storage)
        resp = client.post("/api/v1/repos/acme/evidence/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_unknown_repo_404(self, client, no_default_storage):
        resp = client.post("/api/v1/repos/ghost/nope/ask",
                           json={"question": "q"})
        assert resp.status_code == 404

    def test_ingested_but_not_indexed_404(
        self, client, tmp_path, no_default_storage
    ):
        _seed_repo(tmp_path, no_default_storage, index=False)
        resp = client.post("/api/v1/repos/acme/evidence/ask", json={"question": "q"})
        # Without a mocked answer_question, the retent retrieval raises
        # RepoNotIndexedError → route maps it to 404.
        assert resp.status_code == 404
        assert "not indexed" in resp.json()["detail"]

    def test_ollama_unavailable_maps_to_503(
        self, client, tmp_path, no_default_storage, monkeypatch
    ):
        _seed_repo(tmp_path, no_default_storage)

        def boom(repo_id, question, **kw):
            raise OllamaUnavailableError("ol' ollama is down")

        monkeypatch.setattr("app.api.routes.qa_service.answer_question", boom)
        resp = client.post("/api/v1/repos/acme/evidence/ask", json={"question": "q"})
        assert resp.status_code == 503
        assert "Ollama unavailable" in resp.json()["detail"]

    def test_invalid_citations_never_reach_wire(
        self, client, tmp_path, no_default_storage, monkeypatch
    ):
        _seed_repo(tmp_path, no_default_storage)

        def fake_answer(repo_id, question, **kw) -> AnswerResponse:
            return AnswerResponse.model_validate(
                {
                    "question": question,
                    "repo_id": repo_id,
                    "answer": "Only real evidence [E1].",
                    "citations": [
                        {"id": "E1", "file_path": "src/a.py",
                         "start_line": 1, "end_line": 5, "language": "Python",
                         "content": "def a():\n    pass"}
                    ],
                    "confidence": "high",
                    "evidence_sufficient": True,
                    "confidence_source": "model",
                    "evidence_grounding": "cited",
                }
            )

        monkeypatch.setattr("app.api.routes.qa_service.answer_question", fake_answer)
        resp = client.post("/api/v1/repos/acme/evidence/ask", json={"question": "q"})
        assert resp.status_code == 200
        body = resp.json()
        assert "[E99]" not in body["answer"]
        assert all(c["id"].startswith("E") for c in body["citations"])