"""Tests for M4 claim extraction and evidence verification.

Everything is mocked: no Ollama, no network. Retrieval is faked with
hand-built SearchResult lists; the LLM is faked with an in-memory stub.
"""

from __future__ import annotations

import json

import pytest

from app.config import get_settings
from app.models.schemas import Claim, FileKind, SearchResult
from app.services.detection import extract_structured_claims
from app.services.claims import (
    CLAIM_VERIFICATION_INSTRUCTION,
    CLAIM_VERIFICATION_JSON_INSTRUCTIONS,
    UNSUFFICIENT_VERIFICATION,
    _inline_citation_ids,
    parse_verification,
    verify_claim,
)


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


def _json_verdict(verdict: str, explanation: str, citations, raw_was_json=True) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "explanation": explanation,
            "citations": citations,
        }
    )


def _settings():
    return get_settings()


# ---------------------------------------------------------------------------
# Claim extraction tests
# ---------------------------------------------------------------------------


class TestExtractStructuredClaims:
    def test_empty_text_returns_empty_list(self):
        claims = extract_structured_claims("")
        assert claims == []

    def test_none_text_returns_empty_list(self):
        claims = extract_structured_claims(None)
        assert claims == []

    def test_readme_with_keywords_returns_structured_claims(self):
        text = (
            "Acme Widget\n"
            "A real-time, scalable widget engine with machine learning support.\n"
        )
        claims = extract_structured_claims(text, source="README.md")
        # Now extracts complete propositions (sentences), not keyword fragments
        # The text produces 1 proposition: "A real-time, scalable widget engine with machine learning support."
        assert len(claims) >= 1
        # Check structure
        for c in claims:
            assert isinstance(c, Claim)
            assert c.id
            assert c.text
            assert c.source == "README.md"
            assert c.kind == FileKind.DOCUMENTATION
            assert c.category
            assert c.verdict == "unclear"
            assert c.verdict_explanation == "Awaiting evidence verification."
            assert c.evidence_ids == []

    def test_claim_categories_mapped_correctly(self):
        text = "This is a real-time, scalable machine learning system."
        claims = extract_structured_claims(text)
        # Should extract the proposition with categories
        assert len(claims) >= 1
        # The combined proposition should have one of the relevant categories
        cats = {c.category for c in claims}
        assert "performance" in cats or "ai-capability" in cats


# ---------------------------------------------------------------------------
# parse_verification tests
# ---------------------------------------------------------------------------


class TestParseVerification:
    def test_happy_path_json(self):
        raw = _json_verdict(
            "supported", "The evidence shows [E1] supports the claim.", ["E1"]
        )
        p = parse_verification(raw, {"E1"})
        assert p.raw_was_json
        assert p.verdict == "supported"
        assert p.explanation == "The evidence shows [E1] supports the claim."
        assert p.citations == ["E1"]

    def test_code_fenced_json_accepted(self):
        raw = "```json\n" + _json_verdict("unclear", "No evidence.", []) + "\n```"
        p = parse_verification(raw, set())
        assert p.raw_was_json
        assert p.verdict == "unclear"

    def test_trailing_prose_after_json_ignored(self):
        raw = _json_verdict("supported", "ok [E1].", ["E1"]) + "\nThanks!"
        p = parse_verification(raw, {"E1"})
        assert p.verdict == "supported"

    def test_fallback_plain_text_mines_citations(self):
        p = parse_verification("The claim is true [E2].", {"E2"})
        assert not p.raw_was_json
        assert p.verdict == "unclear"  # no verdict keyword in text
        assert p.citations == ["E2"]

    def test_fallback_with_verdict_keyword(self):
        p = parse_verification("This is contradicted [E1].", {"E1"})
        assert p.verdict == "contradicted"

    def test_verdict_case_normalized(self):
        p = parse_verification(_json_verdict("SUPPORTED", "x", []), set())
        assert p.verdict == "supported"


# ---------------------------------------------------------------------------
# _inline_citation_ids tests
# ---------------------------------------------------------------------------


class TestInlineCitationIds:
    def test_finds_valid_markers(self):
        assert _inline_citation_ids("Uses [E1] and [E2].", {"E1", "E2", "E3"}) == ["E1", "E2"]

    def test_deduplicates(self):
        assert _inline_citation_ids("[E1] and [E1] again.", {"E1"}) == ["E1"]

    def test_ignores_invalid_ids(self):
        assert _inline_citation_ids("Uses [E99].", {"E1"}) == []

    def test_empty_text(self):
        assert _inline_citation_ids("", {"E1"}) == []


# ---------------------------------------------------------------------------
# verify_claim tests
# ---------------------------------------------------------------------------


class TestVerifyClaim:
    def test_no_evidence_returns_unclear(self, monkeypatch):
        """No evidence → unclear, NEVER contradicted."""
        claim = Claim(
            id="claim_1",
            text="This system uses Redis",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="architecture",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )

        llm = StubLLM("should not be called")

        def empty_retrieve(db, rid, q, **kw):
            return []

        result = verify_claim(
            "acme/evidence",
            claim,
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=empty_retrieve,
        )
        assert llm.calls == []  # LLM never called
        assert result.verdict == "unclear"
        assert result.verdict_explanation == UNSUFFICIENT_VERIFICATION
        assert result.evidence_ids == []

    def test_supported_claim_with_valid_citations(self, monkeypatch):
        evidence = _evidence(
            ("src/cache.py", 1, 10, "Python", "import redis\ncache = redis.Redis()"),
        )
        claim = Claim(
            id="claim_1",
            text="This system uses Redis",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="architecture",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )

        llm = StubLLM(
            _json_verdict(
                "supported",
                "The code imports redis and creates a Redis client [E1].",
                ["E1"],
            )
        )

        def fake_retrieve(db, rid, q, **kw):
            return evidence

        result = verify_claim(
            "acme/evidence",
            claim,
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )
        assert result.verdict == "supported"
        assert "[E1]" in result.verdict_explanation
        assert result.evidence_ids == ["E1"]
        assert result.repo_id == "acme/evidence"

    def test_invalid_citation_rejected_and_stripped(self, monkeypatch):
        evidence = _evidence(
            ("src/a.py", 1, 5, "Python", "def a(): pass"),
        )
        claim = Claim(
            id="claim_1",
            text="Uses Redis",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="architecture",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )

        # LLM cites E99 which doesn't exist
        llm = StubLLM(
            _json_verdict(
                "supported",
                "Real code [E1] and fake [E99].",
                ["E1", "E99"],
            )
        )

        def fake_retrieve(db, rid, q, **kw):
            return evidence

        result = verify_claim(
            "acme/evidence",
            claim,
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )
        assert "[E99]" not in result.verdict_explanation
        assert result.evidence_ids == ["E1"]
        assert result.verdict == "supported"

    def test_model_cannot_invent_evidence_even_in_citations(self, monkeypatch):
        evidence = _evidence(("src/a.py", 1, 5, "Python", "def a(): pass"))
        claim = Claim(
            id="claim_1",
            text="Uses Redis",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="architecture",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )

        llm = StubLLM(_json_verdict("supported", "Uses Redis.", ["E7"]))

        def fake_retrieve(db, rid, q, **kw):
            return evidence

        result = verify_claim(
            "acme/evidence",
            claim,
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )
        assert result.evidence_ids == []
        assert result.verdict == "unclear"  # forced to unclear because no valid citations

    def test_inline_declared_mismatch_drops_citations(self, monkeypatch):
        evidence = _evidence(
            ("src/a.py", 1, 5, "Python", "a"),
            ("src/b.py", 1, 5, "Python", "b"),
            ("src/c.py", 1, 5, "Python", "c"),
            ("src/d.py", 1, 5, "Python", "d"),
        )
        claim = Claim(
            id="claim_1",
            text="Claim",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="general",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )

        # Inline says [E4]; declared array says E1. Both real but disagree.
        llm = StubLLM(_json_verdict("supported", "Relies on [E4].", ["E1"]))

        def fake_retrieve(db, rid, q, **kw):
            return evidence

        result = verify_claim(
            "acme/evidence",
            claim,
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )
        assert "[E4]" not in result.verdict_explanation
        assert result.evidence_ids == []
        assert result.verdict == "unclear"

    def test_inline_and_declared_consistent_citations_survive(self, monkeypatch):
        evidence = _evidence(
            ("src/a.py", 1, 5, "Python", "a"),
            ("src/b.py", 1, 5, "Python", "b"),
        )
        claim = Claim(
            id="claim_1",
            text="Claim",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="general",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )

        llm = StubLLM(_json_verdict("partially_supported", "Uses [E1] and [E2].", ["E2", "E1"]))

        def fake_retrieve(db, rid, q, **kw):
            return evidence

        result = verify_claim(
            "acme/evidence",
            claim,
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )
        assert set(result.evidence_ids) == {"E1", "E2"}
        assert "[E1]" in result.verdict_explanation and "[E2]" in result.verdict_explanation
        assert result.verdict == "partially_supported"

    def test_contradicted_requires_positive_evidence(self, monkeypatch):
        # The LLM says contradicted but provides no valid citations
        # Should be forced to unclear
        evidence = _evidence(("src/a.py", 1, 5, "Python", "def a(): pass"))
        claim = Claim(
            id="claim_1",
            text="Uses Redis",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="architecture",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )

        llm = StubLLM(_json_verdict("contradicted", "Actually uses MongoDB.", []))

        def fake_retrieve(db, rid, q, **kw):
            return evidence

        result = verify_claim(
            "acme/evidence",
            claim,
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )
        assert result.verdict == "unclear"  # forced because no citations

    def test_contradicted_with_valid_evidence(self, monkeypatch):
        # Evidence that directly conflicts with the claim
        evidence = _evidence(
            ("src/config.py", 1, 10, "Python", "DATABASE = 'mongodb'\n# We use MongoDB, not Redis"),
        )
        claim = Claim(
            id="claim_1",
            text="This system uses Redis",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="architecture",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )

        llm = StubLLM(
            _json_verdict(
                "contradicted",
                "The config explicitly states MongoDB is used, not Redis [E1].",
                ["E1"],
            )
        )

        def fake_retrieve(db, rid, q, **kw):
            return evidence

        result = verify_claim(
            "acme/evidence",
            claim,
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )
        assert result.verdict == "contradicted"
        assert "[E1]" in result.verdict_explanation
        assert result.evidence_ids == ["E1"]

    def test_prompt_passed_to_llm_contains_claim_and_evidence(self, monkeypatch):
        evidence = _evidence(("src/a.py", 1, 5, "Python", "content x"))
        claim = Claim(
            id="claim_1",
            text="Test claim",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="general",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )
        llm = StubLLM(_json_verdict("unclear", "ok", []))

        def fake_retrieve(db, rid, q, **kw):
            return evidence

        verify_claim(
            "acme/evidence",
            claim,
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )
        (system_msg, user_msg) = llm.calls[0]
        assert system_msg["role"] == "system"
        assert "CLAIM: Test claim" in user_msg["content"]
        assert "SOURCE: README.md" in user_msg["content"]
        assert "[E1]" in user_msg["content"]
        assert "src/a.py" in user_msg["content"]

    def test_ollama_failure_propagates(self, monkeypatch):
        from app.services.llm import OllamaUnavailableError

        evidence = _evidence(("src/a.py", 1, 5, "Python", "a"))
        claim = Claim(
            id="claim_1",
            text="Claim",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="general",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )

        class FailingLLM:
            def complete(self, messages):
                raise OllamaUnavailableError("ollama is down")

        def fake_retrieve(db, rid, q, **kw):
            return evidence

        with pytest.raises(OllamaUnavailableError):
            verify_claim(
                "acme/evidence",
                claim,
                settings=_settings(),
                top_k=5,
                llm=FailingLLM(),
                retrieve=fake_retrieve,
            )

    def test_fallback_non_json_text_is_handled(self, monkeypatch):
        evidence = _evidence(("src/a.py", 1, 5, "Python", "def a(): pass"))
        claim = Claim(
            id="claim_1",
            text="Test claim",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="general",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )
        llm = StubLLM("The evidence shows [E1] implements it.")

        def fake_retrieve(db, rid, q, **kw):
            return evidence

        result = verify_claim(
            "acme/evidence",
            claim,
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )
        assert result.verdict == "unclear"  # no verdict keyword in fallback text
        assert result.evidence_ids == ["E1"]

    def test_unindexed_repo_raises(self, monkeypatch):
        from app.services.retrieval import RepoNotIndexedError

        claim = Claim(
            id="claim_1",
            text="Test",
            source="README.md",
            kind=FileKind.DOCUMENTATION,
            category="general",
            verdict="unclear",
            verdict_explanation="",
            evidence_ids=[],
            repo_id="acme/evidence",
        )

        def raise_not_indexed(db, rid, q, **kw):
            raise RepoNotIndexedError("not indexed")

        with pytest.raises(RepoNotIndexedError):
            verify_claim(
                "acme/evidence",
                claim,
                settings=_settings(),
                top_k=5,
                llm=StubLLM("x"),
                retrieve=raise_not_indexed,
            )