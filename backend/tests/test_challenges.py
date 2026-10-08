"""Tests for M6 challenge generation and red-teaming.

Everything is mocked: no Ollama, no network. Retrieval is faked with
hand-built SearchResult lists; the LLM is faked with an in-memory stub.
"""

from __future__ import annotations

import json

import pytest

from app.config import get_settings
from app.models.schemas import Challenge, ChallengeCategory, ChallengeResult, ChallengeSeverity, Claim, FileKind, SearchResult
from app.services.challenges import (
    CHALLENGE_GENERATION_INSTRUCTION,
    CHALLENGE_JSON_INSTRUCTIONS,
    _coerce_challenges,
    _normalize_category,
    _normalize_confidence,
    _normalize_severity,
    parse_challenges,
    generate_challenges,
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


def _json_challenges(challenges) -> str:
    return json.dumps({"challenges": challenges, "total_challenges": len(challenges)})


def _settings():
    return get_settings()


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalize_severity(self):
        assert _normalize_severity("HIGH") == "high"
        assert _normalize_severity("Medium") == "medium"
        assert _normalize_severity("low") == "low"
        assert _normalize_severity("invalid") == "medium"
        assert _normalize_severity(None) == "medium"

    def test_normalize_category(self):
        assert _normalize_category("UNSUPPORTED_CLAIM") == "unsupported_claim"
        assert _normalize_category("Contradiction") == "contradiction"
        assert _normalize_category("architecture") == "architecture"
        assert _normalize_category("invalid") == "unsupported_claim"
        assert _normalize_category(None) == "unsupported_claim"

    def test_normalize_confidence(self):
        assert _normalize_confidence("HIGH") == "high"
        assert _normalize_confidence("medium") == "medium"
        assert _normalize_confidence("invalid") == "medium"
        assert _normalize_confidence(None) == "medium"


# ---------------------------------------------------------------------------
# Challenge parsing tests
# ---------------------------------------------------------------------------


class TestParseChallenges:
    def test_happy_path_json(self):
        challenges = [
            {
                "id": "challenge_1",
                "claim": "Project claims X",
                "challenge": "Where is X?",
                "severity": "high",
                "category": "contradiction",
                "explanation": "Evidence shows Y [E1]",
                "evidence_ids": ["E1"],
                "confidence": "high",
            }
        ]
        raw = _json_challenges(challenges)
        p = parse_challenges(raw, {"E1"})
        assert p.raw_was_json
        assert len(p.challenges) == 1
        c = p.challenges[0]
        assert c["id"] == "challenge_1"
        assert c["severity"] == "high"
        assert c["category"] == "contradiction"
        assert c["evidence_ids"] == ["E1"]

    def test_code_fenced_json_accepted(self):
        challenges = [{"id": "c1", "claim": "x", "challenge": "y", "severity": "low", "category": "testing", "explanation": "ok", "evidence_ids": [], "confidence": "medium"}]
        raw = "```json\n" + _json_challenges(challenges) + "\n```"
        p = parse_challenges(raw, set())
        assert p.raw_was_json

    def test_invalid_category_rejected(self):
        challenges = [{"id": "c1", "claim": "x", "challenge": "y", "severity": "low", "category": "invalid_cat", "explanation": "ok", "evidence_ids": [], "confidence": "medium"}]
        raw = _json_challenges(challenges)
        p = parse_challenges(raw, set())
        assert p.challenges == []

    def test_invalid_severity_normalized(self):
        challenges = [{"id": "c1", "claim": "x", "challenge": "y", "severity": "INVALID", "category": "testing", "explanation": "ok [E1]", "evidence_ids": ["E1"], "confidence": "medium"}]
        raw = _json_challenges(challenges)
        p = parse_challenges(raw, {"E1"})
        assert p.challenges[0]["severity"] == "medium"

    def test_invalid_citation_rejected(self):
        challenges = [{"id": "c1", "claim": "x", "challenge": "y", "severity": "high", "category": "contradiction", "explanation": "ok [E99]", "evidence_ids": ["E99"], "confidence": "high"}]
        raw = _json_challenges(challenges)
        p = parse_challenges(raw, {"E1"})
        assert p.challenges == []

    def test_challenge_without_valid_citation_dropped(self):
        challenges = [{"id": "c1", "claim": "x", "challenge": "y", "severity": "high", "category": "contradiction", "explanation": "ok", "evidence_ids": [], "confidence": "high"}]
        raw = _json_challenges(challenges)
        p = parse_challenges(raw, {"E1"})
        assert p.challenges == []

    def test_fallback_empty_text(self):
        p = parse_challenges("", {"E1"})
        assert not p.raw_was_json
        assert p.challenges == []


class TestCoerceChallenges:
    def test_coerce_valid_challenges(self):
        raw = [
            {
                "id": "c1",
                "claim": "x",
                "challenge": "y",
                "severity": "high",
                "category": "contradiction",
                "explanation": "ok [E1]",
                "evidence_ids": ["E1"],
                "confidence": "high",
            }
        ]
        result = _coerce_challenges(raw, {"E1"})
        assert len(result) == 1
        assert result[0]["id"] == "c1"
        assert result[0]["evidence_ids"] == ["E1"]

    def test_coerce_invalid_evidence_dropped(self):
        raw = [
            {
                "id": "c1",
                "claim": "x",
                "challenge": "y",
                "severity": "high",
                "category": "contradiction",
                "explanation": "ok",
                "evidence_ids": ["E99"],
                "confidence": "high",
            }
        ]
        result = _coerce_challenges(raw, {"E1"})
        assert result == []

    def test_coerce_missing_fields_dropped(self):
        raw = [
            {"id": "c1", "claim": "x"},  # missing required fields
            {"id": "c2", "challenge": "y"},  # missing claim
        ]
        result = _coerce_challenges(raw, set())
        assert result == []


# ---------------------------------------------------------------------------
# generate_challenges tests
# ---------------------------------------------------------------------------


class TestGenerateChallenges:
    def test_unindexed_repo_raises(self):
        def raise_not_indexed(db, rid, q, **kw):
            from app.services.retrieval import RepoNotIndexedError
            raise RepoNotIndexedError("not indexed")

        with pytest.raises(Exception):
            generate_challenges(
                "acme/evidence",
                settings=_settings(),
                top_k=5,
                llm=StubLLM("x"),
                retrieve=raise_not_indexed,
            )

    def test_full_challenge_flow(self, monkeypatch):
        def fake_retrieve(db, rid, q, **kw):
            # Return evidence for all queries so both E1 and E2 are in valid_ids
            return _evidence(
                ("README.md", 1, 10, "Markdown", "This project is real-time and uses Redis."),
                ("src/main.py", 1, 50, "Python", "class App:\n    def __init__(self):\n        self.redis = Redis()"),
            )

        # Mock LLM response with a valid challenge
        challenge_data = [
            {
                "id": "challenge_1",
                "claim": "This project is real-time and uses Redis.",
                "challenge": "The README claims real-time and Redis, but where is the real-time implementation?",
                "severity": "high",
                "category": "unsupported_claim",
                "explanation": "README mentions real-time [E1] but implementation shows Redis [E2].",
                "evidence_ids": ["E1", "E2"],
                "confidence": "high",
            }
        ]
        llm = StubLLM(_json_challenges(challenge_data))

        result = generate_challenges(
            "acme/evidence",
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )

        assert isinstance(result, ChallengeResult)
        assert result.repo_id == "acme/evidence"
        assert result.total_challenges == 1
        assert result.high_severity == 1
        c = result.challenges[0]
        assert c.id == "challenge_1"
        assert c.severity == "high"
        assert c.category == "unsupported_claim"
        assert c.evidence_ids == ["E1", "E2"]
        assert len(result.evidence_citations) == 2

    def test_claim_integrity_informs_challenges(self, monkeypatch):
        """Test that contradicted claims produce high-severity challenges."""
        def fake_retrieve(db, rid, q, **kw):
            if "claim" in q:
                return _evidence(("README.md", 1, 10, "Markdown", "Uses PostgreSQL"))
            return _evidence(("src/db.py", 1, 10, "Python", "import sqlite3"))

        challenge_data = [
            {
                "id": "challenge_1",
                "claim": "Uses PostgreSQL",
                "challenge": "README claims PostgreSQL but implementation uses SQLite",
                "severity": "high",
                "category": "contradiction",
                "explanation": "README claims PostgreSQL [E1] but code uses sqlite3 [E2].",
                "evidence_ids": ["E1", "E2"],
                "confidence": "high",
            }
        ]
        llm = StubLLM(_json_challenges(challenge_data))

        result = generate_challenges(
            "acme/evidence",
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )

        assert result.total_challenges == 1
        assert result.high_severity == 1
        assert result.challenges[0].category == "contradiction"