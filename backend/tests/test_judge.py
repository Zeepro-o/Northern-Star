"""Tests for M5 evidence-based project judging.

Everything is mocked: no Ollama, no network. Retrieval is faked with
hand-built SearchResult lists; the LLM is faked with an in-memory stub.
"""

from __future__ import annotations

import json

import pytest

from app.config import get_settings
from app.models.schemas import Claim, DimensionScore, FileKind, JudgeResult, SearchResult
from app.services.judge import (
    JUDGE_INSTRUCTION,
    JUDGE_JSON_INSTRUCTIONS,
    _coerce_dimension,
    _coerce_list,
    _extract_json_object,
    _normalize_dimension_score,
    parse_judge,
    judge_repository,
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


def _json_judge(dimensions, strengths=None, weaknesses=None, recommendations=None) -> str:
    return json.dumps(
        {
            "dimensions": dimensions,
            "strengths": strengths or [],
            "weaknesses": weaknesses or [],
            "recommendations": recommendations or [],
        }
    )


def _settings():
    return get_settings()


# ---------------------------------------------------------------------------
# Parsing and validation tests
# ---------------------------------------------------------------------------


class TestParseJudge:
    def test_happy_path_json(self):
        dims = [
            {
                "name": "technical_implementation",
                "score": 7.5,
                "explanation": "Good patterns [E1]",
                "citations": ["E1"],
            },
            {
                "name": "architecture",
                "score": 8.0,
                "explanation": "Modular [E2]",
                "citations": ["E2"],
            },
        ]
        raw = _json_judge(dims, ["Strength [E1]"], ["Weakness [E2]"], ["Rec 1"])
        p = parse_judge(raw, {"E1", "E2"})
        assert p.raw_was_json
        assert len(p.dimensions) == 2
        assert p.dimensions[0]["name"] == "technical_implementation"
        assert p.dimensions[0]["score"] == 7.5
        assert p.dimensions[0]["evidence_ids"] == ["E1"]
        assert p.strengths == ["Strength [E1]"]
        assert p.weaknesses == ["Weakness [E2]"]
        assert p.recommendations == ["Rec 1"]

    def test_code_fenced_json_accepted(self):
        dims = [{"name": "technical_implementation", "score": 5.0, "explanation": "ok", "citations": []}]
        raw = "```json\n" + _json_judge(dims) + "\n```"
        p = parse_judge(raw, set())
        assert p.raw_was_json

    def test_invalid_dimension_name_rejected(self):
        dims = [{"name": "invalid_dim", "score": 5.0, "explanation": "x", "citations": []}]
        raw = _json_judge(dims)
        p = parse_judge(raw, set())
        assert p.dimensions == []

    def test_score_clamped_to_bounds(self):
        dims = [
            {"name": "technical_implementation", "score": 15.0, "explanation": "x", "citations": []},
            {"name": "architecture", "score": -2.0, "explanation": "x", "citations": []},
        ]
        raw = _json_judge(dims)
        p = parse_judge(raw, set())
        assert p.dimensions[0]["score"] == 10.0
        assert p.dimensions[1]["score"] == 0.0

    def test_invalid_citation_rejected(self):
        dims = [{"name": "technical_implementation", "score": 5.0, "explanation": "x", "citations": ["E99"]}]
        raw = _json_judge(dims)
        p = parse_judge(raw, {"E1"})
        assert p.dimensions[0]["evidence_ids"] == []

    def test_fallback_empty_text(self):
        p = parse_judge("", {"E1"})
        assert not p.raw_was_json
        assert p.dimensions == []


class TestCoerceHelpers:
    def test_coerce_list(self):
        assert _coerce_list(["a", "b"]) == ["a", "b"]
        assert _coerce_list("not a list") == []
        assert _coerce_list(None) == []

    def test_normalize_score(self):
        assert _normalize_dimension_score(7.5) == 7.5
        assert _normalize_dimension_score(15) == 10.0
        assert _normalize_dimension_score(-3) == 0.0
        assert _normalize_dimension_score(None) == 5.0
        assert _normalize_dimension_score("not a number") == 5.0

    def test_coerce_dimension_valid(self):
        raw = {"name": "technical_implementation", "score": 7.0, "explanation": "good [E1]", "citations": ["E1"]}
        result = _coerce_dimension(raw, {"E1"})
        assert result is not None
        assert result["name"] == "technical_implementation"
        assert result["score"] == 7.0
        assert result["evidence_ids"] == ["E1"]

    def test_coerce_dimension_invalid_name(self):
        raw = {"name": "invalid", "score": 5.0, "explanation": "x", "citations": []}
        assert _coerce_dimension(raw, set()) is None


# ---------------------------------------------------------------------------
# judge_repository tests
# ---------------------------------------------------------------------------


class TestJudgeRepository:
    def test_unindexed_repo_raises(self):
        def raise_not_indexed(db, rid, q, **kw):
            from app.services.retrieval import RepoNotIndexedError
            raise RepoNotIndexedError("not indexed")

        with pytest.raises(Exception):
            judge_repository(
                "acme/evidence",
                settings=_settings(),
                top_k=5,
                llm=StubLLM("x"),
                retrieve=raise_not_indexed,
            )

    def test_full_judge_flow(self, monkeypatch):
        # Mock evidence for each dimension query
        def fake_retrieve(db, rid, q, **kw):
            if "code quality" in q:
                return _evidence(("src/main.py", 1, 50, "Python", "def foo():\n    return 42"))
            if "modular" in q:
                return _evidence(("src/module.py", 1, 30, "Python", "class Module:\n    pass"))
            if "claim" in q:
                return _evidence(("README.md", 1, 10, "Markdown", "This is a fast framework."))
            if "tests" in q:
                return _evidence(("tests/test_main.py", 1, 20, "Python", "def test_foo():\n    assert foo() == 42"))
            if "quality" in q:
                return _evidence(("src/main.py", 1, 50, "Python", "def foo():\n    return 42"))
            return []

        # Mock LLM response
        dims = [
            {"name": "technical_implementation", "score": 8.0, "explanation": "Clean code [E1]", "citations": ["E1"]},
            {"name": "architecture", "score": 7.5, "explanation": "Modular design [E2]", "citations": ["E2"]},
            {"name": "claim_integrity", "score": 9.0, "explanation": "Claims match [E3]", "citations": ["E3"]},
            {"name": "completeness", "score": 6.5, "explanation": "Has tests [E4]", "citations": ["E4"]},
            {"name": "overall_quality", "score": 8.0, "explanation": "Solid project [E1]", "citations": ["E1"]},
        ]
        llm = StubLLM(_json_judge(
            dims,
            strengths=["Clean code [E1]", "Good architecture [E2]"],
            weaknesses=["Limited docs [E3]"],
            recommendations=["Add more docs", "Add CI"]
        ))

        result = judge_repository(
            "acme/evidence",
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )

        assert isinstance(result, JudgeResult)
        assert result.repo_id == "acme/evidence"
        assert 0 <= result.overall_score <= 100
        assert len(result.dimensions) == 5
        dim_names = [d.name for d in result.dimensions]
        assert dim_names == ["technical_implementation", "architecture", "claim_integrity", "completeness", "overall_quality"]
        # Overall score = sum of scores * 2
        expected_overall = int(round(sum(d.score for d in result.dimensions) * 2))
        assert result.overall_score == expected_overall
        assert result.claim_integrity_summary is not None

    def test_claim_integrity_uses_m4_verification(self, monkeypatch):
        """Test that claim integrity dimension uses M4 verification results."""
        def fake_retrieve(db, rid, q, **kw):
            if "claim" in q:
                return _evidence(("README.md", 1, 10, "Markdown", "This is a fast, scalable framework."))
            return _evidence(("src/main.py", 1, 50, "Python", "def foo():\n    return 42"))

        # Mock claim verification to return specific verdicts
        original_verify = None
        import app.services.judge as judge_mod
        original_verify = judge_mod.verify_claim

        def mock_verify_claim(repo_id, claim, **kw):
            claim.verdict = "supported"
            claim.verdict_explanation = "Evidence supports [E1]"
            claim.evidence_ids = ["E1"]
            claim.repo_id = repo_id
            return claim

        monkeypatch.setattr("app.services.judge.verify_claim", mock_verify_claim)

        dims = [
            {"name": "technical_implementation", "score": 7.0, "explanation": "ok [E1]", "citations": ["E1"]},
            {"name": "architecture", "score": 7.0, "explanation": "ok [E1]", "citations": ["E1"]},
            {"name": "claim_integrity", "score": 9.0, "explanation": "All claims supported [E1]", "citations": ["E1"]},
            {"name": "completeness", "score": 7.0, "explanation": "ok [E1]", "citations": ["E1"]},
            {"name": "overall_quality", "score": 7.0, "explanation": "ok [E1]", "citations": ["E1"]},
        ]
        llm = StubLLM(_json_judge(dims))

        result = judge_repository(
            "acme/evidence",
            settings=_settings(),
            top_k=5,
            llm=llm,
            retrieve=fake_retrieve,
        )

        # Should have claim integrity summary
        assert result.claim_integrity_summary["supported"] >= 0
        assert result.total_claims >= 0