"""Tests for M7 evidence-based improvement engine.

Everything is mocked: no Ollama, no network. Judge/challenges outputs are
faked, retrieval returns hand-built SearchResults, LLM is an in-memory stub.
"""

from __future__ import annotations

import json

import pytest

from app.config import get_settings
from app.models.schemas import (
    Challenge,
    ChallengeResult,
    Citation,
    DimensionScore,
    ImprovementResult,
    JudgeResult,
    SearchResult,
)
from app.services.improvements import (
    _coerce_improvements,
    _normalize_category,
    _normalize_confidence,
    _normalize_priority,
    deterministic_priority,
    generate_improvements,
    parse_improvements,
)


def _settings():
    return get_settings()


def _evidence(*specs) -> list[SearchResult]:
    out = []
    for path, start, end, lang, content in specs:
        out.append(SearchResult(
            file_path=path, start_line=start, end_line=end,
            language=lang, content=content,
        ))
    return out


class StubLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list = []

    def complete(self, messages) -> str:
        self.calls.append(messages)
        return self.content


def _json_improvements(items) -> str:
    return json.dumps({"improvements": items, "total_improvements": len(items)})


def _make_challenge(cid="challenge_1", severity="high", category="contradiction") -> Challenge:
    return Challenge(
        id=cid, claim="Project claims X", challenge="Where is X?",
        severity=severity, category=category,
        explanation="Evidence shows Y [E1]", evidence_ids=["E1"],
        repo_id="acme/evidence", confidence="high",
    )


def _make_judge(weak_score: float = 3.0) -> JudgeResult:
    dims = [
        DimensionScore(name="technical_implementation", score=7.0, explanation="ok [E1]", evidence_ids=["E1"]),
        DimensionScore(name="architecture", score=7.0, explanation="ok", evidence_ids=[]),
        DimensionScore(name="claim_integrity", score=5.0, explanation="mixed", evidence_ids=[]),
        DimensionScore(name="completeness", score=weak_score, explanation="tests missing", evidence_ids=[]),
        DimensionScore(name="overall_quality", score=6.0, explanation="ok", evidence_ids=[]),
    ]
    return JudgeResult(
        repo_id="acme/evidence", overall_score=60, dimensions=dims,
        strengths=["good code"], weaknesses=["tests missing"],
        recommendations=["add tests"],
        claim_integrity_summary={"supported": 1, "partially_supported": 0, "unclear": 1, "contradicted": 1},
        total_claims=3, evidence_citations=[],
    )


def _make_challenge_result(*challenges) -> ChallengeResult:
    chs = list(challenges) or [_make_challenge()]
    high = sum(1 for c in chs if str(c.severity) == "high")
    med = sum(1 for c in chs if str(c.severity) == "medium")
    low = sum(1 for c in chs if str(c.severity) == "low")
    return ChallengeResult(
        repo_id="acme/evidence", challenges=chs, total_challenges=len(chs),
        high_severity=high, medium_severity=med, low_severity=low,
        evidence_citations=[],
    )


# ---------------------------------------------------------------------------
# Normalization + deterministic priority
# ---------------------------------------------------------------------------

class TestPriority:
    def test_normalize_priority(self):
        assert _normalize_priority("CRITICAL") == "critical"
        assert _normalize_priority("high") == "high"
        assert _normalize_priority("bogus") == "medium"
        assert _normalize_priority(None) == "medium"

    def test_normalize_category(self):
        assert _normalize_category("TESTING") == "testing"
        assert _normalize_category("nope") == "unsupported_claim"

    def test_normalize_confidence(self):
        assert _normalize_confidence("HIGH") == "high"
        assert _normalize_confidence("xx") == "medium"

    def test_deterministic_priority_rules(self):
        assert deterministic_priority("high", "contradiction") == "critical"
        assert deterministic_priority("high", "testing") == "high"
        assert deterministic_priority("medium", "testing") == "medium"
        assert deterministic_priority("low", "testing") == "low"
        assert deterministic_priority(None, None) == "medium"
        # LLM cannot sneak critical through unknown severity
        assert deterministic_priority("unknown", "contradiction") == "medium"


# ---------------------------------------------------------------------------
# Parsing / coercion
# ---------------------------------------------------------------------------

class TestParse:
    def test_happy_path(self):
        items = [{
            "id": "improvement_1", "title": "Add tests", "problem": "No tests [E1]",
            "recommendation": "Add pytest suite [E1]", "priority": "high",
            "category": "testing", "rationale": "Weak tests [E1]",
            "evidence_ids": ["E1"], "related_challenge_ids": ["challenge_1"],
            "affected_files": ["tests/test_x.py"], "confidence": "high",
        }]
        p = parse_improvements(
            _json_improvements(items), {"E1"}, {"challenge_1"}, {"tests/test_x.py"})
        assert p.raw_was_json
        assert len(p.improvements) == 1

    def test_no_evidence_dropped(self):
        items = [{
            "id": "improvement_1", "title": "Add tests", "problem": "x",
            "recommendation": "y", "priority": "high", "category": "testing",
            "rationale": "z", "evidence_ids": ["E99"],
            "related_challenge_ids": [], "affected_files": [], "confidence": "high",
        }]
        p = parse_improvements(_json_improvements(items), {"E1"}, set(), None)
        assert p.improvements == []

    def test_affected_files_filtered(self):
        items = [{
            "id": "improvement_1", "title": "T", "problem": "P [E1]",
            "recommendation": "R [E1]", "priority": "medium", "category": "testing",
            "rationale": "W [E1]", "evidence_ids": ["E1"],
            "related_challenge_ids": [], "affected_files": ["real.py", "invented_xyz.py"],
            "confidence": "medium",
        }]
        out = _coerce_improvements(items, {"E1"}, set(), {"real.py"})
        assert out[0]["affected_files"] == ["real.py"]

    def test_related_challenge_ids_filtered(self):
        items = [{
            "id": "improvement_1", "title": "T", "problem": "P [E1]",
            "recommendation": "R [E1]", "priority": "medium", "category": "testing",
            "rationale": "W [E1]", "evidence_ids": ["E1"],
            "related_challenge_ids": ["challenge_1", "challenge_zzz"],
            "affected_files": [], "confidence": "medium",
        }]
        out = _coerce_improvements(items, {"E1"}, {"challenge_1"}, None)
        assert out[0]["related_challenge_ids"] == ["challenge_1"]

    def test_non_json_returns_empty(self):
        p = parse_improvements("not json at all", {"E1"}, set(), None)
        assert not p.raw_was_json
        assert p.improvements == []


# ---------------------------------------------------------------------------
# generate_improvements
# ---------------------------------------------------------------------------

class TestGenerate:
    def _fakes(self, llm_payload: str, ev):
        def fake_retrieve(db, rid, q, **kw):
            return ev
        judge = _make_judge()
        ch_result = _make_challenge_result(_make_challenge())
        def fake_judge(**kw):
            return judge
        def fake_challenges(**kw):
            return ch_result
        llm = StubLLM(llm_payload)
        return fake_retrieve, fake_judge, fake_challenges, llm

    def test_challenge_to_improvement_mapping(self):
        items = [{
            "id": "improvement_1", "title": "Fix contradiction", "problem": "Claims X [E1]",
            "recommendation": "Align docs with code [E1]", "priority": "low",
            "category": "testing", "rationale": "Matters [E1]",
            "evidence_ids": ["E1"], "related_challenge_ids": ["challenge_1"],
            "affected_files": ["src/main.py"], "confidence": "high",
        }]
        ev = _evidence(("src/main.py", 1, 10, "Python", "real implementation content here"))
        fr, fj, fc, llm = self._fakes(_json_improvements(items), ev)
        result = generate_improvements(
            "acme/evidence", settings=_settings(), top_k=5, llm=llm,
            retrieve=fr, judge_fn=fj, challenges_fn=fc)
        assert isinstance(result, ImprovementResult)
        assert result.total_improvements == 1
        imp = result.improvements[0]
        # Deterministic: high+contradiction challenge → critical (LLM said low)
        assert imp.priority == "critical"
        assert imp.category == "contradiction"
        assert imp.related_challenge_ids == ["challenge_1"]
        assert imp.evidence_ids == ["E1"]

    def test_critical_clamped_without_challenge(self):
        items = [{
            "id": "improvement_1", "title": "Add tests", "problem": "Weak tests [E1]",
            "recommendation": "Add pytest [E1]", "priority": "critical",
            "category": "testing", "rationale": "Completeness low [E1]",
            "evidence_ids": ["E1"], "related_challenge_ids": [],
            "affected_files": [], "confidence": "medium",
        }]
        ev = _evidence(("tests/test_x.py", 1, 5, "Python", "placeholder test content"))
        fr, fj, fc, llm = self._fakes(_json_improvements(items), ev)
        # Empty challenges → judge-driven; critical must be demoted to high
        fj_judge = fj
        def no_challenges(**kw):
            return _make_challenge_result()
        # _make_challenge_result with no args still makes one; build truly empty:
        empty = ChallengeResult(repo_id="acme/evidence", challenges=[], total_challenges=0,
                                high_severity=0, medium_severity=0, low_severity=0, evidence_citations=[])
        def fc_empty(**kw):
            return empty
        result = generate_improvements(
            "acme/evidence", settings=_settings(), top_k=5, llm=llm,
            retrieve=fr, judge_fn=fj_judge, challenges_fn=fc_empty)
        assert result.improvements[0].priority == "high"

    def test_no_evidence_returns_empty(self):
        items = [{
            "id": "improvement_1", "title": "T", "problem": "P", "recommendation": "R",
            "priority": "high", "category": "testing", "rationale": "W",
            "evidence_ids": ["E1"], "related_challenge_ids": [],
            "affected_files": [], "confidence": "high",
        }]
        fr, fj, fc, llm = self._fakes(_json_improvements(items), [])
        result = generate_improvements(
            "acme/evidence", settings=_settings(), top_k=5, llm=llm,
            retrieve=fr, judge_fn=fj, challenges_fn=fc)
        assert result.total_improvements == 0
        assert result.improvements == []

    def test_unindexed_repo_raises(self):
        def raise_not_indexed(db, rid, q, **kw):
            from app.services.retrieval import RepoNotIndexedError
            raise RepoNotIndexedError("not indexed")
        def fake_judge(**kw):
            from app.services.retrieval import RepoNotIndexedError
            raise RepoNotIndexedError("not indexed")
        def fake_ch(**kw):
            raise AssertionError("should not reach challenges")
        with pytest.raises(Exception):
            generate_improvements(
                "acme/evidence", settings=_settings(), top_k=5,
                llm=StubLLM("x"), retrieve=raise_not_indexed,
                judge_fn=fake_judge, challenges_fn=fake_ch)

class TestApi:
    def test_improvements_endpoint_returns_result(self, monkeypatch):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.models.schemas import Improvement, ImprovementResult

        result = ImprovementResult(
            repo_id="acme/widget", improvements=[
                Improvement(id="improvement_1", title="Add tests", problem="P [E1]",
                            recommendation="R [E1]", priority="high", category="testing",
                            rationale="W [E1]", evidence_ids=["E1"],
                            related_challenge_ids=["challenge_1"], affected_files=["a.py"],
                            confidence="high"),
            ],
            total_improvements=1, critical_count=0, high_count=1,
            medium_count=0, low_count=0, evidence_citations=[],
        )
        monkeypatch.setattr(
            "app.services.improvements.generate_improvements", lambda **kw: result)
        # Ingested repo guard: create a fake manifest on the real storage path.
        from app.config import get_settings
        settings = get_settings()
        repo_dir = settings.storage_root / "acme" / "widget"
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / settings.manifest_filename).write_text('{"id": "acme/widget"}')

        client = TestClient(app)
        resp = client.get("/api/v1/repos/acme/widget/improvements")
        assert resp.status_code == 200
        body = resp.json()
        assert body["repo_id"] == "acme/widget"
        assert body["total_improvements"] == 1
        assert body["improvements"][0]["priority"] == "high"

    def test_improvements_unknown_repo_404(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/api/v1/repos/acme/does-not-exist-xyz/improvements")
        assert resp.status_code == 404


class TestGenerateJson:
    def test_json_serialization(self):
        items = [{
            "id": "improvement_1", "title": "T", "problem": "P [E1]",
            "recommendation": "R [E1]", "priority": "medium", "category": "testing",
            "rationale": "W [E1]", "evidence_ids": ["E1"],
            "related_challenge_ids": ["challenge_1"], "affected_files": [],
            "confidence": "low",
        }]
        ev = _evidence(("src/a.py", 1, 3, "Python", "some code content"))
        ch = _make_challenge(severity="medium", category="testing")

        def fake_retrieve(db, rid, q, **kw):
            return ev

        def fake_judge(**kw):
            return _make_judge()

        def fake_ch(**kw):
            return _make_challenge_result(ch)

        llm = StubLLM(_json_improvements(items))
        result = generate_improvements(
            "acme/evidence", settings=_settings(), top_k=5, llm=llm,
            retrieve=fake_retrieve, judge_fn=fake_judge, challenges_fn=fake_ch)
        dumped = result.model_dump(mode="json")
        assert dumped["total_improvements"] == 1
        assert dumped["medium_count"] == 1
        assert dumped["improvements"][0]["priority"] == "medium"


class TestCli:
    def test_cli_parser_has_improvements(self):
        from app.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["improvements", "acme/widget", "--json"])
        assert args.command == "improvements"
        assert args.repo == "acme/widget"
        assert args.json is True
