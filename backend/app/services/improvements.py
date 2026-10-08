"""M7 — Evidence-Based Improvement Engine.

Architecture:
  M4 Claims
    → M4 Verification
    → M5 Judge
    → M6 Challenges
    → M7 Improvements

Core principle: "LLMs reason; evidence determines what can be claimed."

M6 asks: "What is wrong / what should we challenge?"
M7 asks: "What specifically should the team do next?"

Every improvement must:
  - derive from an M6 challenge and/or M5 weakness (no generic advice)
  - cite repository evidence by [E#] ID
  - reference only files that actually exist in the indexed repository
  - carry a deterministically-computed priority (LLM may not assign
    critical without a high-severity contradiction behind it)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import Settings
from ..models.schemas import (
    Challenge,
    Citation,
    Improvement,
    ImprovementResult,
)
from .indexing import evidence_db_path
from .llm import OllamaClient
from .prompts import build_qa_messages
from .qa import label_evidence, validate_citations
from .retrieval import RepoNotIndexedError, search_evidence


# ---------------------------------------------------------------------------
# System prompt for improvement generation
# ---------------------------------------------------------------------------

IMPROVEMENT_GENERATION_INSTRUCTION = (
    "You are an evidence-grounded software improvement advisor for the repository "
    "shown in the evidence blocks below. Your job is to turn the listed CHALLENGES "
    "(from red-team analysis) and JUDGE FINDINGS (dimension scores, weaknesses) into "
    "SPECIFIC, ACTIONABLE, EVIDENCE-BACKED improvement recommendations.\n"
    "\n"
    "Improvement rules (non-negotiable):\n"
    "1. Generate improvements ONLY from the challenges, judge findings, and evidence "
    "blocks supplied below. Never invent, guess, or recall facts not present.\n"
    "2. Every improvement MUST cite evidence blocks by their [E#] IDs in its problem, "
    "recommendation, and rationale. Do not cite an evidence block for a point it does "
    "not actually support.\n"
    "3. Every improvement MUST reference the challenge ID(s) it addresses in "
    "related_challenge_ids (when derived from a challenge), or address a listed "
    "judge weakness explicitly.\n"
    "4. Do NOT generate generic advice (e.g. 'add Kubernetes', 'improve scalability') "
    "unless the evidence and challenges actually indicate that specific issue.\n"
    "5. Recommendations must be actionable, repository-specific, and technically "
    "plausible for THIS repository (name concrete modules, tests, docs to add).\n"
    "6. affected_files must list ONLY files shown in the evidence blocks or the "
    "repository file inventory. Never invent file paths.\n"
    "7. Cite only evidence IDs that appear in the supplied blocks.\n"
    "8. If the evidence does not support a concrete improvement, do not emit it.\n"
    "\n"
    "Prioritize (highest first):\n"
    "1. high-severity M6 challenges\n"
    "2. contradicted M4 claims (category=contradiction)\n"
    "3. partially_supported claims\n"
    "4. weak M5 dimensions (lowest scores first)\n"
    "5. medium-severity challenges\n"
    "6. low-severity improvements\n"
    "\n"
    "You must respond with ONLY a single JSON object, no prose around it, in "
    "this exact shape:\n"
    '{"improvements": [\n'
    '  {\n'
    '    "id": "improvement_1",\n'
    '    "title": "Add auth middleware to protected routes",\n'
    '    "problem": "Auth is claimed but no middleware is evident [E1]",\n'
    '    "recommendation": "Add middleware in app/auth.py and cover with tests [E1] [E2]",\n'
    '    "priority": "critical|high|medium|low",\n'
    '    "category": "unsupported_claim|contradiction|missing_implementation|architecture|security|reliability|scalability|testing|completeness|documentation",\n'
    '    "rationale": "Why this matters, grounded in evidence [E1]",\n'
    '    "evidence_ids": ["E1", "E2"],\n'
    '    "related_challenge_ids": ["challenge_1"],\n'
    '    "affected_files": ["app/routes.py", "app/auth.py"],\n'
    '    "confidence": "high|medium|low"\n'
    '  }\n'
    '],\n'
    '"total_improvements": 1\n'
    '}\n'
    "Only include improvements with at least one valid evidence citation.\n"
    "If no valid improvements can be generated, return an empty improvements array.\n"
)

IMPROVEMENT_JSON_INSTRUCTIONS = (
    "Respond with only a single JSON object with exactly these fields:\n"
    '{"improvements": [{"id": "improvement_1", "title": "...", "problem": "... [E1]", '
    '"recommendation": "... [E1]", "priority": "critical|high|medium|low", '
    '"category": "unsupported_claim|contradiction|...", "rationale": "... [E1]", '
    '"evidence_ids": ["E1"], "related_challenge_ids": ["challenge_1"], '
    '"affected_files": ["app/routes.py"], "confidence": "high|medium|low"}], '
    '"total_improvements": 1}\n'
    "Cite evidence by ID like [E1]. "
    "Only include improvements with at least one valid evidence citation.\n"
    "If no valid improvements can be generated, return an empty improvements array."
)


# ---------------------------------------------------------------------------
# Internal parsed response
# ---------------------------------------------------------------------------

@dataclass
class ParsedImprovements:
    improvements: list[dict]
    raw_was_json: bool


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------

_IMPROVEMENT_CATEGORIES = {
    "unsupported_claim", "contradiction", "missing_implementation",
    "architecture", "security", "reliability", "scalability",
    "testing", "completeness", "documentation",
}

_IMPROVEMENT_PRIORITIES = {"critical", "high", "medium", "low"}
_CONFIDENCE_LEVELS = {"high", "medium", "low"}


def _extract_json_object(text: str) -> Optional[dict]:
    """Find and parse the first balanced {...} JSON object in text."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\s*\w*\n?", "", s, flags=re.MULTILINE)
        s = re.sub(r"```\s*$", "", s.strip(), flags=re.MULTILINE)
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start: i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_priority(value: object | None, fallback: str = "medium") -> str:
    if isinstance(value, str):
        v = value.lower().strip()
        if v in _IMPROVEMENT_PRIORITIES:
            return v
    return fallback


def _normalize_category(value: object | None, fallback: str = "unsupported_claim") -> str:
    if isinstance(value, str):
        v = value.lower().strip()
        if v in _IMPROVEMENT_CATEGORIES:
            return v
    return fallback


def _normalize_confidence(value: object | None, fallback: str = "medium") -> str:
    if isinstance(value, str):
        v = value.lower().strip()
        if v in _CONFIDENCE_LEVELS:
            return v
    return fallback


def _coerce_str_list(raw: object | None) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [s for s in raw if isinstance(s, str) and s.strip()]


def deterministic_priority(
    severity: str | None,
    category: str | None,
) -> str:
    """Deterministic priority from source challenge severity + category.

    Rules (per M7 spec):
      contradicted (category=contradiction) + high-severity → critical
      high-severity → high
      medium → medium
      low → low
    Unknown severity → medium (never critical by default).
    """
    sev = (severity or "").lower().strip()
    cat = (category or "").lower().strip()
    if sev == "high" and cat == "contradiction":
        return "critical"
    if sev == "high":
        return "high"
    if sev == "medium":
        return "medium"
    if sev == "low":
        return "low"
    return "medium"


def _coerce_improvements(
    raw: object | None,
    valid_ids: set[str],
    valid_challenge_ids: set[str],
    valid_files: set[str] | None,
) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        iid = item.get("id")
        if not isinstance(iid, str) or not iid.strip():
            continue
        title = item.get("title")
        problem = item.get("problem", "")
        recommendation = item.get("recommendation", "")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(problem, str) or not isinstance(recommendation, str):
            continue
        if not problem.strip() or not recommendation.strip():
            continue
        category = _normalize_category(item.get("category"))
        rationale = item.get("rationale", "")
        if not isinstance(rationale, str):
            rationale = str(rationale)
        confidence = _normalize_confidence(item.get("confidence"))
        suggested_priority = _normalize_priority(item.get("priority"))
        citations = item.get("evidence_ids", [])
        valid_cites = validate_citations(citations, valid_ids)
        if not valid_cites:
            # No evidence → do not emit.
            continue
        related = _coerce_str_list(item.get("related_challenge_ids", []))
        # Keep only real challenge IDs; allow empty (judge-driven improvement).
        related = [r for r in related if r in valid_challenge_ids]
        affected = _coerce_str_list(item.get("affected_files", []))
        if valid_files is not None:
            affected = [f for f in affected if f in valid_files]
        out.append({
            "id": iid,
            "title": title,
            "problem": problem,
            "recommendation": recommendation,
            "suggested_priority": suggested_priority,
            "category": category,
            "rationale": rationale,
            "evidence_ids": valid_cites,
            "related_challenge_ids": related,
            "affected_files": affected,
            "confidence": confidence,
        })
    return out


def parse_improvements(
    raw: str,
    valid_ids: set[str],
    valid_challenge_ids: set[str],
    valid_files: set[str] | None,
) -> ParsedImprovements:
    """Parse the LLM's text into a ParsedImprovements."""
    parsed = _extract_json_object(raw or "")
    if parsed is not None:
        raw_list = parsed.get("improvements", []) if isinstance(parsed, dict) else []
        improvements = _coerce_improvements(raw_list, valid_ids, valid_challenge_ids, valid_files)
        return ParsedImprovements(improvements=improvements, raw_was_json=True)
    return ParsedImprovements(improvements=[], raw_was_json=False)


# ---------------------------------------------------------------------------
# Public entry point: generate improvements
# ---------------------------------------------------------------------------

IMPROVEMENT_QUERIES = {
    "testing": "tests test coverage pytest test files",
    "security": "security authentication authorization encryption",
    "architecture": "modular structure design patterns separation of concerns",
    "reliability": "error handling retry fault tolerance timeout",
    "completeness": "documentation examples CI CD type checking",
}

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _enum_val(v: object) -> str:
    """Return the raw string value of a str-Enum (or plain string)."""
    val = getattr(v, "value", v)
    return str(val).lower().strip() if isinstance(val, str) else str(val)


def generate_improvements(
    repo_id: str,
    *,
    settings: Settings,
    top_k: Optional[int] = None,
    llm: Optional[OllamaClient] = None,
    retrieve: Callable = search_evidence,
    judge_fn: Callable | None = None,
    challenges_fn: Callable | None = None,
) -> ImprovementResult:
    """Generate evidence-backed improvement recommendations.

    Uses M5 judge + M6 challenges as primary inputs, retrieves BM25 evidence
    for the top challenges, and asks the LLM for concrete next steps.
    Priority is computed deterministically from the source challenge so the
    LLM cannot arbitrarily assign critical.

    Raises RepoNotIndexedError when the repo has not been indexed.
    """
    from . import judge as judge_service
    from . import challenges as challenges_service

    judge_fn = judge_fn or judge_service.judge_repository
    challenges_fn = challenges_fn or challenges_service.generate_challenges

    effective_top_k = int(top_k) if top_k is not None else settings.qa_top_k
    per_query_k = min(3, effective_top_k)
    db_path = evidence_db_path(settings.storage_root, settings.db_filename)

    # 1 — M5 + M6 outputs (propagate RepoNotIndexedError).
    judge_result = judge_fn(repo_id=repo_id, settings=settings, top_k=3, llm=llm, retrieve=retrieve)
    challenge_result = challenges_fn(repo_id=repo_id, settings=settings, top_k=3, llm=llm, retrieve=retrieve)

    challenges: list[Challenge] = list(challenge_result.challenges or [])
    # Prioritize: high severity first, then contradiction category.
    challenges.sort(key=lambda c: (
        _SEVERITY_RANK.get(_enum_val(c.severity), 9),
        0 if _enum_val(c.category) == "contradiction" else 1,
    ))
    challenge_by_id = {c.id: c for c in challenges}
    valid_challenge_ids = set(challenge_by_id.keys())

    # 2 — Weak M5 dimensions (score < 6) for context.
    weak_dims = [d for d in (judge_result.dimensions or []) if d.score < 6.0]
    weak_dims.sort(key=lambda d: d.score)

    # 3 — Retrieve evidence grounded in the top challenges + weak areas.
    all_evidence = []
    seen_keys: set[tuple] = set()

    def _add(evidence_list) -> None:
        for r in evidence_list:
            key = (r.file_path, r.start_line, r.end_line)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_evidence.append(r)

    for c in challenges[:5]:
        query = f"{c.claim} {c.challenge}"[:400]
        _add(retrieve(db_path, repo_id, query, limit=per_query_k, default_limit=per_query_k))
    for dim in weak_dims[:3]:
        q = IMPROVEMENT_QUERIES.get(dim.name, dim.name.replace("_", " "))
        _add(retrieve(db_path, repo_id, q, limit=per_query_k, default_limit=per_query_k))

    # Also pull in evidence already cited by challenges (re-resolve via search
    # is unnecessary — reuse file paths as queries is wasteful; instead rely on
    # the retrieved set). If nothing retrieved, return empty (no-evidence rule).
    if not all_evidence:
        return ImprovementResult(
            repo_id=repo_id,
            improvements=[],
            total_improvements=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            evidence_citations=[],
        )

    # 4 — Label evidence blocks.
    blocks = label_evidence(repo_id, all_evidence)
    valid_ids = {b.id for b in blocks}
    by_id = {b.id: b for b in blocks}

    # 5 — Valid file set for affected_files (manifest inventory + evidence files).
    valid_files: set[str] | None = None
    try:
        from .ingestion import load_manifest
        owner, repo = repo_id.split("/")
        repo_dir = settings.storage_root / owner.lower() / repo.lower()
        raw_manifest = load_manifest(repo_dir, settings.manifest_filename)
        if raw_manifest is not None:
            inv = raw_manifest.get("file_inventory", []) if isinstance(raw_manifest, dict) else []
            paths = {e.get("path") for e in inv if isinstance(e, dict) and e.get("path")}
            paths |= {b.file_path for b in blocks}
            valid_files = {p for p in paths if isinstance(p, str)}
    except Exception:
        valid_files = {b.file_path for b in blocks} or None

    # 6 — Build prompt summarizing challenges + judge findings.
    challenge_lines = []
    for c in challenges:
        challenge_lines.append(
            f"  {c.id} | severity={c.severity} | category={c.category} | "
            f"claim={c.claim} | challenge={c.challenge} | explanation={c.explanation}"
        )
    challenge_summary = "\n".join(challenge_lines) if challenge_lines else "No challenges."
    judge_lines = []
    for d in (judge_result.dimensions or []):
        judge_lines.append(f"  {d.name}: {d.score}/10 — {d.explanation}")
    judge_summary = "\n".join(judge_lines) if judge_lines else "No dimensions."
    weakness_summary = "; ".join(judge_result.weaknesses or []) or "None listed."

    improvement_prompt = (
        f"CHALLENGES (M6, prioritized high → low):\n{challenge_summary}\n\n"
        f"JUDGE DIMENSIONS (M5):\n{judge_summary}\n\n"
        f"JUDGE WEAKNESSES: {weakness_summary}\n\n"
        f"CLAIM INTEGRITY: {judge_result.claim_integrity_summary}\n\n"
        "Generate SPECIFIC, ACTIONABLE improvements addressing the above, "
        "highest-priority challenges first. Each improvement must cite evidence "
        "and reference its challenge ID(s)."
    )

    messages = build_qa_messages(
        IMPROVEMENT_GENERATION_INSTRUCTION,
        blocks,
        improvement_prompt,
        json_instructions=IMPROVEMENT_JSON_INSTRUCTIONS,
    )

    # 7 — Call the LLM.
    llm = llm or OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        think=settings.ollama_think,
    )
    raw = llm.complete(messages)

    # 8 — Parse and validate.
    parsed = parse_improvements(raw, valid_ids, valid_challenge_ids, valid_files)

    # 9 — Build Improvement objects with deterministic priority.
    improvements: list[Improvement] = []
    all_evidence_ids: set[str] = set()
    for idx, item in enumerate(parsed.improvements):
        evidence_ids = item["evidence_ids"]
        all_evidence_ids.update(evidence_ids)
        related = item["related_challenge_ids"]
        if related:
            # Max severity among related challenges determines priority.
            best = sorted(
                related,
                key=lambda rid: (
                    _SEVERITY_RANK.get(_enum_val(challenge_by_id[rid].severity), 9),
                    0 if _enum_val(challenge_by_id[rid].category) == "contradiction" else 1,
                ),
            )[0]
            src = challenge_by_id[best]
            priority = deterministic_priority(_enum_val(src.severity), _enum_val(src.category))
            category = _enum_val(src.category)
        else:
            # Judge-driven improvement: never allow arbitrary critical.
            suggested = item["suggested_priority"]
            priority = "high" if suggested == "critical" else suggested
            category = item["category"]
        improvements.append(
            Improvement(
                id=item["id"] if item["id"].strip() else f"improvement_{idx + 1}",
                title=item["title"],
                problem=item["problem"],
                recommendation=item["recommendation"],
                priority=priority,  # type: ignore[arg-type]
                category=category,  # type: ignore[arg-type]
                rationale=item["rationale"],
                evidence_ids=evidence_ids,
                related_challenge_ids=related,
                affected_files=item["affected_files"],
                confidence=item["confidence"],  # type: ignore[arg-type]
            )
        )

    critical_count = sum(1 for i in improvements if i.priority == "critical")
    high_count = sum(1 for i in improvements if i.priority == "high")
    medium_count = sum(1 for i in improvements if i.priority == "medium")
    low_count = sum(1 for i in improvements if i.priority == "low")

    evidence_citations = []
    for eid in sorted(all_evidence_ids):
        if eid in by_id:
            b = by_id[eid]
            evidence_citations.append(
                Citation(
                    id=eid,
                    file_path=b.file_path,
                    start_line=b.start_line,
                    end_line=b.end_line,
                    language=b.language,
                    content=b.content,
                )
            )

    return ImprovementResult(
        repo_id=repo_id,
        improvements=improvements,
        total_improvements=len(improvements),
        critical_count=critical_count,
        high_count=high_count,
        medium_count=medium_count,
        low_count=low_count,
        evidence_citations=evidence_citations,
    )
