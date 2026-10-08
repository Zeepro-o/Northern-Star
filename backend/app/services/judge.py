"""M5 — Evidence-Based Project Judging.

Architecture:
  Repository
    → existing claims (from README)
    → existing evidence retrieval
    → claim verification (M4)
    → judge rubric
    → per-dimension evaluation
    → deterministic final score

Core principle: "LLMs reason; evidence determines what can be claimed."

Dimensions:
1. Technical Implementation (0-10) — code quality, patterns, practices
2. Architecture / Code Quality (0-10) — structure, modularity, maintainability
3. Claim Integrity (0-10) — how well README claims match implementation
4. Completeness (0-10) — docs, tests, examples, CI
5. Overall Quality (0-10) — holistic assessment

Overall score: sum of dimension scores * 2 (0-100)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import Settings
from ..models.schemas import Citation, Claim, DimensionScore, JudgeResult, SearchResult
from .claims import verify_claim
from .indexing import evidence_db_path
from .llm import OllamaClient
from .prompts import EvidenceBlock, build_qa_messages
from .qa import (
    _CITE_ID_RE,
    _BARE_ID_RE,
    _inline_citation_ids,
    label_evidence,
    parse_answer,
    sanitize_answer,
    validate_citations,
)
from .retrieval import RepoNotIndexedError, search_evidence


# ---------------------------------------------------------------------------
# Judge rubric / system prompts
# ---------------------------------------------------------------------------

JUDGE_INSTRUCTION = (
    "You are an evidence-grounded code judge for the repository shown in the "
    "evidence blocks below. Your job is to evaluate the repository across five "
    "dimensions and provide a structured assessment.\n"
    "\n"
    "Judging rules (non-negotiable):\n"
    "1. Evaluate using ONLY the evidence blocks supplied below. Never invent, "
    "guess, or recall facts not present in the evidence.\n"
    "2. Every substantive judgment MUST cite evidence blocks by their [E#] IDs. "
    "Do not cite an evidence block for a point it does not actually support.\n"
    "3. If the evidence does not contain enough information to evaluate a "
    "dimension, say explicitly that the evidence is insufficient and score it "
    "based on what is available (low score for missing evidence).\n"
    "4. README/documentation evidence is a *claim*, not proof of "
    "implementation. When you cite it, mark it as such and prefer source-code "
    "evidence for claims about how the code behaves.\n"
    "5. No evidence for a positive trait = low score in that area, NOT a "
    "contradiction. Lack of evidence is not evidence of lack.\n"
    "6. Cite only evidence IDs that appear in the supplied blocks.\n"
    "\n"
    "Dimensions to evaluate (each 0-10):\n"
    "1. TECHNICAL_IMPLEMENTATION: Code quality, correctness, patterns, error "
    "handling, type hints, modern practices.\n"
    "2. ARCHITECTURE: Structure, modularity, separation of concerns, "
    "extensibility, design patterns.\n"
    "3. CLAIM_INTEGRITY: How well do README/documentation claims match actual "
    "implementation? (Use the claim verification results provided).\n"
    "4. COMPLETENESS: Tests, documentation, examples, CI/CD, type checking, "
    "developer experience.\n"
    "5. OVERALL_QUALITY: Holistic assessment of the project as a whole.\n"
    "\n"
    "You must respond with ONLY a single JSON object, no prose around it, in "
    "this exact shape:\n"
    '{"dimensions": ['
    '  {"name": "technical_implementation", "score": 7.5, "explanation": "...", "evidence_ids": ["E1"]}, '
    '  {"name": "architecture", "score": 8.0, "explanation": "...", "evidence_ids": ["E2"]}, '
    '  {"name": "claim_integrity", "score": 6.0, "explanation": "...", "evidence_ids": ["E3"]}, '
    '  {"name": "completeness", "score": 7.0, "explanation": "...", "evidence_ids": ["E4"]}, '
    '  {"name": "overall_quality", "score": 7.5, "explanation": "...", "evidence_ids": ["E5"]} '
    '], '
    '"strengths": ["Strength 1 [E1]", "Strength 2 [E2]"], '
    '"weaknesses": ["Weakness 1 [E3]", "Weakness 2 [E4]"], '
    '"recommendations": ["Recommendation 1", "Recommendation 2"]}\n'
    "Where every explanation cites evidence by ID like [E1]. "
    "strengths/weaknesses/recommendations may reference evidence IDs.\n"
)

JUDGE_JSON_INSTRUCTIONS = (
    "Respond with only a single JSON object with exactly these fields:\n"
    '{"dimensions": [{"name": "technical_implementation", "score": 0-10, '
    '"explanation": "...", "evidence_ids": ["E1", ...]}, ...], '
    '"strengths": ["... [E1]"], "weaknesses": ["... [E2]"], '
    '"recommendations": ["..."]}'
    "In explanations, cite evidence by ID like [E1]. "
    "strengths/weaknesses/recommendations may reference evidence IDs."
)


# ---------------------------------------------------------------------------
# Internal parsed judge response
# ---------------------------------------------------------------------------

@dataclass
class ParsedJudge:
    dimensions: list[dict]
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    raw_was_json: bool


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------

_DIMENSION_NAMES = [
    "technical_implementation",
    "architecture",
    "claim_integrity",
    "completeness",
    "overall_quality",
]


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
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_dimension_score(value: object | None, fallback: float = 5.0) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(10.0, float(value)))
    return fallback


def _coerce_dimension(raw: object | None, valid_ids: set[str]) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if name not in _DIMENSION_NAMES:
        return None
    score = _normalize_dimension_score(raw.get("score"))
    explanation = raw.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = str(explanation)
    # Accept both "citations" and "evidence_ids" from model response
    citations = raw.get("citations", raw.get("evidence_ids", []))
    valid_cites = validate_citations(citations, valid_ids)
    return {
        "name": name,
        "score": score,
        "explanation": explanation,
        "evidence_ids": valid_cites,
    }


def _coerce_list(raw: object | None) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            out.append(item)
    return out


def parse_judge(raw: str, valid_ids: set[str]) -> ParsedJudge:
    """Parse the LLM's text into a ParsedJudge."""
    parsed = _extract_json_object(raw or "")
    if parsed is not None:
        dims_raw = parsed.get("dimensions", []) if isinstance(parsed, dict) else []
        dimensions = []
        for d in dims_raw:
            coerced = _coerce_dimension(d, valid_ids)
            if coerced:
                # Also mine inline citations from explanation as fallback
                explanation = coerced.get("explanation", "")
                inline = _inline_citation_ids(explanation, valid_ids)
                declared = set(coerced.get("evidence_ids", []))
                # Merge declared and inline, preferring declared order
                merged = list(declared) + [c for c in inline if c not in declared]
                coerced["evidence_ids"] = merged
                dimensions.append(coerced)
        strengths = _coerce_list(parsed.get("strengths"))
        weaknesses = _coerce_list(parsed.get("weaknesses"))
        recommendations = _coerce_list(parsed.get("recommendations"))
        return ParsedJudge(
            dimensions=dimensions,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            raw_was_json=True,
        )
    # Fallback — plain text, try to extract structured info
    text = (raw or "").strip()
    return ParsedJudge(
        dimensions=[],
        strengths=[],
        weaknesses=[],
        recommendations=[],
        raw_was_json=False,
    )


# ---------------------------------------------------------------------------
# Public entry point: judge a repository
# ---------------------------------------------------------------------------

JUDGE_QUERIES = {
    "technical_implementation": "code quality patterns error handling type hints modern practices",
    "architecture": "modular structure design patterns separation of concerns extensibility",
    "claim_integrity": "README claims documentation matches implementation",
    "completeness": "tests documentation examples CI CD type checking",
    "overall_quality": "project quality maintainability usability",
}

UNSUFFICIENT_EVIDENCE = "Insufficient evidence to evaluate this dimension."


def judge_repository(
    repo_id: str,
    *,
    settings: Settings,
    top_k: Optional[int] = None,
    llm: Optional[OllamaClient] = None,
    retrieve: Callable = search_evidence,
) -> JudgeResult:
    """Judge a repository across five dimensions using evidence.

    Returns a JudgeResult with deterministic overall score.
    Raises RepoNotIndexedError when the repo has not been indexed.
    """
    effective_top_k = int(top_k) if top_k is not None else settings.qa_top_k
    # Use smaller budget per dimension to keep total manageable for LLM
    per_dim_k = min(3, effective_top_k)
    db_path = evidence_db_path(settings.storage_root, settings.db_filename)

    # 1 — Get claims from README for claim integrity dimension
    from .detection import extract_structured_claims
    from .ingestion import load_manifest

    # Find the checkout path
    owner, repo = repo_id.split("/")
    checkout_path = settings.storage_root / owner.lower() / repo.lower() / "checkout"
    readme_path = checkout_path / "README.md"
    claims: list[Claim] = []
    if checkout_path.exists():
        if not readme_path.exists():
            for f in checkout_path.iterdir():
                if f.name.lower() == "readme.md":
                    readme_path = f
                    break
        if readme_path.exists():
            readme_text = readme_path.read_text(encoding="utf-8", errors="replace")
            claims = extract_structured_claims(readme_text, source="README.md")
            for c in claims:
                c.repo_id = repo_id

    # 2 — Verify claims (M4) to get claim integrity data
    verified_claims: list[Claim] = []
    for claim in claims:
        try:
            verified = verify_claim(
                repo_id=repo_id,
                claim=claim,
                settings=settings,
                top_k=3,  # smaller budget for claim verification
                llm=llm,
                retrieve=retrieve,
            )
            verified_claims.append(verified)
        except RepoNotIndexedError:
            raise
        except Exception:
            # If claim verification fails, keep original claim with unclear verdict
            claim.verdict = "unclear"
            claim.verdict_explanation = "Claim verification failed."
            claim.evidence_ids = []
            verified_claims.append(claim)

    # 3 — Compute claim integrity summary
    claim_counts = {
        "supported": 0,
        "partially_supported": 0,
        "unclear": 0,
        "contradicted": 0,
    }
    for c in verified_claims:
        if c.verdict in claim_counts:
            claim_counts[c.verdict] += 1

    # 4 — Retrieve evidence for each dimension
    all_evidence: list[SearchResult] = []
    dimension_evidence: dict[str, list[SearchResult]] = {}

    for dim_name, query in JUDGE_QUERIES.items():
        evidence = retrieve(
            db_path,
            repo_id,
            query,
            limit=per_dim_k,
            default_limit=per_dim_k,
        )
        dimension_evidence[dim_name] = evidence
        all_evidence.extend(evidence)

    # Also add claim verification evidence
    claim_evidence_ids = set()
    for c in verified_claims:
        claim_evidence_ids.update(c.evidence_ids)

    # 5 — Label evidence blocks
    blocks = label_evidence(repo_id, all_evidence)
    valid_ids = {b.id for b in blocks}
    by_id = {b.id: b for b in blocks}

    # 6 — Build judge prompt with claim integrity data
    claim_summary_lines = []
    for c in verified_claims:
        claim_summary_lines.append(
            f"  CLAIM: {c.text} | VERDICT: {c.verdict} | EXPLANATION: {c.verdict_explanation}"
        )
    claim_summary = "\n".join(claim_summary_lines) if claim_summary_lines else "No claims extracted."

    judge_prompt = (
        f"CLAIM INTEGRITY SUMMARY (from prior verification):\n"
        f"Total claims: {len(verified_claims)}\n"
        f"Supported: {claim_counts['supported']}\n"
        f"Partially supported: {claim_counts['partially_supported']}\n"
        f"Unclear: {claim_counts['unclear']}\n"
        f"Contradicted: {claim_counts['contradicted']}\n\n"
        f"DETAILS:\n{claim_summary}\n\n"
        "CRITICAL: You MUST evaluate ALL FIVE dimensions below and return them in your JSON response:\n"
        "1. technical_implementation\n"
        "2. architecture\n"
        "3. claim_integrity\n"
        "4. completeness\n"
        "5. overall_quality\n\n"
        "For each dimension, provide: score (0-10), explanation with [E#] citations, and evidence_ids array.\n"
        "Also provide strengths, weaknesses, and recommendations arrays.\n\n"
        "Now evaluate using the evidence blocks below."
    )

    messages = build_qa_messages(
        JUDGE_INSTRUCTION,
        blocks,
        judge_prompt,
        json_instructions=JUDGE_JSON_INSTRUCTIONS,
    )

    # 7 — Call the LLM
    llm = llm or OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        think=settings.ollama_think,
    )
    raw = llm.complete(messages)

    # 8 — Parse and validate
    parsed = parse_judge(raw, valid_ids)

    # 9 — Build DimensionScore objects
    dim_scores: list[DimensionScore] = []
    all_evidence_ids: set[str] = set()

    for dim_data in parsed.dimensions:
        score = _normalize_dimension_score(dim_data.get("score"))
        explanation = dim_data.get("explanation", "")
        evidence_ids = dim_data.get("evidence_ids", [])
        all_evidence_ids.update(evidence_ids)

        dim_scores.append(
            DimensionScore(
                name=dim_data["name"],
                score=score,
                explanation=explanation,
                evidence_ids=evidence_ids,
            )
        )

    # Ensure all 5 dimensions present (fill missing with low scores)
    existing_names = {d.name for d in dim_scores}
    for dim_name in _DIMENSION_NAMES:
        if dim_name not in existing_names:
            dim_scores.append(
                DimensionScore(
                    name=dim_name,
                    score=0.0,
                    explanation=f"No evidence available to evaluate {dim_name}.",
                    evidence_ids=[],
                )
            )

    # Sort by defined order
    dim_scores.sort(key=lambda d: _DIMENSION_NAMES.index(d.name))

    # 10 — Calculate deterministic overall score (sum of 5 dimensions * 2 = 0-100)
    overall_score = int(round(sum(d.score for d in dim_scores) * 2))
    overall_score = max(0, min(100, overall_score))

    # 11 — Resolve Citation objects for all cited evidence
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

    return JudgeResult(
        repo_id=repo_id,
        overall_score=overall_score,
        dimensions=dim_scores,
        strengths=parsed.strengths,
        weaknesses=parsed.weaknesses,
        recommendations=parsed.recommendations,
        claim_integrity_summary=claim_counts,
        total_claims=len(verified_claims),
        evidence_citations=evidence_citations,
    )