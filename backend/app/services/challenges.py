"""M6 — Red-Team / Challenge Engine.

Architecture:
  Existing repository
    → M4 claims
    → M2 evidence retrieval
    → M4 verification
    → M5 judge results
    → challenge generation
    → evidence retrieval
    → challenge validation
    → structured ChallengeResult

Core principle: "LLMs reason; evidence determines what can be claimed."

Challenge types:
  A. Unsupported claim — "Where is X implemented?"
  B. Contradiction — "README says X, but evidence shows Y" (high severity)
  C. Missing implementation — "Claims X, but no evidence found"
  D. Architecture challenge — "Concentrated responsibilities in single module"
  E. Testing challenge — "Core functionality claimed but tests not evident"
  F. Security challenge — "Security claim without implementation evidence"
  G. Reliability challenge — "No evidence of error handling/retry logic"
  H. Scalability challenge — "Scaling claim without supporting architecture"
  I. Completeness challenge — "Core feature lacks documentation/examples"
  J. Documentation challenge — "Documentation claims not backed by implementation"

Every challenge must have:
  - evidence IDs
  - exact file:line citations
  - explanation tied to those evidence chunks

If evidence is insufficient → mark challenge as unclear or do not emit it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import Settings
from ..models.schemas import (
    Challenge,
    ChallengeCategory,
    ChallengeResult,
    ChallengeSeverity,
    Citation,
    Claim,
    SearchResult,
)
from .claims import verify_claim
from .indexing import evidence_db_path
from .llm import OllamaClient
from .prompts import EvidenceBlock, build_qa_messages
from .qa import (
    _CITE_ID_RE,
    _BARE_ID_RE,
    _inline_citation_ids,
    label_evidence,
    sanitize_answer,
    validate_citations,
)
from .retrieval import RepoNotIndexedError, search_evidence


# ---------------------------------------------------------------------------
# System prompt for challenge generation
# ---------------------------------------------------------------------------

CHALLENGE_GENERATION_INSTRUCTION = (
    "You are an evidence-grounded red-team analyst for the repository shown in the "
    "evidence blocks below. Your job is to generate SPECIFIC, EVIDENCE-BACKED challenges "
    "against the project's claims, assumptions, and implementation.\n"
    "\n"
    "Challenge generation rules (non-negotiable):\n"
    "1. Generate challenges ONLY from the evidence blocks supplied below and the "
    "CLAIM INTEGRITY SUMMARY. Never invent, guess, or recall facts not present.\n"
    "2. Every challenge MUST cite evidence blocks by their [E#] IDs. Do not cite "
    "an evidence block for a point it does not actually support.\n"
    "3. If the evidence does not contain enough information to substantiate a challenge, "
    "do not generate that challenge. Do not generate generic criticism.\n"
    "4. README/documentation evidence is a *claim*, not proof of implementation. "
    "When you cite it, mark it as such and prefer source-code evidence.\n"
    "5. No evidence for a positive trait = low score/weakness, NOT a contradiction. "
    "Lack of evidence is not evidence of lack.\n"
    "6. Cite only evidence IDs that appear in the supplied blocks.\n"
    "7. Prioritize challenges from:\n"
    "   - CONTRADICTED claims (high severity)\n"
    "   - PARTIALLY_SUPPORTED claims (medium severity)\n"
    "   - UNCLEAR claims (medium/low severity depending on importance)\n"
    "   - Important SUPPORTED claims whose implementation deserves verification\n"
    "   - Architecture/implementation patterns visible in evidence\n"
    "\n"
    "Challenge types to consider:\n"
    "A. UNSUPPORTED_CLAIM — \"Project claims X, but evidence does not show X\"\n"
    "B. CONTRADICTION — \"README claims X, but evidence shows Y\" (high severity)\n"
    "C. MISSING_IMPLEMENTATION — \"Claims X, but no implementation evidence found\"\n"
    "D. ARCHITECTURE — \"Concentrated responsibilities / poor separation of concerns\"\n"
    "E. SECURITY — \"Security claim without implementation evidence\"\n"
    "F. RELIABILITY — \"No evidence of error handling / retry logic / fault tolerance\"\n"
    "G. SCALABILITY — \"Scaling claim without supporting architecture evidence\"\n"
    "H. TESTING — \"Core functionality claimed but tests not evident\"\n"
    "I. COMPLETENESS — \"Core feature lacks docs/examples/CI\"\n"
    "J. DOCUMENTATION — \"Documentation claims not backed by implementation\"\n"
    "\n"
    "You must respond with ONLY a single JSON object, no prose around it, in "
    "this exact shape:\n"
    '{"challenges": [\n'
    '  {\n'
    '    "id": "challenge_1",\n'
    '    "claim": "The project claims X",\n'
    '    "challenge": "Where is X implemented?",\n'
    '    "severity": "high|medium|low",\n'
    '    "category": "unsupported_claim|contradiction|missing_implementation|architecture|security|reliability|scalability|testing|completeness|documentation",\n'
    '    "explanation": "The evidence shows... [E1]",\n'
    '    "evidence_ids": ["E1", "E2"],\n'
    '    "confidence": "high|medium|low"\n'
    '  }\n'
    '],\n'
    '"total_challenges": 3\n'
    '}\n'
    "Where every explanation cites evidence by ID like [E1]. "
    "Only include challenges that have at least one valid evidence citation.\n"
    "If no valid challenges can be generated, return empty challenges array.\n"
)

CHALLENGE_JSON_INSTRUCTIONS = (
    "Respond with only a single JSON object with exactly these fields:\n"
    '{"challenges": [{"id": "challenge_1", "claim": "...", "challenge": "...", '
    '"severity": "high|medium|low", "category": "unsupported_claim|contradiction|...", '
    '"explanation": "... [E1]", "evidence_ids": ["E1"], "confidence": "high|medium|low"}], '
    '"total_challenges": 3}\n'
    "In explanations, cite evidence by ID like [E1]. "
    "Only include challenges that have at least one valid evidence citation.\n"
    "If no valid challenges can be generated, return empty challenges array."
)


# ---------------------------------------------------------------------------
# Internal parsed challenge response
# ---------------------------------------------------------------------------

@dataclass
class ParsedChallenges:
    challenges: list[dict]
    raw_was_json: bool


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------

_CHALLENGE_CATEGORIES = {
    "unsupported_claim", "contradiction", "missing_implementation",
    "architecture", "security", "reliability", "scalability",
    "testing", "completeness", "documentation"
}

_CHALLENGE_SEVERITIES = {"high", "medium", "low"}
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
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_severity(value: object | None, fallback: str = "medium") -> str:
    if isinstance(value, str):
        v = value.lower().strip()
        if v in _CHALLENGE_SEVERITIES:
            return v
    return fallback


def _normalize_category(value: object | None, fallback: str = "unsupported_claim") -> str:
    if isinstance(value, str):
        v = value.lower().strip()
        if v in _CHALLENGE_CATEGORIES:
            return v
    return fallback


def _normalize_confidence(value: object | None, fallback: str = "medium") -> str:
    if isinstance(value, str):
        v = value.lower().strip()
        if v in _CONFIDENCE_LEVELS:
            return v
    return fallback


def _coerce_challenges(raw: object | None, valid_ids: set[str]) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = item.get("id")
        if not isinstance(cid, str):
            continue
        claim = item.get("claim")
        challenge = item.get("challenge")
        if not isinstance(claim, str) or not isinstance(challenge, str):
            continue
        severity = _normalize_severity(item.get("severity"))
        category = _normalize_category(item.get("category"))
        explanation = item.get("explanation", "")
        if not isinstance(explanation, str):
            explanation = str(explanation)
        confidence = _normalize_confidence(item.get("confidence"))
        citations = item.get("evidence_ids", [])
        valid_cites = validate_citations(citations, valid_ids)
        if not valid_cites:
            # Challenge without any valid evidence citation is not allowed
            continue
        out.append({
            "id": cid,
            "claim": claim,
            "challenge": challenge,
            "severity": severity,
            "category": category,
            "explanation": explanation,
            "evidence_ids": valid_cites,
            "confidence": confidence,
        })
    return out


def parse_challenges(raw: str, valid_ids: set[str]) -> ParsedChallenges:
    """Parse the LLM's text into a ParsedChallenges."""
    parsed = _extract_json_object(raw or "")
    if parsed is not None:
        challenges_raw = parsed.get("challenges", []) if isinstance(parsed, dict) else []
        challenges = _coerce_challenges(challenges_raw, valid_ids)
        return ParsedChallenges(
            challenges=challenges,
            raw_was_json=True,
        )
    # Fallback — plain text, try to extract structured info
    return ParsedChallenges(
        challenges=[],
        raw_was_json=False,
    )


# ---------------------------------------------------------------------------
# Public entry point: generate challenges
# ---------------------------------------------------------------------------

CHALLENGE_QUERIES = {
    "claims": "README claims documentation matches implementation",
    "architecture": "modular structure design patterns separation of concerns",
    "implementation": "code quality patterns error handling type hints",
    "testing": "tests test coverage pytest test files",
    "security": "security authentication authorization encryption",
    "reliability": "error handling retry fault tolerance timeout",
    "scalability": "performance scaling concurrency async",
    "completeness": "documentation examples CI CD type checking",
}


def generate_challenges(
    repo_id: str,
    *,
    settings: Settings,
    top_k: Optional[int] = None,
    llm: Optional[OllamaClient] = None,
    retrieve: Callable = search_evidence,
) -> ChallengeResult:
    """Generate red-team challenges for a repository.

    Returns a ChallengeResult with validated challenges.
    Raises RepoNotIndexedError when the repo has not been indexed.
    """
    effective_top_k = int(top_k) if top_k is not None else settings.qa_top_k
    per_dim_k = min(3, effective_top_k)
    db_path = evidence_db_path(settings.storage_root, settings.db_filename)

    # 1 — Get claims from README and verify them (M4)
    from .detection import extract_structured_claims
    from .ingestion import load_manifest

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
                top_k=3,
                llm=llm,
                retrieve=retrieve,
            )
            verified_claims.append(verified)
        except RepoNotIndexedError:
            raise
        except Exception:
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

    # 4 — Build claim summary for prompt
    claim_summary_lines = []
    for c in verified_claims:
        claim_summary_lines.append(
            f"  CLAIM: {c.text} | VERDICT: {c.verdict} | EXPLANATION: {c.verdict_explanation}"
        )
    claim_summary = "\n".join(claim_summary_lines) if claim_summary_lines else "No claims extracted."

    # 5 — Retrieve evidence for challenge generation
    all_evidence: list[SearchResult] = []

    for query in CHALLENGE_QUERIES.values():
        evidence = retrieve(
            db_path,
            repo_id,
            query,
            limit=per_dim_k,
            default_limit=per_dim_k,
        )
        all_evidence.extend(evidence)

    # Also add claim verification evidence
    claim_evidence_ids = set()
    for c in verified_claims:
        claim_evidence_ids.update(c.evidence_ids)

    # 6 — Label evidence blocks
    blocks = label_evidence(repo_id, all_evidence)
    valid_ids = {b.id for b in blocks}
    by_id = {b.id: b for b in blocks}

    # 7 — Build challenge generation prompt
    challenge_prompt = (
        f"CLAIM INTEGRITY SUMMARY (from M4 verification):\n"
        f"Total claims: {len(verified_claims)}\n"
        f"Supported: {claim_counts['supported']}\n"
        f"Partially supported: {claim_counts['partially_supported']}\n"
        f"Unclear: {claim_counts['unclear']}\n"
        f"Contradicted: {claim_counts['contradicted']}\n\n"
        f"DETAILS:\n{claim_summary}\n\n"
        "Generate SPECIFIC, EVIDENCE-BACKED challenges based on the above and evidence blocks below.\n"
        "Prioritize: contradicted claims > partially_supported > unclear > important supported claims.\n"
        "Only output challenges with valid evidence citations."
    )

    messages = build_qa_messages(
        CHALLENGE_GENERATION_INSTRUCTION,
        blocks,
        challenge_prompt,
        json_instructions=CHALLENGE_JSON_INSTRUCTIONS,
    )

    # 8 — Call the LLM
    llm = llm or OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        think=settings.ollama_think,
    )
    raw = llm.complete(messages)

    # 9 — Parse and validate challenges
    parsed = parse_challenges(raw, valid_ids)

    # 10 — Build Challenge objects
    challenges: list[Challenge] = []
    all_evidence_ids: set[str] = set()

    for c_data in parsed.challenges:
        evidence_ids = c_data.get("evidence_ids", [])
        all_evidence_ids.update(evidence_ids)

        # Determine severity based on category and source claim verdict
        severity = c_data.get("severity", "medium")
        category = c_data.get("category", "unsupported_claim")

        # Boost severity for contradicted claims
        if category == "contradiction":
            severity = "high"

        challenges.append(
            Challenge(
                id=c_data["id"],
                claim=c_data["claim"],
                challenge=c_data["challenge"],
                severity=severity,
                category=category,
                explanation=c_data.get("explanation", ""),
                evidence_ids=evidence_ids,
                repo_id=repo_id,
                confidence=c_data.get("confidence", "medium"),
            )
        )

    # 11 — Calculate severity counts
    high_severity = sum(1 for c in challenges if c.severity == "high")
    medium_severity = sum(1 for c in challenges if c.severity == "medium")
    low_severity = sum(1 for c in challenges if c.severity == "low")

    # 12 — Resolve Citation objects for all cited evidence
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

    return ChallengeResult(
        repo_id=repo_id,
        challenges=challenges,
        total_challenges=len(challenges),
        high_severity=high_severity,
        medium_severity=medium_severity,
        low_severity=low_severity,
        evidence_citations=evidence_citations,
    )