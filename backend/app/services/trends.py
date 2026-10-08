"""M8.2 — Historical Trend Intelligence.

Snapshots of M8.1 discovery metadata are stored in the existing SQLite
evidence database (``discovery_snapshots`` table) and compared
deterministically over 24h / 7d / 30d windows. No LLM anywhere on this path:
every number below is arithmetic over stored snapshots, and every field is
null when history is unavailable — never fabricated.

Rank semantics: rank_change = previous_rank - current_rank, so positive
means the repository moved UP, negative means DOWN, zero means unchanged.

Emerging score (popularity vs emergence — a huge repo must not dominate
merely for being huge, so absolute size never enters the formula):

    star_rate   = clamp(star_delta / max(prev_stars, 1), 0, 5)
    fork_rate   = clamp(fork_delta / max(prev_forks, 1), 0, 5)
    sg, fg      = star_rate / max_star_rate, fork_rate / max_fork_rate
                  (set-normalized; 0 when the set max is 0)
    rank_imp    = (clamp(prev_rank - curr_rank, -10, 20) + 10) / 30
                  (0.0 when either rank is missing)
    activity    = recency of current pushed_at vs the current snapshot:
                  1.0 if <= 30d old, linear to 0.0 at 365d, 0.0 if missing

    emerging_score = round(0.45*sg + 0.20*fg + 0.25*rank_imp + 0.10*activity, 4)

Null when the repository has no previous snapshot in the window.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from ..config import Settings
from ..models.schemas import (
    RepositoryHistoryResult,
    SnapshotPoint,
    TrendRepository,
    TrendResult,
    TrendSnapshotResult,
)
from .discovery import validate_trending_limit
from .indexing import connect_evidence_db, evidence_db_path

WINDOWS: dict[str, int] = {
    "24h": 24 * 3600,
    "7d": 7 * 24 * 3600,
    "30d": 30 * 24 * 3600,
}

VALID_WINDOWS = ("24h", "7d", "30d")

# A previous snapshot must fall within ±20% of the requested window around
# the target instant; otherwise there is no valid comparison.
TOLERANCE_FRACTION = 0.20


class TrendValidationError(ValueError):
    """Malformed window/limit/slug for a trends request."""


def validate_window(window: str) -> str:
    if window not in WINDOWS:
        raise TrendValidationError(
            f"window must be one of: {', '.join(VALID_WINDOWS)}."
        )
    return window


def validate_limit(limit: int, default: int = 20) -> int:
    limit = default if limit is None else limit
    if not isinstance(limit, int) or not 1 <= limit <= 100:
        raise TrendValidationError("limit must be an integer between 1 and 100.")
    return limit


# ---------------------------------------------------------------------------
# Low-level snapshot store (same SQLite file as M1-M8.1; additive table)
# ---------------------------------------------------------------------------

def _db(settings: Settings):
    return connect_evidence_db(
        evidence_db_path(settings.storage_root, settings.db_filename))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_time(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (ValueError, TypeError, AttributeError):
        return None


def store_snapshot_rows(
    settings: Settings,
    snapshot_at: str,
    rows: list[dict[str, Any]],
) -> tuple[int, int]:
    """Insert rows; duplicates on (full_name, snapshot_at) are skipped.

    Returns (new_rows, skipped_duplicates).
    """
    new_rows = 0
    skipped = 0
    conn = _db(settings)
    try:
        with conn:
            for row in rows:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO discovery_snapshots"
                    " (full_name, snapshot_at, rank, stars, forks,"
                    "  open_issues, watchers, pushed_at, trend_score,"
                    "  language, topics, html_url)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row.get("full_name"),
                        snapshot_at,
                        row.get("rank"),
                        int(row.get("stars") or 0),
                        int(row.get("forks") or 0),
                        int(row.get("open_issues") or 0),
                        int(row.get("watchers") or 0),
                        row.get("pushed_at"),
                        row.get("trend_score"),
                        row.get("language"),
                        json.dumps(row.get("topics") or []),
                        row.get("html_url"),
                    ),
                )
                if cur.rowcount and cur.rowcount > 0:
                    new_rows += 1
                else:
                    skipped += 1
    finally:
        conn.close()
    return new_rows, skipped


def list_snapshot_times(settings: Settings) -> list[str]:
    conn = _db(settings)
    try:
        rows = conn.execute(
            "SELECT DISTINCT snapshot_at FROM discovery_snapshots"
            " ORDER BY snapshot_at ASC"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def _rows_at(settings: Settings, snapshot_at: str) -> list[dict[str, Any]]:
    conn = _db(settings)
    try:
        rows = conn.execute(
            "SELECT full_name, snapshot_at, rank, stars, forks, open_issues,"
            " watchers, pushed_at, trend_score, language, topics, html_url"
            " FROM discovery_snapshots WHERE snapshot_at = ?"
            " ORDER BY full_name ASC",
            (snapshot_at,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _rows_for_repo(
    settings: Settings, full_name: str, since: Optional[str] = None
) -> list[dict[str, Any]]:
    conn = _db(settings)
    try:
        if since is None:
            rows = conn.execute(
                "SELECT full_name, snapshot_at, rank, stars, forks, open_issues,"
                " watchers, pushed_at, trend_score, language, topics, html_url"
                " FROM discovery_snapshots WHERE full_name = ?"
                " ORDER BY snapshot_at ASC",
                (full_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT full_name, snapshot_at, rank, stars, forks, open_issues,"
                " watchers, pushed_at, trend_score, language, topics, html_url"
                " FROM discovery_snapshots WHERE full_name = ? AND snapshot_at >= ?"
                " ORDER BY snapshot_at ASC",
                (full_name, since),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _to_point(row: dict[str, Any]) -> SnapshotPoint:
    try:
        topics = json.loads(row.get("topics") or "[]")
        if not isinstance(topics, list):
            topics = []
    except (ValueError, TypeError):
        topics = []
    return SnapshotPoint(
        full_name=row.get("full_name") or "",
        snapshot_at=row.get("snapshot_at") or "",
        rank=row.get("rank"),
        stars=int(row.get("stars") or 0),
        forks=int(row.get("forks") or 0),
        open_issues=int(row.get("open_issues") or 0),
        watchers=int(row.get("watchers") or 0),
        pushed_at=row.get("pushed_at"),
        trend_score=row.get("trend_score"),
        language=row.get("language"),
        topics=[t for t in topics if isinstance(t, str)],
        html_url=row.get("html_url"),
    )


# ---------------------------------------------------------------------------
# Snapshot capture (explicit action only — no scheduler)
# ---------------------------------------------------------------------------

def capture_trending_snapshot(
    *,
    settings: Settings,
    limit: int = 100,
    transport: Optional[Any] = None,
) -> TrendSnapshotResult:
    """Fetch the current M8.1 trending set and store it. Reuses the
    discovery GitHub client (token handling stays there)."""
    from . import discovery as discovery_service

    limit = validate_trending_limit(limit)
    trending = discovery_service.get_trending(
        settings=settings, limit=limit, transport=transport, fresh=True)
    snapshot_at = _utc_now_iso()
    rows = [
        {
            "full_name": r.full_name,
            "rank": r.rank,
            "stars": r.stars,
            "forks": r.forks,
            "open_issues": r.open_issues,
            "watchers": r.watchers,
            "pushed_at": r.pushed_at,
            "trend_score": r.trend_score,
            "language": r.language,
            "topics": r.topics,
            "html_url": r.html_url,
        }
        for r in trending.repositories
    ]
    new_rows, skipped = store_snapshot_rows(settings, snapshot_at, rows)
    return TrendSnapshotResult(
        snapshot_at=snapshot_at,
        limit=limit,
        repositories_captured=len(rows),
        new_rows=new_rows,
        skipped_duplicates=skipped,
    )


# ---------------------------------------------------------------------------
# Window comparison (deterministic; nulls when history is unavailable)
# ---------------------------------------------------------------------------

def _find_previous_time(
    times: list[str], current: datetime, window_secs: int
) -> Optional[str]:
    """Closest snapshot instant to (current - window) within ±20% tolerance."""
    target = current - timedelta(seconds=window_secs)
    tolerance = timedelta(seconds=window_secs * TOLERANCE_FRACTION)
    best: Optional[str] = None
    best_key: Optional[tuple[float, str]] = None
    for raw in times:
        parsed = _parse_time(raw)
        if parsed is None or parsed >= current:
            continue
        diff = abs((parsed - target).total_seconds())
        if diff <= tolerance.total_seconds():
            key = (diff, raw)
            if best_key is None or key < best_key:
                best_key = key
                best = raw
    return best


def _growth_percent(delta: int, previous: int) -> Optional[float]:
    if previous <= 0:
        return None
    return round(delta / previous * 100.0, 2)


def _recency(pushed_at: Optional[str], current: datetime) -> float:
    parsed = _parse_time(pushed_at) if pushed_at else None
    if parsed is None:
        return 0.0
    days = max(0.0, (current - parsed).total_seconds() / 86400)
    if days <= 30:
        return 1.0
    if days >= 365:
        return 0.0
    return round(1.0 - (days - 30) / 335.0, 4)


def emerging_scores(
    pairs: list[tuple[dict[str, Any], Optional[dict[str, Any]]]],
    current_time: datetime,
) -> dict[str, Optional[float]]:
    """Emerging score per full_name; None where history is unavailable.

    ``pairs`` is (current_row, previous_row|None). See module docstring.
    """
    raw: dict[str, tuple[float, float, float]] = {}
    for curr, prev in pairs:
        name = curr.get("full_name") or ""
        if not prev:
            continue
        star_rate = min(max(
            ((curr.get("stars") or 0) - (prev.get("stars") or 0))
            / max(prev.get("stars") or 0, 1), 0.0), 5.0)
        fork_rate = min(max(
            ((curr.get("forks") or 0) - (prev.get("forks") or 0))
            / max(prev.get("forks") or 0, 1), 0.0), 5.0)
        curr_rank = curr.get("rank")
        prev_rank = prev.get("rank")
        if isinstance(curr_rank, int) and isinstance(prev_rank, int):
            rank_imp = (min(max(prev_rank - curr_rank, -10), 20) + 10) / 30.0
        else:
            rank_imp = 0.0
        raw[name] = (star_rate, fork_rate, rank_imp)
    max_star = max([v[0] for v in raw.values()] + [0.0])
    max_fork = max([v[1] for v in raw.values()] + [0.0])
    out: dict[str, Optional[float]] = {}
    for curr, prev in pairs:
        name = curr.get("full_name") or ""
        if not prev or name not in raw:
            out[name] = None
            continue
        star_rate, fork_rate, rank_imp = raw[name]
        sg = star_rate / max_star if max_star > 0 else 0.0
        fg = fork_rate / max_fork if max_fork > 0 else 0.0
        activity = _recency(curr.get("pushed_at"), current_time)
        out[name] = round(0.45 * sg + 0.20 * fg + 0.25 * rank_imp + 0.10 * activity, 4)
    return out


def _compare_pair(
    curr: dict[str, Any],
    prev: Optional[dict[str, Any]],
    emerging: Optional[float],
    window: str,
) -> TrendRepository:
    if not prev:
        return TrendRepository(
            full_name=curr.get("full_name") or "",
            html_url=curr.get("html_url"),
            language=curr.get("language"),
            stars=int(curr.get("stars") or 0),
            forks=int(curr.get("forks") or 0),
            current_rank=curr.get("rank"),
            pushed_at=curr.get("pushed_at"),
            current_trend_score=curr.get("trend_score"),
            emerging_score=None,
            history_available=False,
            comparison_window=window,
        )
    curr_rank = curr.get("rank")
    prev_rank = prev.get("rank")
    rank_change = (
        (prev_rank - curr_rank)
        if isinstance(curr_rank, int) and isinstance(prev_rank, int)
        else None
    )
    star_delta = (curr.get("stars") or 0) - (prev.get("stars") or 0)
    fork_delta = (curr.get("forks") or 0) - (prev.get("forks") or 0)
    curr_ts = curr.get("trend_score")
    prev_ts = prev.get("trend_score")
    ts_change = None
    if isinstance(curr_ts, (int, float)) and isinstance(prev_ts, (int, float)):
        ts_change = round(float(curr_ts) - float(prev_ts), 4)
    return TrendRepository(
        full_name=curr.get("full_name") or "",
        html_url=curr.get("html_url"),
        language=curr.get("language"),
        stars=int(curr.get("stars") or 0),
        forks=int(curr.get("forks") or 0),
        current_rank=curr_rank,
        previous_rank=prev_rank,
        rank_change=rank_change,
        previous_stars=prev.get("stars"),
        star_delta=star_delta,
        star_growth_percent=_growth_percent(star_delta, prev.get("stars") or 0),
        previous_forks=prev.get("forks"),
        fork_delta=fork_delta,
        fork_growth_percent=_growth_percent(fork_delta, prev.get("forks") or 0),
        current_trend_score=curr_ts,
        previous_trend_score=prev_ts,
        trend_score_change=ts_change,
        pushed_at=curr.get("pushed_at"),
        previous_pushed_at=prev.get("pushed_at"),
        emerging_score=emerging,
        history_available=True,
        comparison_window=window,
    )


def compare_window(
    *,
    settings: Settings,
    window: str = "7d",
    limit: int = 20,
) -> TrendResult:
    """Compare the latest snapshot against the window's historical snapshot."""
    window = validate_window(window)
    limit = validate_limit(limit)
    generated_at = _utc_now_iso()
    times = list_snapshot_times(settings)
    if not times:
        return TrendResult(
            window=window, generated_at=generated_at, has_history=False,
            history_reason="No snapshots captured yet. "
            "POST /api/v1/discover/snapshots to capture one.",
            repositories=[], total=0, limit=limit,
        )
    current_raw = times[-1]
    current_time = _parse_time(current_raw) or datetime.now(timezone.utc)
    current_rows = _rows_at(settings, current_raw)
    if len(times) < 2:
        repos = [_compare_pair(r, None, None, window) for r in current_rows]
        repos.sort(key=lambda r: (-(r.current_trend_score or 0.0), r.full_name))
        repos = repos[:limit]
        return TrendResult(
            window=window, generated_at=generated_at, has_history=False,
            history_reason="Only one snapshot exists; capture another after "
            "the window elapses to enable comparison.",
            current_snapshot_at=current_raw, previous_snapshot_at=None,
            repositories=repos, total=len(repos), limit=limit,
        )
    previous_raw = _find_previous_time(times, current_time, WINDOWS[window])
    if previous_raw is None:
        repos = [_compare_pair(r, None, None, window) for r in current_rows]
        repos.sort(key=lambda r: (-(r.current_trend_score or 0.0), r.full_name))
        repos = repos[:limit]
        return TrendResult(
            window=window, generated_at=generated_at, has_history=False,
            history_reason=f"No snapshot close enough to {window} ago "
            "(±20% tolerance). Capture snapshots regularly to build history.",
            current_snapshot_at=current_raw, previous_snapshot_at=None,
            repositories=repos, total=len(repos), limit=limit,
        )
    previous_rows = {r.get("full_name"): r for r in _rows_at(settings, previous_raw)}
    pairs = [(r, previous_rows.get(r.get("full_name"))) for r in current_rows]
    scores = emerging_scores(pairs, current_time)
    repos = [_compare_pair(curr, prev, scores.get(curr.get("full_name")), window)
             for curr, prev in pairs]
    # Growth-heavy ranking: history rows by emerging score first, then the
    # rest by popularity trend score. Deterministic tie-breaks throughout.
    with_history = sorted(
        [r for r in repos if r.history_available],
        key=lambda r: (-(r.emerging_score or 0.0),
                       -(r.current_trend_score or 0.0), r.full_name),
    )
    without_history = sorted(
        [r for r in repos if not r.history_available],
        key=lambda r: (-(r.current_trend_score or 0.0), r.full_name),
    )
    ordered = (with_history + without_history)[:limit]
    return TrendResult(
        window=window, generated_at=generated_at, has_history=True,
        history_reason=None,
        current_snapshot_at=current_raw, previous_snapshot_at=previous_raw,
        repositories=ordered, total=len(ordered), limit=limit,
    )


def get_repository_history(
    *,
    settings: Settings,
    owner: str,
    repo: str,
    window: str = "30d",
) -> RepositoryHistoryResult:
    """Stored snapshots for one repo plus a window comparison (if possible)."""
    from . import github as github_service

    window = validate_window(window)
    if not github_service.is_valid_slug(owner) or not github_service.is_valid_slug(repo):
        raise TrendValidationError("Invalid owner/repo slug.")
    full_name = f"{owner.lower()}/{repo.lower()}"
    current_raw: Optional[str] = None
    times = list_snapshot_times(settings)
    if times:
        current_raw = times[-1]
        since_dt = (_parse_time(current_raw) or datetime.now(timezone.utc)) \
            - timedelta(seconds=WINDOWS[window])
        rows = _rows_for_repo(settings, full_name, since_dt.isoformat())
    else:
        rows = []
    points = [_to_point(r) for r in rows]
    if len(points) < 2 or current_raw is None:
        return RepositoryHistoryResult(
            full_name=full_name, window=window, snapshots=points,
            total_snapshots=len(points), has_history=False, comparison=None,
        )
    current_time = _parse_time(current_raw) or datetime.now(timezone.utc)
    repo_times = [p.snapshot_at for p in points]
    previous_raw = _find_previous_time(repo_times, current_time, WINDOWS[window])
    if previous_raw is None:
        return RepositoryHistoryResult(
            full_name=full_name, window=window, snapshots=points,
            total_snapshots=len(points), has_history=False, comparison=None,
        )
    by_time = {p.snapshot_at: p for p in points}
    curr_row = by_time[current_raw] if current_raw in by_time else points[-1]
    prev_point = by_time[previous_raw]
    curr_dict = curr_row.model_dump()
    prev_dict = prev_point.model_dump()
    scores = emerging_scores([(curr_dict, prev_dict)], current_time)
    comparison = _compare_pair(
        curr_dict, prev_dict, scores.get(curr_dict.get("full_name") or ""), window)
    return RepositoryHistoryResult(
        full_name=full_name, window=window, snapshots=points,
        total_snapshots=len(points), has_history=True, comparison=comparison,
    )
