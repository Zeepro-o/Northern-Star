"""M7.1 — Evidence-Grounded Repository Architecture Graph.

Pipeline (no LLM — fully deterministic):

    Repository
        ↓
    Manifest (file inventory)
        ↓
    File classification (path/extension signals)
        ↓
    Directory hierarchy (contains edges)
        ↓
    Import extraction (Python AST + JS/TS relative imports)
        ↓
    Resolved in-repo relationships (imports/tests edges)
        ↓
    Evidence mapping (indexed chunks → [E#] citations)
        ↓
    ArchitectureResult

Core principle: "LLMs reason; evidence determines what can be claimed."
Structure and dependencies are discovered from the repository itself. When
classification is ambiguous we fall back to ``directory`` / ``module`` /
``unknown`` rather than guessing, and unresolved/external imports never
become nodes or edges.
"""

from __future__ import annotations

import ast
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from ..config import Settings
from ..models.schemas import (
    ArchitectureEdge,
    ArchitectureNode,
    ArchitectureResult,
    Citation,
)
from .indexing import connect_evidence_db, evidence_db_path, repo_is_indexed
from .ingestion import load_manifest
from .retrieval import RepoNotIndexedError

# ---------------------------------------------------------------------------
# Node / relationship vocabularies (service-emitted values; schema stays
# extensible with plain strings so the frontend can evolve independently).
# ---------------------------------------------------------------------------

NODE_TYPES = frozenset({
    "project", "directory", "module", "api", "service", "model",
    "database", "test", "config", "frontend", "backend", "utility",
    "unknown",
})

RELATIONSHIPS = frozenset({
    "contains", "imports", "depends_on", "calls",
    "tests", "configures", "reads_from", "writes_to",
})

DEFAULT_MAX_NODES = 200

# ---------------------------------------------------------------------------
# Deterministic classification signals
# ---------------------------------------------------------------------------

_TEST_DIR_NAMES = {"test", "tests", "__tests__", "testing", "spec", "e2e"}
_FRONTEND_DIR_NAMES = {"frontend", "client", "web", "public", "static", "assets",
                       "components", "pages", "app", "src"}
_BACKEND_DIR_NAMES = {"backend", "server", "api"}
_CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".ini", ".cfg", ".env"}
_CONFIG_BASENAMES = {
    "package.json", "pyproject.toml", "setup.py", "setup.cfg",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "makefile", "requirements.txt", "tsconfig.json", "vite.config.js",
    "vite.config.ts", "webpack.config.js", ".env.example",
}
_DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}

_JS_TS_EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}

# Ordered (segment, node-type) rules for source files — first match wins.
# Conservative: only well-established layout conventions.
_FILE_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("test", "test"),
    ("tests", "test"),
    ("__tests__", "test"),
    ("testing", "test"),
    ("api", "api"),
    ("routes", "api"),
    ("route", "api"),
    ("controllers", "api"),
    ("controller", "api"),
    ("views", "api"),
    ("endpoints", "api"),
    ("handlers", "api"),
    ("services", "service"),
    ("service", "service"),
    ("models", "model"),
    ("model", "model"),
    ("schemas", "model"),
    ("schema", "model"),
    ("entities", "model"),
    ("entity", "model"),
    ("migrations", "database"),
    ("migration", "database"),
    ("db", "database"),
    ("database", "database"),
    ("storage", "database"),
    ("components", "frontend"),
    ("pages", "frontend"),
    ("public", "frontend"),
    ("static", "frontend"),
    ("assets", "frontend"),
    ("utils", "utility"),
    ("util", "utility"),
    ("helpers", "utility"),
    ("helper", "utility"),
    ("common", "utility"),
    ("shared", "utility"),
    ("lib", "utility"),
    ("backend", "backend"),
    ("server", "backend"),
)


def _segments(posix_path: str) -> list[str]:
    return [s.lower() for s in PurePosixPath(posix_path).parts[:-1]]


def _basename(posix_path: str) -> str:
    return PurePosixPath(posix_path).name.lower()


def is_test_file(path: str) -> bool:
    """Conservative test-file detection across common conventions."""
    base = _basename(path)
    segs = _segments(path)
    if any(s in _TEST_DIR_NAMES for s in segs):
        return True
    if base.startswith("test_") or base.startswith("spec_"):
        return True
    for stem, suffixes in (
        ("_test", {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rb"}),
        (".test", {".js", ".jsx", ".ts", ".tsx"}),
        (".spec", {".js", ".jsx", ".ts", ".tsx"}),
    ):
        for suffix in suffixes:
            if base.endswith(stem + suffix):
                return True
    if base in {"conftest.py", "pytest.ini", "setup_tests.py"}:
        return True
    return False


def is_config_file(path: str, manifest_config_files: set[str] | None = None) -> bool:
    if manifest_config_files and path in manifest_config_files:
        return True
    base = _basename(path)
    if base in _CONFIG_BASENAMES:
        return True
    suffix = PurePosixPath(base).suffix
    if suffix in _CONFIG_EXTENSIONS:
        return True
    if base.startswith("requirements") and base.endswith(".txt"):
        return True
    return False


def is_doc_file(path: str) -> bool:
    return PurePosixPath(_basename(path)).suffix in _DOC_EXTENSIONS


def classify_file(path: str, manifest_config_files: set[str] | None = None) -> str:
    """Classify a repository-relative file path. Ambiguous → module/unknown."""
    if is_test_file(path):
        return "test"
    if is_config_file(path, manifest_config_files):
        return "config"
    suffix = PurePosixPath(_basename(path)).suffix
    if suffix in _JS_TS_EXTENSIONS:
        segs = _segments(path)
        for segment, ntype in _FILE_TYPE_RULES:
            if segment in segs:
                return ntype
        base = _basename(path)
        if base.endswith((".tsx", ".jsx", ".vue", ".svelte")):
            return "frontend"
        return "module"
    if suffix == ".sql":
        return "database"
    if suffix in {".py", ".java", ".go", ".rb", ".rs", ".php", ".cs", ".c",
                  ".cpp", ".h", ".hpp", ".swift", ".kt", ".scala", ".css",
                  ".html", ".vue", ".svelte"}:
        segs = _segments(path)
        for segment, ntype in _FILE_TYPE_RULES:
            if segment in segs:
                return ntype
        return "module"
    if is_doc_file(path):
        return "unknown"
    return "unknown"


def classify_directory(dirname: str, has_package_json: bool = False) -> str:
    """Classify a directory by name. Ambiguous → directory."""
    name = dirname.lower()
    if name in _TEST_DIR_NAMES:
        return "test"
    if name in _BACKEND_DIR_NAMES:
        return "backend"
    if name in {"frontend", "client", "web"} or (
        name in {"public", "static", "components", "pages"} and has_package_json
    ):
        return "frontend"
    return "directory"


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawImport:
    """One import statement: raw module specifier + 1-indexed line number."""

    specifier: str  # e.g. "app.services.judge" or "./foo"
    line: int
    relative_level: int = 0  # Python relative-import dots (0 = absolute)


def extract_python_imports(source: str) -> list[RawImport]:
    """Extract imports from Python source via AST (regex fallback)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return _extract_python_imports_regex(source)
    out: list[RawImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(RawImport(specifier=alias.name, line=node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None and (node.level or 0) == 0:
                continue
            out.append(RawImport(
                specifier=node.module or "",
                line=node.lineno,
                relative_level=node.level or 0,
            ))
    return out


def _extract_python_imports_regex(source: str) -> list[RawImport]:
    out: list[RawImport] = []
    for i, line in enumerate(source.splitlines(), start=1):
        m = re.match(r"\s*import\s+([a-zA-Z_][\w\.]*)", line)
        if m:
            out.append(RawImport(specifier=m.group(1), line=i))
            continue
        m = re.match(r"\s*from\s+(\.*)\s*([a-zA-Z_][\w\.]*)?\s+import\s+", line)
        if m:
            dots, mod = m.group(1), m.group(2) or ""
            out.append(RawImport(specifier=mod, line=i, relative_level=len(dots)))
    return out


_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+(?:[^'"]*?\s+from\s+)?|require\s*\(\s*|import\s*\(\s*)"""
    r"""['"]([^'"]+)['"]""",
)


def extract_js_imports(source: str) -> list[RawImport]:
    """Extract relative (``./x``) imports from JS/TS source. Line-aware."""
    out: list[RawImport] = []
    for m in _JS_IMPORT_RE.finditer(source):
        spec = m.group(1)
        if spec.startswith("."):
            line = source.count("\n", 0, m.start()) + 1
            out.append(RawImport(specifier=spec, line=line))
    return out


# ---------------------------------------------------------------------------
# Module resolution (in-repo only)
# ---------------------------------------------------------------------------

def build_module_index(repo_files: set[str]) -> dict[str, str]:
    """Map dotted module paths → repo-relative file paths for .py files.

    Indexes each file relative to the repo root and to ``src/`` (the common
    src-layout), so ``flask.app`` resolves to ``src/flask/app.py``.
    ``__init__.py`` maps to its package path.
    """
    index: dict[str, str] = {}
    for path in repo_files:
        if not path.endswith(".py"):
            continue
        for root in ("", "src/"):
            if root and not path.startswith(root):
                continue
            rel = path[len(root):] if root else path
            parts = PurePosixPath(rel).parts
            if parts[-1] == "__init__.py":
                dotted = ".".join(parts[:-1])
            else:
                dotted = ".".join(parts[:-1] + (parts[-1][:-3],))
            if dotted and dotted not in index:
                index[dotted] = path
    return index


def resolve_python_import(
    importer_path: str,
    raw: RawImport,
    module_index: dict[str, str],
) -> Optional[str]:
    """Resolve an import to a repo-relative file path, or None."""
    if raw.relative_level:
        # Relative: anchor at the importer's package.
        parts = PurePosixPath(importer_path).parts
        package = list(parts[:-1])
        if parts[-1] != "__init__.py":
            pass  # package dir stays as the file's directory
        up = raw.relative_level - 1
        if up > len(package):
            return None
        base = package[: len(package) - up] if up else package
        dotted = ".".join(base + ([raw.specifier] if raw.specifier else []))
        dotted = dotted.strip(".")
        if not dotted:
            return None
        return module_index.get(dotted)
    if not raw.specifier:
        return None
    return module_index.get(raw.specifier)


_JS_RESOLVE_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json")


def resolve_js_import(
    importer_path: str,
    raw: RawImport,
    repo_files: set[str],
) -> Optional[str]:
    """Resolve a relative JS/TS specifier to a repo file, or None."""
    base_dir = PurePosixPath(importer_path).parent
    target = (base_dir / raw.specifier).as_posix()
    # Normalise ./ and ../ segments without touching the filesystem.
    parts: list[str] = []
    for seg in PurePosixPath(target).parts:
        if seg == "..":
            if parts:
                parts.pop()
        elif seg != ".":
            parts.append(seg)
    norm = "/".join(parts)
    candidates = [norm]
    candidates += [norm + ext for ext in _JS_RESOLVE_EXTENSIONS]
    candidates += [f"{norm}/index{ext}" for ext in _JS_RESOLVE_EXTENSIONS]
    for cand in candidates:
        if cand in repo_files:
            return cand
    return None


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _node_id_for_file(path: str) -> str:
    return f"file:{path}"


def _node_id_for_dir(path: str) -> str:
    return f"dir:{path}" if path not in (".", "") else "project"


def _label_for_path(path: str) -> str:
    return PurePosixPath(path).name or path


def build_architecture_graph(
    repo_id: str,
    *,
    settings: Settings,
    max_nodes: Optional[int] = None,
) -> ArchitectureResult:
    """Build a deterministic architecture graph for an indexed repository.

    Raises RepoNotIndexedError when the repo has no manifest or index.
    """
    owner, _, repo = repo_id.partition("/")
    if not owner or not repo:
        raise RepoNotIndexedError(f"Invalid repo id: {repo_id!r}.")
    repo_dir = settings.storage_root / owner.lower() / repo.lower()
    raw_manifest = load_manifest(repo_dir, settings.manifest_filename)
    if raw_manifest is None:
        raise RepoNotIndexedError(
            f"Repository '{repo_id}' has not been ingested yet."
        )
    db_path = evidence_db_path(settings.storage_root, settings.db_filename)
    if not repo_is_indexed(db_path, repo_id):
        raise RepoNotIndexedError(
            f"Repository '{repo_id}' has not been indexed yet."
        )

    checkout = repo_dir / "checkout"
    inventory = raw_manifest.get("file_inventory", []) if isinstance(raw_manifest, dict) else []
    manifest_config_files = set(raw_manifest.get("config_files", []) or [])
    repo_files = sorted({
        e["path"] for e in inventory
        if isinstance(e, dict) and isinstance(e.get("path"), str)
    })

    limit = int(max_nodes) if max_nodes is not None else DEFAULT_MAX_NODES
    limit = max(2, min(limit, 2000))

    has_package_json = "package.json" in repo_files

    # -- Directories present in the inventory -------------------------------
    dir_paths: set[str] = set()
    for path in repo_files:
        parent = str(PurePosixPath(path).parent)
        while parent not in (".", ""):
            dir_paths.add(parent)
            parent = str(PurePosixPath(parent).parent)

    # -- File nodes (source + config + test only; docs/data stay grouped) ---
    file_candidates: list[tuple[str, str]] = []  # (path, type)
    for path in repo_files:
        ntype = classify_file(path, manifest_config_files)
        if ntype in ("module", "api", "service", "model", "database",
                     "test", "config", "frontend", "backend", "utility"):
            file_candidates.append((path, ntype))
    file_candidates.sort(key=lambda t: t[0])

    # Enforce the node cap across directories + files. Directories are
    # usually few, but under a tight cap the deepest ones are dropped first
    # and orphaned children re-home to the nearest surviving ancestor.
    dir_list = sorted(dir_paths)
    file_candidates.sort(key=lambda t: t[0])
    keep_dirs: set[str] = set(dir_list)
    if 1 + len(dir_list) + len(file_candidates) > limit:
        # Trim deepest directories first (deterministic: depth desc, path).
        by_depth = sorted(dir_list,
                          key=lambda d: (-len(PurePosixPath(d).parts), d))
        # Drop deepest directories until everything fits.
        while 1 + len(keep_dirs) + len(file_candidates) > limit and by_depth:
            keep_dirs.discard(by_depth.pop(0))
        file_candidates = file_candidates[: max(0, limit - 1 - len(keep_dirs))]
    dir_list = sorted(keep_dirs)
    graphed_files = {p for p, _ in file_candidates}

    def _nearest_kept_ancestor(path: str) -> str:
        """Nearest surviving ancestor dir id, else project."""
        parent = str(PurePosixPath(path).parent)
        while parent not in (".", ""):
            if parent in keep_dirs:
                return _node_id_for_dir(parent)
            parent = str(PurePosixPath(parent).parent)
        return "project"

    nodes: dict[str, ArchitectureNode] = {}
    nodes["project"] = ArchitectureNode(
        id="project", label=repo_id, type="project",
        files=[], description=f"Repository {repo_id}",
    )
    for d in dir_list:
        dtype = classify_directory(PurePosixPath(d).name, has_package_json)
        nodes[_node_id_for_dir(d)] = ArchitectureNode(
            id=_node_id_for_dir(d), label=_label_for_path(d),
            type=dtype, files=[],
        )
    for path, ntype in file_candidates:
        nodes[_node_id_for_file(path)] = ArchitectureNode(
            id=_node_id_for_file(path), label=_label_for_path(path),
            type=ntype, files=[path],
        )

    # Directory membership (files field = direct children present in repo).
    children_by_dir: dict[str, list[str]] = {}
    for path in repo_files:
        parent = str(PurePosixPath(path).parent)
        children_by_dir.setdefault(parent, []).append(path)
    nodes["project"].files = sorted(
        p for p in repo_files if "/" not in p)[:50]
    for d in dir_list:
        node = nodes[_node_id_for_dir(d)]
        node.files = sorted(children_by_dir.get(d, []))[:100]

    # -- Contains edges ------------------------------------------------------
    edges: dict[tuple[str, str, str], ArchitectureEdge] = {}

    def add_edge(source: str, target: str, relationship: str) -> None:
        if source not in nodes or target not in nodes or source == target:
            return
        key = (source, target, relationship)
        if key not in edges:
            edges[key] = ArchitectureEdge(
                source=source, target=target, relationship=relationship,
                evidence_ids=[],
            )

    for d in dir_list:
        add_edge(_nearest_kept_ancestor(d), _node_id_for_dir(d), "contains")
    for path in graphed_files:
        add_edge(_nearest_kept_ancestor(path), _node_id_for_file(path), "contains")
    # -- Import edges (in-repo only) -----------------------------------------
    module_index = build_module_index(set(repo_files))
    import_evidence: dict[tuple[str, str, str], int] = {}  # edge key → line
    for path in sorted(graphed_files):
        suffix = PurePosixPath(path).suffix
        full = checkout / path
        if not full.is_file():
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if suffix == ".py":
            raws = extract_python_imports(text)
            resolver = lambda r: resolve_python_import(path, r, module_index)
        elif suffix in _JS_TS_EXTENSIONS:
            raws = extract_js_imports(text)
            resolver = lambda r: resolve_js_import(path, r, set(repo_files))
        else:
            continue
        seen_targets: set[str] = set()
        for raw in raws:
            target = resolver(raw)
            if target is None or target not in graphed_files or target == path:
                continue
            if target in seen_targets:
                continue
            seen_targets.add(target)
            source_type = nodes[_node_id_for_file(path)].type
            rel = "tests" if source_type == "test" else "imports"
            key = (_node_id_for_file(path), _node_id_for_file(target), rel)
            if key not in edges:
                edges[key] = ArchitectureEdge(
                    source=key[0], target=key[1], relationship=rel,
                    evidence_ids=[],
                )
            import_evidence.setdefault(key, raw.line)

    # -- Evidence mapping (indexed chunks → [E#]) ------------------------------
    chunks = _chunks_for_repo(db_path, repo_id)
    # Deterministic E-numbering: file path, then start line.
    chunks.sort(key=lambda r: (r[0], r[1]))
    labels: list[tuple[str, int, int, str | None, str]] = []  # (file,start,end,lang,content)
    for row in chunks:
        labels.append(row)
    by_file: dict[str, list[int]] = {}
    for i, (fpath, _s, _e, _l, _c) in enumerate(labels):
        by_file.setdefault(fpath, []).append(i)

    def chunk_for_line(fpath: str, line: int) -> Optional[int]:
        for i in by_file.get(fpath, []):
            _f, s, e, _l, _c = labels[i]
            if s <= line <= e:
                return i
        if by_file.get(fpath):
            return by_file[fpath][0]
        return None

    referenced: set[int] = set()
    for path in graphed_files:
        idxs = by_file.get(path, [])[:2]
        if idxs:
            nodes[_node_id_for_file(path)].evidence_ids = [f"E{i + 1}" for i in idxs]
            referenced.update(idxs)
    for key, line in import_evidence.items():
        if key not in edges:
            continue
        src_file = edges[key].source[len("file:"):]
        idx = chunk_for_line(src_file, line)
        if idx is not None:
            edges[key].evidence_ids = [f"E{idx + 1}"]
            referenced.add(idx)

    evidence_citations = []
    for i in sorted(referenced):
        fpath, s, e, lang, content = labels[i]
        evidence_citations.append(Citation(
            id=f"E{i + 1}", file_path=fpath, start_line=s, end_line=e,
            language=lang, content=content,
        ))

    node_list = sorted(nodes.values(), key=lambda n: n.id)
    edge_list = sorted(edges.values(),
                       key=lambda e: (e.source, e.target, e.relationship))

    n_source = sum(1 for _, t in file_candidates if t != "test" and t != "config")
    n_tests = sum(1 for _, t in file_candidates if t == "test")
    n_config = sum(1 for _, t in file_candidates if t == "config")
    n_imports = sum(1 for e in edge_list if e.relationship in ("imports", "tests"))
    summary = (
        f"Repository {repo_id} contains {n_source} source modules across "
        f"{len(dir_list)} directories, with {n_imports} detected internal "
        f"dependencies, {n_tests} test modules and {n_config} configuration files."
    )

    return ArchitectureResult(
        repo_id=repo_id,
        root="project",
        nodes=node_list,
        edges=edge_list,
        total_nodes=len(node_list),
        total_edges=len(edge_list),
        evidence_citations=evidence_citations,
        summary=summary,
    )


def _chunks_for_repo(
    db_path, repo_id: str
) -> list[tuple[str, int, int, str | None, str]]:
    """Return (file_path, start_line, end_line, language, content) rows."""
    conn: sqlite3.Connection = connect_evidence_db(db_path)
    try:
        rows = conn.execute(
            "SELECT file_path, start_line, end_line, language, content"
            " FROM chunks WHERE repo_id = ?",
            (repo_id,),
        ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]
    finally:
        conn.close()
