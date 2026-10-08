"""Deterministic file, language and framework detection.

No LLM, no external services: everything here is derived from filenames,
extensions, and a small content peek. The output is stable and unit-testable.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..models.schemas import Claim, FileKind, FrameworkInfo

# ---------------------------------------------------------------------------
# Directories that are excluded from analysis entirely (not descended into).
# ---------------------------------------------------------------------------

IGNORED_DIR_BASENAMES: dict[str, str] = {
    ".git": "version control metadata",
    "node_modules": "vendored dependencies",
    "vendor": "vendored dependencies",
    "Pods": "CocoaPods dependencies",
    "dist": "build output",
    "build": "build output",
    "out": "build output",
    "target": "build output (Rust/Cargo)",
    "obj": "build output (C/C++)",
    ".next": "build output (Next.js)",
    ".nuxt": "build output (Nuxt)",
    ".svelte-kit": "build output (SvelteKit)",
    ".turbo": "Turborepo cache",
    ".parcel-cache": "Parcel cache",
    "coverage": "test coverage output",
    "__pycache__": "Python bytecode cache",
    ".venv": "virtual environment",
    "venv": "virtual environment",
    "env": "virtual environment",
    ".tox": "tox test environments",
    ".mypy_cache": "type-checker cache",
    ".pytest_cache": "test cache",
    ".ruff_cache": "linter cache",
    ".gradle": "Gradle cache",
    ".cache": "cache",
    ".idea": "JetBrains IDE metadata",
    ".vscode": "editor metadata",
}


def _dir_skip_reason(dirname: str) -> Optional[str]:
    """Return why a directory should be skipped, or None to keep it."""
    reason = IGNORED_DIR_BASENAMES.get(dirname)
    if reason:
        return reason
    if dirname.startswith(".") and dirname not in {".github"}:
        return "hidden directory"
    return None


# Files excluded from the inventory (secrets / OS junk).
def _file_skip_reason(filename: str) -> Optional[str]:
    lower = filename.lower()
    if filename == ".env" or filename.startswith(".env."):
        return "potential secrets file"
    if lower in {".ds_store", "thumbs.db", ".gitmodules"}:
        return "dotfile / OS metadata"
    return None


# ---------------------------------------------------------------------------
# Extension and filename fingerprints.
# ---------------------------------------------------------------------------

EXTENSION_LANGUAGE: dict[str, str] = {
    ".py": "Python", ".pyi": "Python",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".mts": "TypeScript", ".cts": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".html": "HTML", ".htm": "HTML",
    ".css": "CSS", ".scss": "SCSS", ".sass": "Sass", ".less": "Less",
    ".json": "JSON",
    ".md": "Markdown", ".markdown": "Markdown",
    ".rst": "reStructuredText",
    ".yaml": "YAML", ".yml": "YAML",
    ".toml": "TOML",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin", ".kts": "Kotlin",
    ".c": "C", ".h": "C",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++", ".hh": "C++", ".hxx": "C++",
    ".cs": "C#",
    ".php": "PHP",
    ".rb": "Ruby",
    ".swift": "Swift",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
    ".sql": "SQL",
    ".graphql": "GraphQL", ".gql": "GraphQL",
    ".proto": "Protocol Buffers",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".sol": "Solidity",
    ".ex": "Elixir", ".exs": "Elixir",
    ".erl": "Erlang",
    ".scala": "Scala",
    ".dart": "Dart",
    ".lua": "Lua",
    ".r": "R",
    ".jl": "Julia",
    ".pl": "Perl",
    ".nim": "Nim",
    ".zig": "Zig",
    ".hs": "Haskell",
    ".clj": "Clojure", ".cljs": "Clojure",
    ".fs": "F#", ".fsx": "F#",
    ".vb": "Visual Basic",
    ".asm": "Assembly", ".s": "Assembly",
    ".objc": "Objective-C", ".m": "Objective-C",
    ".xml": "XML", ".xsl": "XML",
    ".ini": "INI", ".cfg": "INI", ".conf": "INI",
    ".csv": "CSV", ".tsv": "TSV",
    ".svg": "SVG",
    ".tex": "TeX",
}

FILENAME_LANGUAGE: dict[str, str] = {
    "dockerfile": "Dockerfile",
    "makefile": "Makefile",
    "cmakelists.txt": "CMake",
    "gemfile": "Ruby",
    "rakefile": "Ruby",
    "justfile": "Justfile",
}

BINARY_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".avif",
    ".pdf", ".zip", ".gz", ".bz2", ".xz", ".tar", ".7z", ".rar", ".zst",
    ".so", ".dylib", ".dll", ".o", ".a", ".obj", ".class", ".jar", ".war",
    ".exe", ".bin", ".wasm", ".deb", ".rpm", ".apk", ".aab",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".pyc", ".pyo", ".pyd",
    ".sqlite", ".db", ".sqlite3",
    ".xlsx", ".xls", ".docx", ".doc", ".pptx", ".odt",
    ".dat", ".data", ".pkl", ".h5", ".hdf5",
}

_DOC_NAMES = {"license", "copying", "changelog", "changes", "authors", "notice", "contributing", "security"}

CONFIG_FILENAMES: set[str] = {
    ".gitignore", ".dockerignore", ".editorconfig", ".npmrc", ".babelrc",
    ".prettierrc", ".prettierrc.json", ".eslintignore", ".eslintrc",
}

CONFIG_EXTENSIONS: set[str] = {
    ".gradle", ".gradle.kts", ".lock", ".properties", ".bzl", ".plist",
}

_DOC_LANGS = {"Markdown", "reStructuredText", "TeX"}
_CONFIG_LANGS = {"JSON", "YAML", "TOML", "XML", "INI"}
_DATA_LANGS = {"CSV", "TSV"}


def detect_language(path: Path) -> Optional[str]:
    """Best-effort language from filename/extension. ``None`` if unknown."""
    return EXTENSION_LANGUAGE.get(path.suffix.lower()) or FILENAME_LANGUAGE.get(path.name.lower())


def _looks_binary(path: Path, sniff_bytes: int) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(sniff_bytes)
    except OSError:
        return False
    return b"\x00" in head


def classify_file(path: Path, sniff_bytes: int = 1024) -> tuple[Optional[str], FileKind]:
    """Classify a file into (language, category). Pure and deterministic."""
    name = path.name
    name_lower = name.lower()
    ext = path.suffix.lower()

    # Documentation by name (README, LICENSE, ...) wins over extension.
    if name_lower.startswith("readme") or name_lower.rstrip(".") in _DOC_NAMES:
        lang = EXTENSION_LANGUAGE.get(ext) or ("Text" if not ext else None)
        return lang, FileKind.DOCUMENTATION

    if name_lower in CONFIG_FILENAMES:
        return detect_language(path), FileKind.CONFIG

    if name_lower in FILENAME_LANGUAGE:
        return FILENAME_LANGUAGE[name_lower], FileKind.SOURCE

    if ext in BINARY_EXTENSIONS:
        return None, FileKind.BINARY

    lang = EXTENSION_LANGUAGE.get(ext)
    if lang in _DOC_LANGS:
        return lang, FileKind.DOCUMENTATION
    if lang in _CONFIG_LANGS:
        return lang, FileKind.CONFIG
    if lang in _DATA_LANGS:
        return lang, FileKind.DATA
    if lang:
        return lang, FileKind.SOURCE

    if ext in CONFIG_EXTENSIONS:
        return lang, FileKind.CONFIG

    if _looks_binary(path, sniff_bytes):
        return None, FileKind.BINARY

    return None, FileKind.OTHER


# ---------------------------------------------------------------------------
# Framework / package-manager detection — deterministic config-file signals.
# ---------------------------------------------------------------------------

def _dep_name(raw: str) -> str:
    """Extract a canonical dependency name from a PEP 508-ish spec string."""
    raw = raw.strip()
    m = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?", raw)
    if m:
        return m.group(1)
    return raw.split("==")[0].split(">=")[0].split("<")[0].strip().lower()


def _match(names: Iterable[str], mapping: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    hits: list[tuple[str, str]] = []
    for raw in names:
        key = _dep_name(raw).lower()
        hit = mapping.get(key)
        if hit and hit[0] not in seen:
            seen.add(hit[0])
            hits.append(hit)
    return hits


NPM_MAP: dict[str, tuple[str, str]] = {
    "react": ("React", "framework"), "react-dom": ("React", "framework"),
    "next": ("Next.js", "framework"),
    "vue": ("Vue", "framework"), "nuxt": ("Nuxt", "framework"),
    "svelte": ("Svelte", "framework"), "@sveltejs/kit": ("SvelteKit", "framework"),
    "angular": ("Angular", "framework"), "@angular/core": ("Angular", "framework"),
    "express": ("Express.js", "framework"), "fastify": ("Fastify", "framework"),
    "koa": ("Koa", "framework"), "hapi": ("Hapi", "framework"),
    "@nestjs/core": ("NestJS", "framework"),
    "astro": ("Astro", "framework"), "@astrojs/react": ("Astro", "framework"),
    "gatsby": ("Gatsby", "framework"), "@remix-run/react": ("Remix", "framework"),
    "electron": ("Electron", "framework"),
    "socket.io": ("Socket.IO", "library"),
    "vite": ("Vite", "tooling"), "webpack": ("Webpack", "tooling"),
    "rollup": ("Rollup", "tooling"), "esbuild": ("esbuild", "tooling"),
    "tailwindcss": ("Tailwind CSS", "framework"),
    "typescript": ("TypeScript", "tooling"),
    "jest": ("Jest", "tooling"), "vitest": ("Vitest", "tooling"),
    "mocha": ("Mocha", "tooling"), "@playwright/test": ("Playwright", "tooling"),
}

PYTHON_MAP: dict[str, tuple[str, str]] = {
    "fastapi": ("FastAPI", "framework"), "flask": ("Flask", "framework"),
    "django": ("Django", "framework"), "starlette": ("Starlette", "framework"),
    "tornado": ("Tornado", "framework"), "aiohttp": ("aiohttp", "framework"),
    "celery": ("Celery", "library"), "sqlalchemy": ("SQLAlchemy", "library"),
    "alembic": ("Alembic", "tooling"),
    "transformers": ("Hugging Face Transformers", "library"),
    "langchain": ("LangChain", "framework"), "llama-index": ("LlamaIndex", "framework"),
    "dspy": ("DSPy", "framework"),
    "torch": ("PyTorch", "library"), "tensorflow": ("TensorFlow", "library"),
    "pandas": ("Pandas", "library"), "numpy": ("NumPy", "library"),
    "scikit-learn": ("scikit-learn", "library"),
    "pytest": ("pytest", "tooling"), "ruff": ("Ruff", "tooling"),
    "mypy": ("Mypy", "tooling"),
}

CARGO_MAP: dict[str, tuple[str, str]] = {
    "axum": ("Axum", "framework"), "actix-web": ("Actix Web", "framework"),
    "rocket": ("Rocket", "framework"), "warp": ("Warp", "framework"), "tide": ("Tide", "framework"),
    "tokio": ("Tokio", "runtime"), "async-std": ("async-std", "runtime"),
    "tauri": ("Tauri", "framework"), "bevy": ("Bevy", "game engine"),
    "serde": ("Serde", "library"), "sqlx": ("SQLx", "library"),
    "diesel": ("Diesel", "library"), "clap": ("clap", "library"),
    "reqwest": ("reqwest", "library"),
}

GO_MODULE_MAP: list[tuple[str, str, str]] = [
    ("github.com/gin-gonic/gin", "Gin", "framework"),
    ("github.com/labstack/echo", "Echo", "framework"),
    ("github.com/go-chi/chi", "Chi", "framework"),
    ("github.com/gofiber/fiber", "Fiber", "framework"),
    ("github.com/gorilla/mux", "Gorilla Mux", "framework"),
    ("google.golang.org/grpc", "gRPC", "framework"),
    ("gorm.io/gorm", "GORM", "library"),
    ("github.com/spf13/cobra", "Cobra", "library"),
]

MAVEN_KEYWORDS: dict[str, tuple[str, str]] = {
    "spring-boot": ("Spring Boot", "framework"),
    "quarkus": ("Quarkus", "framework"),
    "micronaut": ("Micronaut", "framework"),
    "vertx": ("Vert.x", "framework"),
    "javafx": ("JavaFX", "framework"),
    "hibernate": ("Hibernate", "library"),
    "junit": ("JUnit", "tooling"),
}

RUBY_MAP: dict[str, tuple[str, str]] = {
    "rails": ("Ruby on Rails", "framework"), "sinatra": ("Sinatra", "framework"),
    "grape": ("Grape", "framework"), "hanami": ("Hanami", "framework"),
    "rack": ("Rack", "framework"), "sidekiq": ("Sidekiq", "library"),
}

PHP_MAP: dict[str, tuple[str, str]] = {
    "laravel/framework": ("Laravel", "framework"), "laravel/laravel": ("Laravel", "framework"),
    "cakephp/cakephp": ("CakePHP", "framework"), "codeigniter4/framework": ("CodeIgniter", "framework"),
    "laminas/laminas-mvc": ("Laminas MVC", "framework"),
}


def _detect_npm(content: str, source: str) -> list[tuple[str, str]]:
    try:
        data = json.loads(content)
    except ValueError:
        return []
    deps: list[str] = []
    for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        sec = data.get(section) or {}
        if isinstance(sec, dict):
            deps.extend(sec.keys())
    return _match(deps, NPM_MAP)


def _detect_python_toml(content: str, source: str) -> list[tuple[str, str]]:
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return []
    deps: list[str] = []
    project = data.get("project")
    if isinstance(project, dict):
        deps += [d for d in project.get("dependencies", []) if isinstance(d, str)]
        for opt_deps in (project.get("optional-dependencies") or {}).values():
            if isinstance(opt_deps, list):
                deps += [d for d in opt_deps if isinstance(d, str)]
    tool = data.get("tool")
    if isinstance(tool, dict) and isinstance(tool.get("poetry"), dict):
        pd = tool["poetry"].get("dependencies") or {}
        if isinstance(pd, dict):
            deps.extend(pd.keys())
    return _match(deps, PYTHON_MAP)


def _detect_requirements(content: str, source: str) -> list[tuple[str, str]]:
    deps = [
        line for line in content.splitlines()
        if line and not line.lstrip().startswith(("#", "-"))
    ]
    return _match(deps, PYTHON_MAP)


def _detect_cargo(content: str, source: str) -> list[tuple[str, str]]:
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return []
    deps: dict[str, object] = {}
    for key in ("dependencies", "dev-dependencies", "build-dependencies"):
        value = data.get(key)
        if isinstance(value, dict):
            deps.update(value)
    return _match(deps.keys(), CARGO_MAP)


def _detect_go(content: str, source: str) -> list[tuple[str, str]]:
    hits = [(name, kind) for needle, name, kind in GO_MODULE_MAP if needle in content]
    return hits


def _detect_maven(content: str, source: str) -> list[tuple[str, str]]:
    lower = content.lower()
    return [(name, kind) for kw, (name, kind) in MAVEN_KEYWORDS.items() if kw in lower]


def _detect_gemfile(content: str, source: str) -> list[tuple[str, str]]:
    gems = re.findall(r"^\s*gem\s+[\"']([^\"']+)[\"']", content, re.MULTILINE)
    return _match(gems, RUBY_MAP)


def _detect_composer(content: str, source: str) -> list[tuple[str, str]]:
    try:
        data = json.loads(content)
    except ValueError:
        return []
    req = data.get("require") or {}
    if not isinstance(req, dict):
        return []
    hits: list[tuple[str, str]] = []
    for dep in req.keys():
        if dep.startswith("symfony/"):
            hits.append(("Symfony", "framework"))
        else:
            hit = PHP_MAP.get(dep)
            if hit:
                hits.append(hit)
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, kind in hits:
        if name not in seen:
            seen.add(name)
            deduped.append((name, kind))
    return deduped


def _detect_dart(content: str, source: str) -> list[tuple[str, str]]:
    if "flutter" in content.lower():
        return [("Flutter", "framework")]
    return []


_FRAMEWORK_HANDLERS: dict[str, object] = {
    "package.json": _detect_npm,
    "pyproject.toml": _detect_python_toml,
    "requirements.txt": _detect_requirements,
    "Cargo.toml": _detect_cargo,
    "go.mod": _detect_go,
    "pom.xml": _detect_maven,
    "build.gradle": _detect_maven,
    "build.gradle.kts": _detect_maven,
    "Gemfile": _detect_gemfile,
    "composer.json": _detect_composer,
    "pubspec.yaml": _detect_dart,
}

# Frameworks inferred purely from a filename being present (no content parse).
_PRESENCE_SIGNALS: dict[str, tuple[str, str]] = {
    "dockerfile": ("Docker", "tooling"),
    "docker-compose.yml": ("Docker Compose", "tooling"),
    "docker-compose.yaml": ("Docker Compose", "tooling"),
    "makefile": ("Make", "tooling"),
}

# File-name → framework, useful when the package-manifest handler already ran
# (tsconfig.json without package.json, etc.). Presence-only.
_FILENAME_SIGNALS: dict[str, tuple[str, str]] = {
    "tsconfig.json": ("TypeScript", "tooling"),
    "next.config.js": ("Next.js", "framework"),
    "next.config.mjs": ("Next.js", "framework"),
    "next.config.ts": ("Next.js", "framework"),
    "vite.config.ts": ("Vite", "tooling"),
    "vite.config.js": ("Vite", "tooling"),
    "vite.config.mjs": ("Vite", "tooling"),
    "astro.config.mjs": ("Astro", "framework"),
    "astro.config.ts": ("Astro", "framework"),
    "svelte.config.js": ("SvelteKit", "framework"),
    "svelte.config.ts": ("SvelteKit", "framework"),
    "tailwind.config.js": ("Tailwind CSS", "framework"),
    "tailwind.config.ts": ("Tailwind CSS", "framework"),
    "docusaurus.config.js": ("Docusaurus", "framework"),
    "docusaurus.config.ts": ("Docusaurus", "framework"),
    "manage.py": ("Django", "framework"),
    "alembic.ini": ("Alembic", "tooling"),
}

_PM_ORDER: list[tuple[str, str]] = [
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("package-lock.json", "npm"),
    ("uv.lock", "uv"),
    ("poetry.lock", "poetry"),
    ("Pipfile.lock", "pipenv"),
    ("Cargo.lock", "cargo"),
    ("go.sum", "go modules"),
    ("requirements.txt", "pip"),
]


def detect_frameworks(
    config_texts: dict[str, str], entry_paths: set[str]
) -> tuple[list[FrameworkInfo], Optional[str]]:
    """Return (frameworks, package_manager) from deterministic file signals.

    ``config_texts`` maps relative config-file paths to their (small) content.
    ``entry_paths`` is the set of every discovered file path, used for
    presence-only signals.
    """
    found: dict[tuple[str, str], FrameworkInfo] = {}

    def add(name: str, kind: str, source: str) -> None:
        key = (name, kind)
        if key not in found:
            found[key] = FrameworkInfo(name=name, kind=kind, source=source)

    package_manager: Optional[str] = None
    for manifest, manager in _PM_ORDER:
        if manifest in entry_paths:
            package_manager = manager
            break
    if package_manager is None and "package.json" in config_texts:
        package_manager = "npm"
    def _has_tool_table(content: str, name: str) -> bool:
        # Matches both "[tool.name]" and "[tool.name.dependencies]".
        return re.search(rf"^\[tool\.{name}(?:\.|\])", content, re.MULTILINE) is not None

    if package_manager is None and "pyproject.toml" in config_texts:
        content = config_texts["pyproject.toml"]
        if _has_tool_table(content, "uv"):
            package_manager = "uv"
        elif _has_tool_table(content, "poetry"):
            package_manager = "poetry"
        else:
            package_manager = "pip"

    for path, content in sorted(config_texts.items()):
        handler = _FRAMEWORK_HANDLERS.get(Path(path).name)
        if handler is None:
            continue
        for name, kind in handler(content, path):
            add(name, kind, path)

    paths_lower = {p.lower() for p in entry_paths}
    for basename, signal in _PRESENCE_SIGNALS.items():
        if Path(basename).name in paths_lower or basename in entry_paths:
            name, kind = signal
            add(name, kind, basename)
    for filename, signal in _FILENAME_SIGNALS.items():
        if filename in entry_paths:
            name, kind = signal
            add(name, kind, filename)
    if any(p.startswith(".github/workflows/") for p in entry_paths):
        add("GitHub Actions", "tooling", ".github/workflows/")

    return list(found.values()), package_manager


# ---------------------------------------------------------------------------
# README claims — recorded as claims, not as verified facts.
# ---------------------------------------------------------------------------

README_CLAIM_KEYWORDS: list[str] = [
    "machine learning",
    "artificial intelligence",
    "deep learning",
    "neural network",
    "nlp",
    "natural language processing",
    "large language model",
    "llm",
    "real-time",
    "scalable",
    "high performance",
    "performant",
    "distributed",
    "decentralized",
    "microservices",
    "cloud-native",
    "serverless",
    "kubernetes",
    "secure",
    "production-ready",
    "lightweight",
    "blazing fast",
    "zero dependencies",
    "self-hosted",
    "autonomous",
]


def extract_readme_claims(text: str) -> list[str]:
    """Return README keyword mentions (unverified claims for future fact-checking)."""
    if not text:
        return []
    lower = text.lower()
    found: set[str] = set()
    for keyword in README_CLAIM_KEYWORDS:
        parts = keyword.split()
        if len(parts) == 1 and len(keyword) <= 6 and keyword.isalpha():
            if re.search(rf"\b{re.escape(keyword)}\b", lower):
                found.add(keyword)
        elif keyword in lower:
            found.add(keyword)
    return sorted(found)


def extract_structured_claims(text: str, source: str = "README.md") -> list[Claim]:
    """Return structured Claim objects from README text.

    Each claim gets a deterministic ID, is tagged as DOCUMENTATION kind,
    and receives a category hint derived from the keyword that triggered it.
    The verdict and evidence_ids are left unpopulated — the verification
    service fills those in later.
    """
    from ..models.schemas import FileKind

    if not text:
        return []

    raw_claims = extract_readme_claims(text)
    claims: list[Claim] = []

    # Map keywords to categories for more meaningful grouping
    keyword_to_category: dict[str, str] = {
        "machine learning": "ai-capability",
        "artificial intelligence": "ai-capability",
        "deep learning": "ai-capability",
        "neural network": "ai-capability",
        "nlp": "ai-capability",
        "natural language processing": "ai-capability",
        "large language model": "ai-capability",
        "llm": "ai-capability",
        "real-time": "performance",
        "scalable": "performance",
        "high performance": "performance",
        "performant": "performance",
        "distributed": "architecture",
        "decentralized": "architecture",
        "microservices": "architecture",
        "cloud-native": "architecture",
        "serverless": "architecture",
        "kubernetes": "architecture",
        "secure": "security",
        "production-ready": "quality",
        "lightweight": "quality",
        "blazing fast": "quality",
        "zero dependencies": "quality",
        "self-hosted": "deployment",
        "autonomous": "quality",
    }

    # Also map individual keywords that appear in the text
    lower_text = text.lower()
    for claim in raw_claims:
        # Determine category from keyword-to-category map, fallback to "general"
        category = keyword_to_category.get(claim, "general")

        # Generate a deterministic claim ID from the claim text
        claim_id = (
            claim.lower().replace(" ", "_").replace("-", "_")[:40]
            or "claim"
        )

        claims.append(
            Claim(
                id=claim_id,
                text=claim,
                source=source,
                kind=FileKind.DOCUMENTATION,
                category=category,
                verdict="unclear",  # default until verification
                verdict_explanation="Awaiting evidence verification.",
                evidence_ids=[],
                repo_id="",  # filled in by verification service
            )
        )

    return claims


def is_readme_filename(filename: str) -> bool:
    return filename.lower().startswith("readme")