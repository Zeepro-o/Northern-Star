"""Shared fixtures for Northern Star tests.

Tests never touch the default ``storage/repos`` directory: the storage root is
redirected to a per-session temp directory before anything else runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `app` importable when running pytest from inside backend/.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(autouse=True)
def no_default_storage(monkeypatch, tmp_path: Path) -> Path:
    """Never let tests touch the real ``storage/repos`` directory: point
    NORTHERN_STAR_STORAGE at a throwaway directory for every test."""
    storage = tmp_path / "test-storage"
    monkeypatch.setenv("NORTHERN_STAR_STORAGE", str(storage))
    return storage


def write_files(root: Path, files: dict[str, str]) -> None:
    """Create a nested file tree from {relative_path: text_content}."""
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


@pytest.fixture()
def sample_repo(tmp_path: Path) -> Path:
    """A small deterministic fake repository used across service and API tests."""
    root = tmp_path / "sample-repo"
    root.mkdir(parents=True, exist_ok=True)
    write_files(root, {
        "README.md": (
            "Acme Widget\n"
            "A real-time, scalable widget engine with machine learning support.\n"
        ),
        "package.json": (
            '{\n'
            '  "name": "acme-widget",\n'
            '  "dependencies": {"react": "^18.0.0", "express": "^4.19.0"},\n'
            '  "devDependencies": {"vite": "^5.0.0", "typescript": "^5.0.0"}\n'
            '}\n'
        ),
        "src/main.ts": "import { render } from 'react';\nexport function main() { return 1; }\n",
        "src/utils/helper.py": "def helper():\n    return 42\n",
        "public/logo.png": b"\x89PNG\r\n\x1a\nbinarypayload".decode("latin-1"),
        "dist/bundle.js": "// should be ignored\n",
        "node_modules/react/index.js": "// should be ignored\n",
        ".gitignore": "node_modules/\n",
        ".env": "API_KEY=should-be-skipped\n",
    })
    return root


# ---------------------------------------------------------------------------
# M2 evidence fixtures — deterministic content with known retrieval concepts.
# ---------------------------------------------------------------------------

EVIDENCE_FILES: dict[str, str] = {
    "src/auth/__init__.py": "from .middleware import require_auth\n",
    "src/auth/middleware.py": (
        "from fastapi import HTTPException, Request\n"
        "\n"
        "\n"
        "def require_auth(request: Request) -> None:\n"
        "    \"\"\"Reject requests that carry no authentication.\"\"\"\n"
        "    auth = request.headers.get(\"Authorization\")\n"
        "    if auth is None or not auth.startswith(\"Bearer \"):\n"
        "        raise HTTPException(status_code=401, detail=\"missing bearer token\")\n"
        "    token = auth.split(\" \", 1)[1]\n"
        "    validate_token(token)\n"
    ),
    "src/auth/guards.py": (
        "def require_permission(user, permission: str) -> bool:\n"
        "    \"\"\"Gate an action behind an authorization check.\"\"\"\n"
        "    return permission in user.permissions\n"
    ),
    "src/db/connection.py": (
        "import sqlite3\n"
        "\n"
        "\n"
        "def connect_database(path: str) -> sqlite3.Connection:\n"
        "    \"\"\"Open the application database connection.\"\"\"\n"
        "    conn = sqlite3.connect(path)\n"
        "    conn.execute(\"PRAGMA foreign_keys = ON\")\n"
        "    return conn\n"
    ),
    "src/api/routes.py": (
        "from fastapi import APIRouter, Depends\n"
        "from src.auth.middleware import require_auth\n"
        "\n"
        "router = APIRouter()\n"
        "\n"
        "\n"
        "@router.get(\"/items\")\n"
        "def list_items(require_auth: None = Depends(lambda: None)):\n"
        "    \"\"\"HTTP API endpoint listing all items.\"\"\"\n"
        "    return {\"items\": []}\n"
    ),
    "src/components/Button.tsx": (
        "import * as React from 'react'\n"
        "\n"
        "interface Props {\n"
        "  label: string\n"
        "  onClick: () => void\n"
        "}\n"
        "\n"
        "export function Button({ label, onClick }: Props) {\n"
        "  return <button onClick={onClick}>{label}</button>\n"
        "}\n"
    ),
    "pyproject.toml": (
        '[project]\n'
        'name = "evidence-app"\n'
        'version = "0.1.0"\n'
        'dependencies = ["fastapi>=0.100", "sqlmodel"]\n'
    ),
    "README.md": (
        "Evidence App\n"
        "===========\n"
        "\n"
        "A demonstration repository used by Northern Star's retrieval tests.\n"
    ),
}


def _large_python_file(n: int = 260) -> str:
    """A generated >150-line Python file that must be split into chunks."""
    lines = ["# Generated for chunk-splitting tests", ""]
    for i in range(n):
        lines.append(f"def fn_{i:04d}(x):")
        lines.append(f"    return x + {i}")
        lines.append("")
    return "\n".join(lines)


EVIDENCE_FILES["src/generated_big.py"] = _large_python_file()


def build_evidence_checkout(root: Path, files: dict[str, str] | None = None) -> Path:
    """Materialize the evidence repository on disk and return its path."""
    target = root / "evidence-repo"
    target.mkdir(parents=True, exist_ok=True)
    write_files(target, files or EVIDENCE_FILES)
    return target


@pytest.fixture()
def evidence_repo(no_default_storage: Path) -> tuple[Path, Path, Path]:
    """Prep an evidence checkout + manifest for indexing tests.

    Returns ``(checkout_root, manifest, db_path)`` using the real ingestion
    analysis pipeline so the manifest's file inventory is authoritative.
    """
    from app.config import get_settings
    from app.models.schemas import RepositoryManifest
    from app.services.ingestion import _assemble_manifest, analyze_repository
    from app.services.github import FetchedRepository

    checkout = build_evidence_checkout(no_default_storage / "fixtures")
    settings = get_settings()
    analysis = analyze_repository(checkout, settings, tree_root_name="evidence")
    fetched = FetchedRepository(
        owner="acme",
        repo="evidence",
        github_url="https://github.com/acme/evidence",
        checkout_root=checkout,
        commit_hash="deadbeef",
        default_branch="main",
    )
    manifest: RepositoryManifest = _assemble_manifest(fetched, analysis, settings, None)
    db_path = no_default_storage / settings.db_filename
    return checkout, manifest, db_path