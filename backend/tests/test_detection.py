"""Tests for deterministic language / framework detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.schemas import FileKind
from app.services.detection import (
    classify_file,
    detect_frameworks,
    extract_readme_claims,
    is_readme_filename,
)
from app.services.ingestion import analyze_repository


def _text_file(name: str, content: str = "hello\n") -> Path:
    p = Path(f"/tmp/unit-files/{name}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _binary_file(name: str) -> Path:
    p = Path(f"/tmp/unit-files/{name}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00\x01\x02bytes")
    return p


class TestClassifyFile:
    @pytest.mark.parametrize(
        "name,expected_lang,expected_kind",
        [
            ("main.py", "Python", FileKind.SOURCE),
            ("index.ts", "TypeScript", FileKind.SOURCE),
            ("App.tsx", "TypeScript", FileKind.SOURCE),
            ("main.go", "Go", FileKind.SOURCE),
            ("main.rs", "Rust", FileKind.SOURCE),
            ("Dockerfile", "Dockerfile", FileKind.SOURCE),
            ("Makefile", "Makefile", FileKind.SOURCE),
            ("schema.sql", "SQL", FileKind.SOURCE),
            ("styles.css", "CSS", FileKind.SOURCE),
            ("data.csv", "CSV", FileKind.DATA),
            ("package.json", "JSON", FileKind.CONFIG),
            ("pyproject.toml", "TOML", FileKind.CONFIG),
            ("deploy.yml", "YAML", FileKind.CONFIG),
            ("Cargo.lock", None, FileKind.CONFIG),
            ("README.md", "Markdown", FileKind.DOCUMENTATION),
            ("README", "Text", FileKind.DOCUMENTATION),
            ("LICENSE", "Text", FileKind.DOCUMENTATION),
            ("logo.png", None, FileKind.BINARY),
            ("archive.zip", None, FileKind.BINARY),
        ],
    )
    def test_language_and_kind(self, name, expected_lang, expected_kind):
        lang, kind = classify_file(_text_file(name), sniff_bytes=1024)
        assert kind is expected_kind
        assert lang == expected_lang

    def test_unknown_text_extension_is_other(self):
        lang, kind = classify_file(_text_file("notes.weird"), sniff_bytes=1024)
        assert lang is None
        assert kind is FileKind.OTHER

    def test_unknown_extension_binary_sniff(self):
        lang, kind = classify_file(_binary_file("data.weird"), sniff_bytes=1024)
        assert lang is None
        assert kind is FileKind.BINARY

    def test_readme_filename(self):
        assert is_readme_filename("README.md")
        assert is_readme_filename("readme")
        assert not is_readme_filename("src/main.py")


class TestFrameworkDetection:
    def test_npm_react_express_vite(self):
        configs = {
            "package.json": (
                '{"dependencies": {"react": "^18", "express": "^4"},'
                ' "devDependencies": {"vite": "^5"}}'
            )
        }
        paths = {"package.json"}
        frameworks, pm = detect_frameworks(configs, paths)
        names = {f.name for f in frameworks}
        assert {"React", "Express.js", "Vite"} <= names
        assert pm == "npm"

    def test_python_fastapi(self):
        configs = {
            "pyproject.toml": '[project]\ndependencies = ["fastapi>=0.100", "pytest"]\n'
        }
        paths = {"pyproject.toml"}
        frameworks, pm = detect_frameworks(configs, paths)
        names = {f.name for f in frameworks}
        assert "FastAPI" in names
        assert pm == "pip"

    def test_python_poetry_manager(self):
        configs = {
            "pyproject.toml": '[tool.poetry.dependencies]\nfastapi = "^0.100"\ndjango = "*"\n'
        }
        paths = {"pyproject.toml"}
        frameworks, pm = detect_frameworks(configs, paths)
        assert pm == "poetry"
        names = {f.name for f in frameworks}
        assert "FastAPI" in names and "Django" in names

    def test_requirements_django_torch(self):
        configs = {"requirements.txt": "django==5.0\ntorch==2.2"}
        paths = {"requirements.txt"}
        frameworks, pm = detect_frameworks(configs, paths)
        names = {f.name for f in frameworks}
        assert "Django" in names and "PyTorch" in names
        assert pm == "pip"

    def test_cargo_axum(self):
        configs = {"Cargo.toml": '[dependencies]\naxum = "0.7"\n'}
        paths = {"Cargo.toml"}
        frameworks, _ = detect_frameworks(configs, paths)
        assert any(f.name == "Axum" for f in frameworks)

    def test_go_gin(self):
        configs = {"go.mod": "require github.com/gin-gonic/gin v1.9.0\n"}
        paths = {"go.mod"}
        frameworks, _ = detect_frameworks(configs, paths)
        assert any(f.name == "Gin" for f in frameworks)

    def test_package_manager_precedence_yarn_over_npm(self):
        configs = {"package.json": "{}"}
        paths = {"package-lock.json", "yarn.lock"}
        _, pm = detect_frameworks(configs, paths)
        assert pm == "yarn"

    def test_framework_evidence_source(self):
        configs = {"package.json": '{"dependencies": {"express": "^4"}}'}
        paths = {"package.json"}
        frameworks, _ = detect_frameworks(configs, paths)
        express = next(f for f in frameworks if f.name == "Express.js")
        assert express.source == "package.json"

    def test_dedupes_same_framework_from_two_files(self):
        configs = {
            "package.json": '{"dependencies": {"react": "^18"}}',
            "vite.config.ts": "// vite",
        }
        paths = {"package.json", "vite.config.ts"}
        frameworks, _ = detect_frameworks(configs, paths)
        react_entries = [f for f in frameworks if f.name == "React"]
        assert len(react_entries) == 1


class TestReadmeClaims:
    def test_extracts_keywords(self):
        text = "This is a scalablE, real-time engine using machine learning."
        claims = extract_readme_claims(text)
        assert {"scalable", "real-time", "machine learning"} <= set(claims)

    def test_no_false_claim_for_substring(self):
        text = "built with roadside assistance"
        claims = extract_readme_claims(text)
        assert "ai" not in claims


class TestAnalyzeIntegration:
    def test_sample_repo_integration(self, sample_repo, tmp_path, monkeypatch):
        monkeypatch.setenv("NORTHERN_STAR_STORAGE", str(tmp_path / "storage"))
        from app.config import get_settings

        analysis = analyze_repository(sample_repo, get_settings(), tree_root_name="sample-repo")

        assert analysis.readme_present is True
        assert {"machine learning", "real-time"} <= set(analysis.readme_claims)
        assert analysis.package_manager == "npm"

        framework_names = {f.name for f in analysis.frameworks}
        assert {"React", "Express.js", "Vite"} <= framework_names

        # node_modules and dist must be excluded; referenced in ignored list
        entry_paths = {e.path for e in analysis.entries}
        assert "dist/bundle.js" not in entry_paths
        assert "node_modules/react/index.js" not in entry_paths
        ignored_paths = {i.path for i in analysis.ignored_directories}
        assert "node_modules" in ignored_paths and "dist" in ignored_paths

        # .env is excluded as a potential secrets file
        assert ".env" not in entry_paths

        # provenance fields
        src = {e.path for e in analysis.entries if e.category == FileKind.SOURCE}
        assert "src/main.ts" in src and "src/utils/helper.py" in src
        logo = next(e for e in analysis.entries if e.path == "public/logo.png")
        assert logo.category is FileKind.BINARY

        # directory structure includes dirs but not files
        top_level = {c.name for c in analysis.directory_structure.children}
        assert {"src", "public"} <= top_level