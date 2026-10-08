"""Configuration for Northern Star.

Values are read from the environment on every call so that tests (and future
deployments) can point storage or limits elsewhere without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    schema_version: int
    storage_root: Path
    clone_timeout_seconds: int
    analyze_max_files: int
    analyze_max_repo_size_bytes: int
    binary_sniff_bytes: int
    max_config_file_size_bytes: int
    dir_tree_max_depth: int
    dir_tree_max_children: int
    manifest_filename: str
    # M2 — chunking / indexing / retrieval
    chunk_lines: int
    search_limit: int
    db_filename: str
    # M3 — evidence-grounded Q&A
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    ollama_think: bool  # qwen3-style hidden reasoning; OFF by default (speed)
    qa_top_k: int


def get_settings() -> Settings:
    env = os.environ
    storage_root = Path(env.get("NORTHERN_STAR_STORAGE", "storage/repos")).resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    return Settings(
        schema_version=1,
        storage_root=storage_root,
        clone_timeout_seconds=int(env.get("NORTHERN_STAR_CLONE_TIMEOUT_SECONDS", "300")),
        analyze_max_files=int(env.get("NORTHERN_STAR_MAX_FILES", "50000")),
        analyze_max_repo_size_bytes=int(env.get("NORTHERN_STAR_MAX_REPO_SIZE_BYTES", str(500 * 1024 * 1024))),
        binary_sniff_bytes=int(env.get("NORTHERN_STAR_BINARY_SNIFF_BYTES", "1024")),
        max_config_file_size_bytes=int(env.get("NORTHERN_STAR_MAX_CONFIG_SIZE_BYTES", str(512 * 1024))),
        dir_tree_max_depth=int(env.get("NORTHERN_STAR_TREE_MAX_DEPTH", "5")),
        dir_tree_max_children=int(env.get("NORTHERN_STAR_TREE_MAX_CHILDREN", "200")),
        manifest_filename="manifest.json",
        # M2
        chunk_lines=int(env.get("NORTHERN_STAR_CHUNK_LINES", "100")),
        search_limit=int(env.get("NORTHERN_STAR_SEARCH_LIMIT", "20")),
        db_filename=env.get("NORTHERN_STAR_DB_FILENAME", "northern_star.db"),
        # M3 — evidence-grounded Q&A
        ollama_base_url=env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        # qwen2.5:3b: the model that answers fast and cleanly on CPU-only dev
        # boxes (~8s vs qwen3:4b's ~60-130s, whose reasoning leaks into the
        # answer even with OLLAMA_THINK=false — "still thinking"). qwen3:4b
        # stays available via --model / OLLAMA_MODEL.
        ollama_model=env.get("OLLAMA_MODEL", "qwen2.5:3b"),
        # 180s headroom: a 4B model's first answer on CPU-only hardware includes
        # a cold weight-load on top of generation, which the old 120s default
        # failed on even with thinking disabled.
        ollama_timeout_seconds=int(env.get("OLLAMA_TIMEOUT_SECONDS", "180")),
        # Thinking is OFF by default: on qwen3-style models it costs ~2.4x
        # latency on this hardware. Set OLLAMA_THINK=true to re-enable it.
        ollama_think=env.get("OLLAMA_THINK", "false").strip().lower()
        in ("1", "true", "yes", "on"),
        qa_top_k=int(env.get("QA_TOP_K", "5")),
    )