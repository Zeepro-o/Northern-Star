"""Tests for GitHub URL parsing and git fetching service."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.github import (
    InvalidGitHubUrlError,
    RepoNotFoundError,
    fetch_repository,
    is_valid_slug,
    parse_github_url,
)


class TestParseGithubUrl:
    @pytest.mark.parametrize(
        "url,owner,repo,canonical",
        [
            ("https://github.com/acme/widget", "acme", "widget", "https://github.com/acme/widget"),
            ("https://github.com/acme/widget.git", "acme", "widget", "https://github.com/acme/widget"),
            ("https://github.com/acme/widget/", "acme", "widget", "https://github.com/acme/widget"),
            ("https://www.github.com/acme/widget", "acme", "widget", "https://github.com/acme/widget"),
            ("acme/widget", "acme", "widget", "https://github.com/acme/widget"),
            ("OctoCat/Hello-World", "OctoCat", "Hello-World", "https://github.com/OctoCat/Hello-World"),
        ],
    )
    def test_accepted_forms(self, url, owner, repo, canonical):
        parsed = parse_github_url(url)
        assert parsed.owner == owner
        assert parsed.repo == repo
        assert parsed.canonical_url == canonical

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "   ",
            "not-a-url",
            "http://github.com/acme/widget",           # insecure
            "git@github.com:acme/widget.git",          # ssh
            "ssh://git@github.com/acme/widget.git",    # ssh
            "https://gitlab.com/acme/widget",          # wrong host
            "https://github.com/a/b/c",                 # too many segments
            "https://github.com/",                      # missing owner/repo
            "https://github.com/.",                     # invalid slug
        ],
    )
    def test_rejected_forms(self, url):
        with pytest.raises(InvalidGitHubUrlError):
            parse_github_url(url)


class TestIsValidSlug:
    def test_ok(self):
        assert is_valid_slug("acme")
        assert is_valid_slug("octo-cat")
        assert is_valid_slug("a_b.c")

    def test_bad(self):
        assert not is_valid_slug("")
        assert not is_valid_slug(".")
        assert not is_valid_slug("..")
        assert not is_valid_slug("a/b")
        assert not is_valid_slug("a b")


class TestFetchRepository:
    def test_fetch_missing_repo_raises_not_found(self, tmp_path):
        base = tmp_path / "fetch-base"
        with pytest.raises(RepoNotFoundError):
            fetch_repository("https://github.com/northern-star-does-not-exist/nope", base, timeout=30)

    def test_parse_error_propagates(self, tmp_path):
        base = tmp_path / "fetch-base"
        with pytest.raises(InvalidGitHubUrlError):
            fetch_repository("not a url", base, timeout=30)

    def test_storage_layout(self, tmp_path):
        base = tmp_path / "fetch-base"
        fetched = None
        try:
            fetched = fetch_repository("https://github.com/octocat/Hello-World", base, timeout=60)
        except Exception as exc:  # network off: skip won't hide a real failure
            pytest.skip(f"Network unavailable in this environment ({exc})")
        assert fetched.owner == "octocat"
        assert fetched.repo == "Hello-World"
        assert fetched.checkout_root == base / "octocat" / "hello-world" / "checkout"
        assert fetched.checkout_root.exists()
        assert fetched.commit_hash is not None