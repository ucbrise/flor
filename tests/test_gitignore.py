import os
import subprocess

import pytest

from flordb import versions


def write_gitignore(root, text):
    path = os.path.join(root, ".gitignore")
    with open(path, "w") as f:
        f.write(text)
    return path


def read_gitignore(root):
    with open(os.path.join(root, ".gitignore")) as f:
        return f.read()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    monkeypatch.setattr(versions, "CURRDIR", str(root))
    return str(root)


class TestEnsureGitignored:
    def test_creates_the_file_when_absent(self, repo):
        versions.ensure_gitignored()
        assert read_gitignore(repo).split() == [
            ".flor/*",
            "!.flor/runs/",
            "!.flor/extracted/",
        ]

    def test_is_idempotent(self, repo):
        versions.ensure_gitignored()
        first = read_gitignore(repo)
        versions.ensure_gitignored()
        assert read_gitignore(repo) == first

    def test_migrates_the_legacy_bare_entry(self, repo):
        write_gitignore(repo, "*.pyc\n.flor/\nbuild/\n")
        versions.ensure_gitignored()
        entries = read_gitignore(repo).split()
        # The legacy line excluded the directory outright; git will not descend
        # into an excluded directory, so leaving it would make the re-include
        # below it dead.
        assert ".flor/" not in entries
        assert entries == [
            "*.pyc",
            "build/",
            ".flor/*",
            "!.flor/runs/",
            "!.flor/extracted/",
        ]

    def test_preserves_unrelated_entries(self, repo):
        write_gitignore(repo, "# comment\n\n.venv/\n")
        versions.ensure_gitignored()
        text = read_gitignore(repo)
        assert "# comment" in text
        assert ".venv/" in text

    def test_appends_last_so_the_reinclude_wins(self, repo):
        write_gitignore(repo, ".flor/\n*.jsonl\n")
        versions.ensure_gitignored()
        entries = read_gitignore(repo).split()
        # gitignore is last-match-wins: the re-include has to sit after any
        # broader pattern that would otherwise swallow the run files.
        assert entries.index("!.flor/runs/") > entries.index("*.jsonl")


class TestGitAgreesWithTheIntent:
    """The patterns above are only correct if git actually reads them that way."""

    def check_ignored(self, root, path):
        full = os.path.join(root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write("x")
        result = subprocess.run(
            ["git", "-C", root, "check-ignore", "-q", path],
            capture_output=True,
        )
        return result.returncode == 0

    def test_runs_are_tracked_cache_and_objstore_are_not(self, repo):
        versions.ensure_gitignored()
        assert not self.check_ignored(repo, ".flor/runs/2026-01-01T00:00:00.jsonl")
        assert self.check_ignored(repo, ".flor/obj_store/2026-01-01/ckpt.pth")
        assert self.check_ignored(repo, ".flor/repo.db")

    def test_migrated_repo_tracks_runs_too(self, repo):
        write_gitignore(repo, ".flor/\n")
        versions.ensure_gitignored()
        assert not self.check_ignored(repo, ".flor/runs/2026-01-01T00:00:00.jsonl")
        assert self.check_ignored(repo, ".flor/obj_store/2026-01-01/ckpt.pth")
