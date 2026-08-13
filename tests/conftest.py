"""Shared fixtures.

Importing flordb has two process-wide side effects that have to be contained
before the first import happens, hence the module-level work below:

  * `flordb.constants` resolves the project root from the CWD and creates
    `.flor/` there; `flordb/__init__` then opens `.flor/<projid>.db`. Left
    alone, running the suite from a checkout would write into the developer's
    own cache.
  * `flordb.cli.parse_args()` runs at import and would try to interpret
    pytest's argv as flor flags.
"""

import atexit
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )


def init_repo(path):
    """A git repo flor is willing to run in: has a branch and a commit."""
    os.makedirs(path, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "flor-tests@example.com")
    git(path, "config", "user.name", "Flor Tests")
    git(path, "config", "commit.gpgsign", "false")
    with open(os.path.join(path, "seed.txt"), "w") as f:
        f.write("flordb test repo\n")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "init")
    return path


_SANDBOX = init_repo(tempfile.mkdtemp(prefix="flordb-tests-"))
atexit.register(shutil.rmtree, _SANDBOX, ignore_errors=True)
os.chdir(_SANDBOX)

_argv = list(sys.argv)
sys.argv = [_argv[0]]
try:
    import flordb  # noqa: F401  (import for side effects; see module docstring)
finally:
    sys.argv = _argv

# Importing flordb also tees sys.stdout/sys.stderr and wraps logging dispatch,
# which would absorb pytest's own output into flordb.api.output_buffer. The
# subprocess-based tests exercise the installed path for real; in-process tests
# drive `capture` directly.
flordb.capture.uninstall()
flordb.api.output_buffer.clear()


@pytest.fixture
def sandbox():
    """Path to the git repo flordb was imported against."""
    return _SANDBOX


@pytest.fixture
def clean_flags():
    """Reset cli.flags around a test that mutates replay state."""
    from flordb import cli

    saved = cli.flags
    cli.flags = cli.Flags()
    try:
        yield cli.flags
    finally:
        cli.flags = saved


@pytest.fixture
def logs_db():
    """An in-memory `logs` table built by the real schema code."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    from flordb import database

    database.create_tables(cursor)
    conn.commit()
    try:
        yield conn, cursor
    finally:
        conn.close()


@pytest.fixture
def project(tmp_path):
    """A fresh git repo plus a helper that runs scripts in it as flor would.

    Runs are real subprocesses so the import-time machinery (project root
    discovery, argv parsing, shadow branch, auto-commit) is exercised end to
    end rather than mocked.
    """

    class Project:
        def __init__(self, root):
            self.root = str(root)
            init_repo(self.root)

        def write(self, name, source):
            path = os.path.join(self.root, name)
            with open(path, "w") as f:
                f.write(source)
            return path

        def run(self, *argv, check=True, env=None):
            environ = dict(os.environ)
            environ["PYTHONPATH"] = (
                REPO_ROOT + os.pathsep + environ.get("PYTHONPATH", "")
            )
            environ["PYTHONUNBUFFERED"] = "1"
            if env:
                environ.update(env)
            env = environ
            proc = subprocess.run(
                [sys.executable, *argv],
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if check and proc.returncode != 0:
                raise AssertionError(
                    f"{argv} failed ({proc.returncode})\n"
                    f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
                )
            return proc

        @property
        def runs_dir(self):
            return os.path.join(self.root, ".flor", "runs")

        def run_files(self):
            return sorted(
                os.path.join(self.runs_dir, f)
                for f in os.listdir(self.runs_dir)
                if f.endswith(".jsonl")
            )

        def records(self, path=None):
            path = path or self.run_files()[-1]
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]

        def db(self):
            projid = os.path.basename(self.root)
            return sqlite3.connect(
                os.path.join(self.root, ".flor", f"{projid}.db")
            )

        def git_log(self):
            return git(self.root, "log", "--format=%B%x00").stdout.split("\0")

    return Project(tmp_path / "project")
