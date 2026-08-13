"""How commit() sequences the sqlite write against the git commit.

In-process rather than subprocess: the interesting cases are all "git_commit
blew up halfway", and the cheapest way to get that is to substitute one that
raises. The buffer and DB are the sandbox repo conftest imported flordb
against, so these tests write real rows -- each one cleans up after itself.
"""

import sqlite3

import pytest

from flordb import api, constants, orm, versions


def _row_count(name):
    conn = sqlite3.connect(constants.DB_PATH)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM logs WHERE value_name = ?", (name,)
        ).fetchone()[0]
    finally:
        conn.close()


def _writes_without_blocking():
    """True if a second connection can take the write lock right now.

    timeout=0 so a lock held by a leaked connection surfaces as
    OperationalError immediately instead of stalling the suite.
    """
    conn = sqlite3.connect(constants.DB_PATH, timeout=0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.rollback()
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


@pytest.fixture
def staged(sandbox, monkeypatch):
    """A run with one record buffered, and a name to find it by."""
    name = "commit-test::marker"
    api.output_buffer.clear()
    api.output_buffer.append(
        orm.Log(
            constants.PROJID,
            api.Clock.get_datetime(),
            constants.SCRIPTNAME,
            None,
            name,
            1,
            constants.VALUE_TYPE_LOG,
        )
    )
    monkeypatch.setattr(api, "skip_cleanup", False)
    try:
        yield name
    finally:
        api.output_buffer.clear()
        api.skip_cleanup = True
        conn = sqlite3.connect(constants.DB_PATH)
        conn.execute("DELETE FROM logs WHERE value_name = ?", (name,))
        conn.commit()
        conn.close()


class TestGitFailureDoesNotStrandTheConnection:
    """Regression: Ctrl-C through `git add -A` used to poison the session.

    The interrupt landed while commit()'s connection sat open mid-write, so
    its RESERVED lock outlived the call and every later commit() died with
    "database is locked" -- the failure a user hit by running flor from an
    interactive shell in a large repo.
    """

    def test_the_database_is_writable_afterwards(self, staged, monkeypatch):
        def boom(message):
            raise KeyboardInterrupt

        monkeypatch.setattr(versions, "git_commit", boom)
        with pytest.raises(KeyboardInterrupt):
            api.commit()
        assert _writes_without_blocking()

    def test_the_database_is_writable_while_git_runs(self, staged, monkeypatch):
        """Not just released on the way out -- released *before* git starts.

        git_commit is the slow, interruptible part; holding the write lock
        across it is what made the window wide enough to hit.
        """
        seen = {}

        def check(message):
            seen["unlocked"] = _writes_without_blocking()

        monkeypatch.setattr(versions, "git_commit", check)
        api.commit()
        assert seen["unlocked"]

    def test_the_run_is_recorded_even_though_git_failed(self, staged, monkeypatch):
        def boom(message):
            raise RuntimeError

        monkeypatch.setattr(versions, "git_commit", boom)
        with pytest.raises(RuntimeError):
            api.commit()
        assert _row_count(staged) == 1

    def test_the_buffer_is_not_replayed_into_a_second_run(self, staged, monkeypatch):
        """The atexit hook, or the next IPython cell, must not double-write.

        The rows are durable by the time git runs, so the reset has to have
        happened too -- otherwise a retry inserts every record again.
        """
        calls = []

        def fails_once(message):
            calls.append(message)
            if len(calls) == 1:
                raise RuntimeError

        monkeypatch.setattr(versions, "git_commit", fails_once)
        with pytest.raises(RuntimeError):
            api.commit()
        assert not api.output_buffer
        assert api.skip_cleanup is True
        api.commit()  # what a retrying caller would do
        assert _row_count(staged) == 1
