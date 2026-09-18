"""Concurrent ranks in one repo: what survives, and what gets dropped.

`torchrun --nproc_per_node=N` is N processes sharing one working tree, so N
flor runs record and finalize at once. Nothing in flor is rank-aware, and two
distinct failures come out of that. Neither needs a GPU to reproduce -- both
are races between ordinary processes -- so nothing here imports torch.

1. Git finalization (`TestGitFinalizationIsNotRankAware`). The ranks collide on
   `.git/index.lock` during auto-commit, so fewer auto-commits survive than
   there were runs, and the losers still exit 0. The observations themselves
   are safe: `git add -A` in whichever commit wins sweeps in every rank's
   JSONL. What is lost is the *link* from a run to the commit that produced it.
   `versions.get_latest_autocommit` pairs a tstamp with a hexsha by reading
   auto-commit subjects, and `repl.Schedule.iter_dims` looks every scheduled
   run up in that mapping -- so a run with no auto-commit of its own raises
   KeyError there. Every rank but the winner becomes unreplayable, which takes
   hindsight logging with it.

2. Run identity (`TestRunIdentityIsNotRankAware`). A run is identified by
   `Clock.current_datetime`, a microsecond timestamp taken at import and used
   to name the JSONL, the object-store shelf, and the commit subject. Ranks
   launched together import within a few hundred microseconds of each other,
   so two of them can take the *same* timestamp -- observed here with 4 ranks
   on an idle machine, at roughly one trial in six:

       RANK 0 tstamp=2026-09-17T22:13:12.226664
       RANK 3 tstamp=2026-09-17T22:13:12.226664   <- same run id

   `orm.to_jsonl` opens the path with "w", so the second rank's run file
   replaces the first's outright and those observations are gone. This one is
   quiet in a way the git collision is not: nothing is printed and no error is
   raised. A job that stages ranks apart (real NCCL init takes milliseconds)
   will usually miss it, which is what makes it worth pinning down in a test
   rather than leaving to luck.

The invariants that hold today are asserted as passing, to keep them from
regressing. The ones that do not are `xfail(strict=True)`: they state what
rank-aware recording has to deliver, and once it does they turn into XPASS,
which fails the suite until the marker is removed.
"""

import os
import subprocess
import sys

import pytest
from conftest import REPO_ROOT, git

pytestmark = pytest.mark.slow

WORLD_SIZE = 4

# Each rank logs a distinguishable value, then waits at a file-based barrier
# until every other rank has finished logging. The barrier is what makes the
# git collision reliable: left to chance the ranks can serialize, finalizing
# one after another, and the race under test simply wouldn't happen. Flor's
# commit runs from an atexit hook, so releasing the barrier at the end of the
# script lands all N ranks in `git add -A` within milliseconds of each other.
TRAIN = '''
import os
import time

import flordb as flor

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
gate = os.environ["FLOR_TEST_GATE"]

lr = flor.arg("lr", 0.01)
for epoch in flor.loop("epoch", range(2)):
    flor.log("loss", round(1.0 / (epoch + rank + 1), 4))
flor.log("rank", rank)

os.makedirs(gate, exist_ok=True)
open(os.path.join(gate, str(rank)), "w").close()
deadline = time.time() + 60
while len(os.listdir(gate)) < world_size and time.time() < deadline:
    time.sleep(0.01)
'''

# Two ranks whose `import flordb` lands in the same microsecond hold the same
# Clock.current_datetime, and so the same run identity. Pinning the clock right
# after import reproduces that condition exactly, without depending on winning
# a race -- the ranks here run one after another, which also isolates the
# identity collision from the git collision above.
TRAIN_PINNED = '''
import os

import flordb as flor
from flordb.clock import Clock

Clock.current_datetime = os.environ["FLOR_PINNED_TSTAMP"]

flor.log("rank", int(os.environ["RANK"]))
flor.log("loss", round(1.0 / (int(os.environ["RANK"]) + 1), 4))
'''


def rank_env(rank, **extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["RANK"] = str(rank)
    env.update(extra)
    return env


class Ranks:
    """The outcome of one simulated `torchrun` over a flor project."""

    def __init__(self, project, processes):
        self.project = project
        self.processes = processes

    @property
    def returncodes(self):
        return [p["returncode"] for p in self.processes]

    def recorded_tstamps(self):
        """Run tstamps on disk -- one JSONL per run identity that survived."""
        return {
            os.path.splitext(os.path.basename(path))[0]
            for path in self.project.run_files()
        }

    def committed_tstamps(self):
        """Run tstamps named by an auto-commit subject.

        Read the way `versions.get_latest_autocommit` reads them, by subject
        prefix. Reimplemented against `git log` rather than called directly
        because that module resolves its repo from the CWD at import time,
        which is the suite's sandbox, not this test's project.
        """
        prefix = "FLOR::Auto-commit::"
        subjects = git(self.project.root, "log", "--all", "--format=%s").stdout
        return {
            line[len(prefix):].strip()
            for line in subjects.splitlines()
            if line.startswith(prefix)
        }

    def records_of(self, tstamp):
        return self.project.records(
            os.path.join(self.project.runs_dir, f"{tstamp}.jsonl")
        )

    def rank_of(self, tstamp):
        """Which rank wrote the run file with this tstamp."""
        for record in self.records_of(tstamp):
            if record["name"] == "rank":
                return int(record["value"])
        raise AssertionError(f"no rank recorded in {tstamp}.jsonl")


@pytest.fixture
def ranks(project, tmp_path):
    """WORLD_SIZE flor runs started at once in one repo, as torchrun starts them."""
    project.write("train.py", TRAIN)
    gate = tmp_path / "gate"

    processes = []
    for rank in range(WORLD_SIZE):
        processes.append(
            subprocess.Popen(
                [sys.executable, "train.py"],
                cwd=project.root,
                env=rank_env(rank, WORLD_SIZE=str(WORLD_SIZE), FLOR_TEST_GATE=str(gate)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    results = []
    for rank, proc in enumerate(processes):
        output, _ = proc.communicate(timeout=300)
        results.append(
            {"rank": rank, "returncode": proc.returncode, "output": output}
        )
    return Ranks(project, results)


@pytest.fixture
def same_microsecond(project):
    """Two runs that took the same run identity, the way co-launched ranks can.

    Sequential, so the only thing under test is the identity collision.
    """
    project.write("train.py", TRAIN_PINNED)
    tstamp = "2026-09-17T22:13:12.226664"
    results = []
    for rank in range(2):
        proc = project.run(
            "train.py", env={"RANK": str(rank), "FLOR_PINNED_TSTAMP": tstamp}
        )
        results.append({"rank": rank, "returncode": proc.returncode, "output": proc.stdout})
    return Ranks(project, results)


class TestConcurrentRanksDoNotCorruptEachOther:
    """What holds today at the data level. These guard it against regression."""

    def test_each_run_file_is_complete_and_belongs_to_one_rank(self, ranks):
        """Ranks interleave their writes without tearing each other's records.

        Asserted per surviving run file rather than over all WORLD_SIZE ranks:
        two ranks sharing a tstamp leaves one file, which is a different
        failure with its own test below.
        """
        recorded = ranks.recorded_tstamps()
        assert recorded

        owners = [ranks.rank_of(tstamp) for tstamp in recorded]
        assert len(set(owners)) == len(recorded)

        for tstamp in recorded:
            names = [record["name"] for record in ranks.records_of(tstamp)]
            assert names.count("loss") == 2
            assert names.count("lr") == 1

    def test_every_run_file_is_tracked_by_git(self, ranks):
        """`git add -A` in whichever commit wins sweeps in every rank's JSONL,
        so the observations reach git even when their own commit was lost."""
        tracked = set(ranks.project.git_tracked_files())
        for tstamp in ranks.recorded_tstamps():
            assert f".flor/runs/{tstamp}.jsonl" in tracked


class TestGitFinalizationIsNotRankAware:
    """Ranks race to auto-commit, and the losers go quiet.

    Remove the xfail markers together with whatever change makes finalization
    rank-aware -- serializing the ranks behind a lock, retrying the collided
    commit, or electing one rank to commit on behalf of the job.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "ranks collide on .git/index.lock: `git add -A` fails for all but "
            "one, so runs are recorded with no auto-commit naming them"
        ),
    )
    def test_every_run_gets_an_auto_commit_naming_it(self, ranks):
        """The tstamp->commit mapping replay depends on has to cover every run.

        `repl.Schedule.iter_dims` raises KeyError on a run missing from it, so
        a run without its own auto-commit cannot be replayed at all.
        """
        assert ranks.recorded_tstamps() <= ranks.committed_tstamps()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "versions.git_commit swallows the index.lock failure, so a rank "
            "that got nothing into git still exits 0"
        ),
    )
    def test_a_rank_that_could_not_commit_does_not_report_success(self, ranks):
        """A collided rank prints the git error but exits 0, and launchers --
        torchrun, Slurm -- read exit 0 as success, so the job looks clean while
        most of its runs went unversioned."""
        uncommitted = ranks.recorded_tstamps() - ranks.committed_tstamps()
        silent = [
            rank
            for rank in (ranks.rank_of(tstamp) for tstamp in uncommitted)
            if ranks.returncodes[rank] == 0
        ]
        assert not silent


class TestRunIdentityIsNotRankAware:
    """Two runs that start in the same microsecond are one run as far as flor
    can tell, and the second overwrites the first.

    Remove the xfail marker together with whatever makes a run's identity
    unique per process -- a pid or rank component, or a uniquifying suffix
    when the timestamp is already taken.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "a run is identified by its import-time microsecond timestamp "
            "alone, so co-launched ranks can collide and orm.to_jsonl "
            "overwrites the earlier rank's run file"
        ),
    )
    def test_two_runs_starting_together_get_separate_run_files(
        self, same_microsecond
    ):
        assert len(same_microsecond.recorded_tstamps()) == 2

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "the losing rank's JSONL is replaced outright, so its records are "
            "not on disk anywhere"
        ),
    )
    def test_neither_rank_loses_its_observations(self, same_microsecond):
        """The quiet part: no error, no warning, one rank's run simply gone."""
        surviving = {
            int(record["value"])
            for tstamp in same_microsecond.recorded_tstamps()
            for record in same_microsecond.records_of(tstamp)
            if record["name"] == "rank"
        }
        assert surviving == {0, 1}

    def test_the_collision_is_not_reported(self, same_microsecond):
        """Documents the silence rather than endorsing it: whatever makes the
        overwrite stop happening should make this assertion fail, at which
        point it should be deleted along with the xfails above."""
        assert all(code == 0 for code in same_microsecond.returncodes)
        assert not any(
            "overwrit" in process["output"].lower()
            or "collision" in process["output"].lower()
            for process in same_microsecond.processes
        )
