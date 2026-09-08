# What FlorDB Writes

Everything is project-local; nothing lands in your home directory.

```
.flor/
  runs/<tstamp>.jsonl     tracked: one immutable record per forward run
  obj_store/<tstamp>/     ignored: checkpoints, addressable by loop iteration
  <projid>.db             ignored: sqlite query cache, rebuildable at any time
.flor.cmd                 tracked: the run's tstamp and command line
```

Each run makes one `FLOR::Auto-commit::<tstamp>` commit whose message body
carries the run's hyperparameters — so a run stays reproducible even if the logs
are gone. Lost or moved the cache? Rebuild it from the JSONL:

```bash
python -m flordb unpack
```

## What syncs, and what doesn't

The three things under `.flor/` have very different economics, so FlorDB treats
them differently in `.gitignore` (written on first run):

| | Recomputable? | In git? |
|---|---|---|
| `runs/*.jsonl` | No — the one irreplaceable observation | **Yes**, ~69KB packed per run |
| `obj_store/` | Yes, by replaying | No — ~19MB per run, and checkpoints don't dedup |
| `<projid>.db` | Yes, `flor unpack` in seconds | No |

So run history travels with the code that produced it. A teammate runs
`git fetch && git checkout flor.branch && python -m flordb unpack` and has
everyone's metrics — no server, no bucket, no bill.

## Auto-commits never land on your working branch

Those commits are noisy — one per run — so FlorDB keeps them off the branch you
review and merge. On the first run from any branch not already prefixed
`flor.`, FlorDB creates a shadow branch named `flor.branch` (`flor.branch1`,
`flor.branch2`, … if that name is taken) and switches to it, then auto-commits
there. Run again from that branch and you stay on it; nothing new is created.
So `main` never accumulates run history unless you merge it there yourself.

Two things follow. The auto-commit is a `git add -A`, so whatever is
uncommitted when you launch a run — the edit you were mid-way through — is
captured in that run's commit. That is the point: the commit is an exact record
of the code that produced the metrics. It lands on the shadow branch, so your
working branch's history stays yours to curate.

And because the runs live on the shadow branch, a collaborator has to check it
out. A `git fetch` alone leaves `.flor/runs/` empty on `main`, and `unpack`
reads whatever JSONL is in the working tree.

Checkpoints don't travel. What that costs the first replay in a fresh clone, and
how FlorDB pays it down, is covered in [Replay](replay.md#replaying-in-a-fresh-clone).
