# What FlorDB Writes

Everything is project-local; nothing lands in your home directory.

```
.flor/
  runs/<tstamp>.jsonl      tracked: one immutable record per forward run
  obj_store/<tstamp>/      ignored: the run's copy of each checkpoint file it saved
  <projid>.db              ignored: sqlite query cache, rebuildable at any time
.<branch>.cmd              tracked: the branch's latest run tstamp and command line
```

For example, runs on `flor.trial-a` update `.flor.trial-a.cmd`. Each branch
has its own file, so merging trials preserves their latest commands without
conflicts over a shared command file. Branch names are percent-encoded in the
filename (`flor.trial/a` becomes `.flor.trial%2Fa.cmd`). Older `.flor.cmd`
files are left intact; new runs write only the branch-specific file.

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
| `obj_store/` | Yes, by retraining | No — one copy per checkpoint file per run, and checkpoints don't dedup |
| `<projid>.db` | Yes, `flor unpack` in seconds | No |

Only the first row is an observation. Everything else is a function of it, and
a derived thing FlorDB can recompute is one it declines to commit — including
metrics read out of captured text, which live in the cache alone (see
[capture](capture.md)).

So run history travels with the code that produced it, but only as far as you
push it. Auto-commits stay local until you push a `flor.` branch, and you can
merge selected trial branches into one, such as `flor.dev`, before sharing. A
teammate who checks out a pushed branch runs `python -m flordb unpack` to load
its metrics — no server, no bucket, no bill.

## Where auto-commits land

Those commits are noisy — one per run — so FlorDB keeps them off the branch you
review and merge. On the first run from any branch not already prefixed
`flor.`, FlorDB creates a shadow branch named `flor.branch` (`flor.branch1`,
`flor.branch2`, … if that name is taken) and switches to it, then auto-commits
there. You stay on that branch after the run. If you're already on any branch
whose name starts with `flor.`, including one you created yourself, FlorDB uses
it without creating another branch.
So `main` never accumulates run history unless you merge it there yourself.

See [Working on a FlorDB branch](branches.md) for saving later edits, publishing
the branch, and bringing selected code changes back for review.

Two things follow. The auto-commit is a `git add -A`, so whatever is
uncommitted when you launch a run — the edit you were mid-way through — is
captured in that run's commit. That is the point: the commit is an exact record
of the code that produced the metrics. It lands on the shadow branch, so your
working branch's history stays yours to curate.

And because the runs live on the shadow branch, a collaborator has to check it
out. A `git fetch` alone leaves `.flor/runs/` empty on `main`, and `unpack`
reads whatever JSONL is in the working tree.

Checkpoints don't travel either. Replay doesn't need them (see
[Replay](replay.md#replaying-in-a-fresh-clone)), but comparing models in a
notebook does: copy `.flor/obj_store/<tstamp>/` from the machine that trained a
run to load its model elsewhere.
