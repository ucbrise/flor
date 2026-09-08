# What FlorDB Writes

Everything is project-local; nothing lands in your home directory.

```
.flor/
  runs/<tstamp>.jsonl     tracked: one immutable record per forward run
  obj_store/<tstamp>/     ignored: checkpoints, addressable by loop iteration
  <projid>.db             ignored: sqlite query cache, rebuildable at any time
.flor.cmd                 tracked: the run's tstamp and command line
```

Each run makes one `FLOR::Auto-commit::<tstamp>` commit on a shadow branch whose
message body carries the run's hyperparameters — so a run stays reproducible
even if the logs are gone. Lost or moved the cache? Rebuild it from the JSONL:

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
`git fetch && python -m flordb unpack` and has everyone's metrics — no server,
no bucket, no bill.

Checkpoints don't travel. What that costs the first replay in a fresh clone, and
how FlorDB pays it down, is covered in [Replay](replay.md#replaying-in-a-fresh-clone).
