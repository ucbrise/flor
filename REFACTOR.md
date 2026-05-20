# Version 4

## Least Redundancy Overhaul

If a script performs io, like `print` or `logging`, FlorDB will automatically capture the output of subsequent runs, and also be able to parse and integrate historical logs of various formats and layouts. This means you can start using FlorDB with zero code changes, and it will automatically index and structure your logs for easy retrieval and analysis (with minimal duplication).

### Object Store and Flor Checkpointing

In many cases, students will clone a project, which already does torch logging, and fail to do flor checkpointing. This leads to a failure where there shouldn't be any, we can just piggy back off the checkpoints that were already taken. It will take some clever engineering but an elegant solution is possible.

## LLM Ready Data Layout

The legacy layout — a single `.flor.json` overwritten per run plus a sqlite DB at `~/.flor/<projid>.db` — was thin enough for `flor.log` / `flor.dataframe` but too thin for the LLM Access Path. New layout (implemented):

1. **Per-run logs** at `.flor/runs/<tstamp>.jsonl` (microsecond tstamps, one JSON record per line). The whole `.flor/` directory is auto-added to `.gitignore` on first run; data sync is a separate concern.
2. **Reproducibility metadata** (`flor.arg` values, including seeds) lives in the shadow-branch auto-commit message body as `k=v` lines under the `FLOR::Auto-commit::<tstamp>` subject — survives even when log files are gone.
3. **No `~/.flor` state.** Everything is project-local under `.flor/`: query-cache DB at `.flor/<projid>.db`, object store at `.flor/obj_store/<tstamp>/`.
4. **One commit per run is guaranteed** even when source and args are unchanged: each run rewrites `.flor.cmd` (tracked at repo root) with the run tstamp and CLI invocation, which dirties the tree.
5. **`flor unpack` rebuilds the cache** by walking `.flor/runs/*.jsonl` directly — no historical git checkouts.

## Data sync-ing

We can't really have the logs living in git, but we want some measure or reproducibility. That's a balancing act.