# Version 4

## Least Redundancy Overhaul

If a script performs io, like `print` or `logging`, FlorDB will automatically capture the output of subsequent runs, and also be able to parse and integrate historical logs of various formats and layouts. This means you can start using FlorDB with zero code changes, and it will automatically index and structure your logs for easy retrieval and analysis (with minimal duplication).

### Object Store and Flor Checkpointing

In many cases, students will clone a project, which already does torch logging, and fail to do flor checkpointing. This leads to a failure where there shouldn't be any, we can just piggy back off the checkpoints that were already taken. It will take some clever engineering but an elegant solution is possible.

I should be able to checkpoint on a single `flor.loop`, I shouldn't require nested loops for flor profiling (time measurments) and adaptive checkpointing to work.

Implemented (v4):

1. **`torch.save` / `torch.load` are piggy-backed.** Cloned scripts that already call `torch.save` get checkpointed into `.flor/obj_store/` for free; no `flor.checkpointing(...)` block needed. On replay, `torch.load` is redirected to the matching mirror. See [examples/v4/train.py](examples/v4/train.py).
2. **Adaptive trigger works on a single `flor.loop`.** Outermost iteration boundary + time guard (default 60s, tunable via `flor.set_ckpt_interval`). No nesting required. The same throttle gates the `torch.save` mirror to keep disk bounded.
3. **`flor.checkpointing(...)` is now optional** — kept as an explicit-enrollment path for non-torch objects (sklearn, etc.). Profiling records are anchored on `flor.loop` / `flor.iteration` boundaries instead of this block.
4. **Profiling renamed `delta::*` → `time::*`** with `setup` / `iter` / `loop` / `teardown` scopes, plus a new always-emitted `time::script` (total wall time) so flat `flor.log`-only scripts also get a profiling number.

## LLM Ready Data Layout

The legacy layout — a single `.flor.json` overwritten per run plus a sqlite DB at `~/.flor/<projid>.db` — was thin enough for `flor.log` / `flor.dataframe` but too thin for the LLM Access Path. New layout (implemented):

1. **Per-run logs** at `.flor/runs/<tstamp>.jsonl` (microsecond tstamps, one JSON record per line). The whole `.flor/` directory is auto-added to `.gitignore` on first run; data sync is a separate concern.
2. **Reproducibility metadata** (`flor.arg` values, including seeds) lives in the shadow-branch auto-commit message body as `k=v` lines under the `FLOR::Auto-commit::<tstamp>` subject — survives even when log files are gone.
3. **No `~/.flor` state.** Everything is project-local under `.flor/`: query-cache DB at `.flor/<projid>.db`, object store at `.flor/obj_store/<tstamp>/`.
4. **One commit per run is guaranteed** even when source and args are unchanged: each run rewrites `.flor.cmd` (tracked at repo root) with the run tstamp and CLI invocation, which dirties the tree.
5. **`flor unpack` rebuilds the cache** by walking `.flor/runs/*.jsonl` directly — no historical git checkouts.
6. **Self-describing loop context.** Each record's `ctx` is either `null` (run-level) or a flat list of `{name, iteration, value}` segments from outermost to innermost loop. No opaque IDs, no recursive `p_ctx` — nesting depth is `len(ctx)`, and the JSONL is readable end-to-end without a join table. Internally the cache stores `ctx` as JSON-encoded text on `logs`; the old `loops` table and `ctx_id` FK are gone.

## Data sync-ing

We can't really have the logs living in git, but we want some measure or reproducibility. That's a balancing act.

## Replay is a query, not a run (implemented)

`.flor/runs/<tstamp>.jsonl` is the immutable observation of one forward run,
pinned to its `FLOR::Auto-commit::<tstamp>` shadow-branch commit. `flor.replay`
never touches that file — it inserts into the sqlite cache only, tagged
`source='replay'`. Replay rows are user-requested data (the whole point of
`--apply val_acc --override device=cpu` is to *see* the replayed value), so
they ride alongside forward rows in the cache, distinguishable but not
hidden:

- New `logs.source` column (`'forward'` | `'replay'`). Existing DBs are
  migrated via `ALTER TABLE ... DEFAULT 'forward'` — historically all rows
  came from JSONL, which is forward truth, so the backfill is correct.
- `database.unpack(buffer, cursor, source=...)` tags every insert. The
  `flor unpack` CLI and forward `commit()` pass `'forward'`; replay
  `commit()` passes `'replay'`.
- `database.pivot` / `flor.dataframe(...)` surface a `source` column on
  every row so forward and replay observations are visible together. Joins
  across multiple variables stay within a source (because `source` lands in
  the common-columns set for the per-variable merge), so you won't
  accidentally pair a forward `loss` with a replay `val_acc`.
- The repl cost-estimator queries (`time::loop`, `time::iter::n`,
  `time::setup`/`teardown`) *do* filter `source='forward'`. Cost estimation
  must be a forward baseline — a narrowed replay's `time::loop` would be
  misleadingly fast.
- `flor unpack` runs `DELETE FROM logs WHERE source='replay'` before
  walking JSONL, so it actually rebuilds to forward-run truth (previously
  replay rows accumulated and silently survived re-unpack). Replays are
  always reproducible from the historical commit, so wiping them on
  rebuild is loss-free.
- `flor.query(sql)` is a raw pass-through — filter on `source` yourself
  for replay-only or forward-only slices.

## Time logging is heavy, log aggregates (implemented)

`flor.loop` no longer emits one `time::iter` per inner iteration. Per-iter
wall times are collected in memory and summarized at loop exit as three
records anchored on the loop's parent ctx:

- `time::iter` — mean (so `flor.dataframe("time::iter")` keeps returning a
  single time-valued column)
- `time::iter::std` — sample stdev (0.0 when n=1)
- `time::iter::n` — iteration count

For the v4 train example this collapses ~9,400 `time::iter` rows per run
into 6 (one summary triple per loop scope). The repl cost-estimator's
fallback (`flordb/repl.py`) now reads `time::iter::n` directly with
`ctx IS NULL` to recover the outermost iteration count without a
`COUNT(DISTINCT ctx)` scan. `flor.iteration()` (the explicit single-iter
context manager) is unchanged — it's already one record per call.

## --replay_flor syntax and UX

Old surface (v3 and prior):

```
python train.py --replay_flor "loss,val_acc epoch=0,2 step="
```

One flag, one quoted string, positional/keyword detection inside the string,
magic-int defaults (no key → last only, `step=` → skip entirely), hard mutex
with `--kwargs`, name-vs-lineno ambiguity.

New surface (v4):

```
python train.py --replay_flor \
    --apply loss,val_acc \
    --iter epoch=0,2 \
    --iter step=all \
    --override device=cpu
```

- `--replay_flor` is the **mode switch** — presence ≡ replay. `in_replay_mode()`
  is `flags.replay_flor is True`. `--apply` / `--iter` never imply replay.
- `--apply VARS` — comma-separated projection. Linenos use `@N` (e.g. `@42`).
  Absent ≡ no projection (every `flor.log` flows).
- `--iter NAME=SPEC`, repeatable. SPEC ∈ {`all`, `last`, `none`, `0,2,5`}.
  Loops not mentioned default to `last` — adaptive-checkpoint replay needs a
  default jump-target and `last` is the cheapest correct one. First time a
  loop is defaulted, flor prints a one-line tip pointing at the explicit verbs.
- `--override k=v`, repeatable. Replaces the blanket `--kwargs` mutex.
  Allowed for any key not logged in the historical run, plus an explicit
  env-knob allowlist (`cli.ENV_OVERRIDE_ALLOWLIST` = `{device, ckpt_interval_s}`).
  Overriding a historically-logged `flor.arg` that isn't on the allowlist is
  rejected with an error pointing at re-running forward.

### Orchestrator parity (`flor replay`)

The flags above are the *child* surface. `flor replay` spawns those children,
and the two surfaces have to agree:

- **`--override` is forwarded.** `flor replay VARS --override device=cpu`
  passes the pair through to every replayed run. The child still does the
  validation (allowlist + historical-arg check), so a bad override fails in
  the child; a nonzero child exit is now reported per-run instead of silently
  yielding an empty result frame.
- **`@LINENO` is resolved to a log name once, in the orchestrator.**
  `LoggedExpVisitor.linenos` (the inverse of `.names`, and not derivable from
  it — two `flor.log` calls can share a name) turns `@137` into `val_acc`
  before it is used as a schedule column, a child `--apply` value, or a result
  column. Only `backprop` still consumes the raw lineno. Previously `@N` was
  passed through verbatim and matched no log name in the child, so the
  projection filtered out everything the replay computed.
- **`flor.arg` records bypass the `--apply` projection.** Args are run
  configuration, not observations: without them a projected replay row can't
  be joined against the hyperparameters that produced it (and the join is
  source-scoped, so it would drop out entirely).
- **`--override` values are cast to the type they replace.** Historical args
  come back JSON-typed from the run's JSONL; CLI overrides arrive as strings.
  `cli.flags.historical_args` keeps the typed originals so `flor.arg` can
  `duck_cast` the override against them (falling back to the declared default
  for keys with no history).
- **A `flor.arg` the replayed run never logged falls back to its default**,
  with a one-time warning naming the key and pointing at `--override`. Adding
  a new `flor.arg` alongside a hindsight `flor.log` is normal, and it used to
  abort replay with a bare `AssertionError`. A new arg with *no* default still
  raises, but with a message that says what to do about it.
- **Flag detection accepts `--flag=value`.** The argv scan that decides
  whether to run argparse at all used to match bare tokens only, so
  `python train.py --apply=loss` ran forward silently instead of reporting
  that `--apply` requires `--replay_flor`.