# Version 4

## Least Redundancy Overhaul

If a script performs io, like `print` or `logging`, FlorDB will automatically capture the output of subsequent runs, and also be able to parse and integrate historical logs of various formats and layouts. This means you can start using FlorDB with zero code changes, and it will automatically index and structure your logs for easy retrieval and analysis (with minimal duplication).

### Capturing live runs (implemented)

`flordb/capture.py` tees `sys.stdout` / `sys.stderr` and wraps `logging`
dispatch, installed from `flordb/__init__` at import time — not from
`_deferred_init()`, because the whole point is the script whose only flor
reference is `import flordb`, which never reaches a first flor call.

Captured lines become ordinary records on `output_buffer`, so JSONL writing,
`ctx` addressing, and forward/replay `source` tagging all come for free:

- **Channels** ride in `value_name`: `io::stdout`, `io::stderr`, and
  `io::log::<level>` (`io::log::info`, …). `LIKE 'io::%'` selects all captured
  io; the logger name prefixes the value when it isn't `root`.
- **`value_type = 2`** (`VALUE_TYPE_IO`) keeps io out of `flor.dataframe()`,
  whose pivot filters `value_type = 1`. Reading it back is `flor.io()`, or
  `flor.dataframe("io::stdout")` to treat it as a column like any other.
- **Narrowing is shared with `flor.log`.** Both go through `api._recording`, so
  captured io obeys the `--apply` projection (`--apply loss,io::stdout` to
  include it) and is suppressed on logical-replay fast-forward iterations.

"Least redundancy" is the constraint that shapes the implementation. Two paths
would otherwise record the same observation twice:

1. `flor.log` echoes every value to the terminal itself. That `tqdm.write` now
   runs inside `capture.muted()`, as do flor's other user-facing messages
   (converted to `capture.flor_print`) and flor's own progress bars (given the
   raw stream via `capture.raw_stderr()`).
2. A `logging` record that reaches a `StreamHandler` also lands on stderr.
   Registering a `logging.Handler` would put flor in handler order *beside*
   that one, so the tee would see its write and record a second row. Instead
   `logging.Logger.handle` is wrapped: the record is captured once, from the
   structured side where `levelname` and the logger name still exist, and the
   entire downstream dispatch runs muted. Level and filter decisions have
   already been made by the caller, so the user's logging config is respected.

Volume is bounded by construction. A repainting progress bar emits thousands of
`\r` fragments and no newline; only the text after the final carriage return is
still on screen, so a bar collapses to one record at close. Lines are truncated
at `max_line`, runs are capped at `max_records`, consecutive identical lines
within one loop iteration collapse to one row (the next iteration has a
different `ctx`, so it survives), and `flor.set_capture(...)` / `FLOR_CAPTURE=0`
turn any of it off.

**Structuring is opt-in.** `capture.extract_pairs` recognizes `k: v` / `k=v`
where the value is numeric, and `flor.set_capture(extract=True)` promotes hits
to real `value_type=1` metric rows *alongside* the raw line — raw is truth,
promotion is derived. It is off by default because a false positive silently
invents a column in `flor.dataframe`. The regex lookarounds reject the cases
that matter (`http://host:80`, `2026-08-13T11:27:06`, `acc: 90%`,
`ckpt=/tmp/x.pth`), and `flor capture --preview` dry-runs the extractor over
the io already in the cache so the user sees the false-positive rate on their
own logs before enabling it.

**Captured io registers the run.** `commit()` is reached from an `atexit` hook
that returns early unless `skip_cleanup` has been flipped, and only
`_deferred_init()` flips it — from `log` / `arg` / `loop` / `iteration`. A
script whose only flor reference is `import flordb` reaches none of those, so
before `api._register_run` the headline case recorded its io into
`output_buffer` and then dropped the entire run at exit. The init is *reported*
rather than raised on failure: it runs underneath a user `print`, inside the
tee's catch-all, so an exception there would be swallowed silently.

That in turn means capture must stay off wherever a `print` shouldn't amount to
a run: `python -c '...'` and the bare REPL (`SCRIPTNAME` of `-c` or `""`) are
skipped alongside the `flor` subcommands, or every ad-hoc query would record a
run and add an auto-commit.

Known limitation: capture is off under IPython, which swaps `sys.stdout` per
cell and routes results through its own displayhook. `flor.log` is the
interactive path.

**Still open** (the other half of this section): parsing and integrating
*historical* logs — a `flor ingest slurm-*.out` that turns log files produced
before FlorDB was in the picture into `.flor/runs/*.jsonl`. The record shape
above is what it would target.

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

1. **Per-run logs** at `.flor/runs/<tstamp>.jsonl` (microsecond tstamps, one JSON record per line), committed alongside the run that produced them. The rest of `.flor/` is auto-added to `.gitignore` on first run — see "Data sync-ing" below.
2. **Reproducibility metadata** (`flor.arg` values, including seeds) lives in the shadow-branch auto-commit message body as `k=v` lines under the `FLOR::Auto-commit::<tstamp>` subject — survives even when log files are gone.
3. **No `~/.flor` state.** Everything is project-local under `.flor/`: query-cache DB at `.flor/<projid>.db`, object store at `.flor/obj_store/<tstamp>/`.
4. **One commit per run is guaranteed** even when source and args are unchanged: each run rewrites `.flor.cmd` (tracked at repo root) with the run tstamp and CLI invocation, which dirties the tree.
5. **`flor unpack` rebuilds the cache** by walking `.flor/runs/*.jsonl` directly — no historical git checkouts.
6. **Self-describing loop context.** Each record's `ctx` is either `null` (run-level) or a flat list of `{name, iteration, value}` segments from outermost to innermost loop. No opaque IDs, no recursive `p_ctx` — nesting depth is `len(ctx)`, and the JSONL is readable end-to-end without a join table. Internally the cache stores `ctx` as JSON-encoded text on `logs`; the old `loops` table and `ctx_id` FK are gone.

## Data sync-ing (implemented)

The balancing act resolved once the three things under `.flor/` were measured
separately — they differ by three orders of magnitude, and only one of them is
irreplaceable:

| Tier | Per run | Recomputable? | Decision |
|---|---|---|---|
| `runs/*.jsonl` | 12KB–2.4MB raw, **~69KB packed in git** | No | **Commit it** |
| `obj_store/` | ~19MB | Yes, by replaying | Ignore |
| `<projid>.db` | ~760KB | Yes, `flor unpack` | Ignore |

Two measurements drove this:

1. **Git packs run JSONL at 15.4×** (5 runs, 5.35MB → 347KB after `gc`),
   including delta compression across structurally-identical runs. Ten students
   × 100 runs/semester ≈ 70MB. That tier is free, so it goes in git and syncs
   by `git fetch` — no server, no bucket, no egress bill for a team spread
   across the country.
2. **Reshaping the record to fight redundancy isn't worth it.** Hoisting the
   per-line `projid`/`tstamp`/`filename` into a header cuts uncompressed size
   39% but compressed size only 3.5%. Compression already eats that redundancy,
   so the self-describing record shape stays as it is.

A third measurement closed off the obvious alternative for the heavy tier:
**checkpoints don't dedup** (21 mirror files, 21 distinct hashes), so
content-addressed storage — git-annex, DVC, restic-style chunking — buys
nothing here. Object storage stays a future optimization; if it ever lands,
the lever that matters for this topology is *egress* pricing, not storage rate.

`versions.ensure_gitignored()` writes `.flor/*` + `!.flor/runs/` rather than a
bare `.flor/`: git does not descend into an excluded directory, so a re-include
underneath one is dead. Repos carrying the pre-v4 `.flor/` line are migrated in
place — the legacy line is removed, not appended to.

### Replay warms the object store

Not syncing checkpoints means a fresh clone has nothing to restore from, so
replay recomputes — and now keeps what it recomputed. `_ckpt_warming_enabled()`
gates it, and two rules keep a reconstruction from being mistaken for truth:

- **Never overwrite.** An existing mirror is forward-run truth; a warmed one is
  a reconstruction, and truth wins every collision (`ckpt(only_if_absent=)`,
  `obj_store.has_shelved`).
- **Never warm under an override that could move the numbers.** The
  reconstruction is only sound because the replay re-ran the same code over the
  same args. `ckpt_interval_s` is numerics-neutral and stays allowed;
  `device=cpu` is not (cpu/cuda kernels don't agree bit-for-bit) and disables
  warming.

`_build_outer_replay_plan` no longer aborts when *no* mirror exists at or
before the target. It replays from iteration 0 instead, which is sound because
the seed is a `flor.arg` restored from the historical run.

That soundness has one prerequisite, and getting it right is the subtle part.
The user's resume block runs at *module scope*, where `layers` is still empty —
so `_flor_torch_load`'s per-iteration redirect cannot reach it, and it loads
whatever `ckpt.pth` holds, which after a forward run is **end-of-run** weights.
Replaying from zero on top of that would report wrong numbers and then warm
them into the shelf as if they were truth.

`_neutralized_resume_state` fixes it without touching the user's files: when
the shelf has no mirror for the resume path, flor's `torch.load` hook returns
each target's *own current* `state_dict()`, so `model.load_state_dict(...)`
becomes a genuine no-op and the seed-initialized weights survive to the loop.
When the shelf does have mirrors, the block is left alone — the per-iteration
restore overwrites it anyway. If a resume shape can't be neutralized faithfully
(a target flor can't snapshot), it returns None and the plan builder refuses
rather than guessing, which is what `_resume_neutralized` gates.

`flor.checkpointing`-enrolled (non-torch) objects reach the same place by a
shorter road: they have no module-scope resume block to neutralize, so the
script's own initialization *is* the correct starting state. What they were
missing was the fallback itself — `_mirror_exists_at` only consulted the torch
resume spec, so an enrollment-only script took the `resume is None` early-out,
skipped the fresh-clone path entirely, and died inside `deserialize`. It now
checks both restore paths (`obj_store.has_shelved` for every enrolled name), and
`_restore_at` anchors both at a fast-forward's starting mirror.

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

The orchestrator (`python -m flordb replay`) takes the same verbs:

```
python -m flordb replay --apply grad_norm --where "epoch > 2" \
    --iter epoch=0,2 --override device=cpu
```

`--iter` / `--override` are forwarded to each replayed child process. The
pre-v4 positional form (`flor replay grad_norm "epoch > 2"`) still parses;
`cli.resolve_replay_args` reconciles the two and rejects mixing them, because
a stray positional next to `--apply` would land in the VARS slot and get
replayed as if it were a variable name.

## flor.iteration under replay (implemented)

`flor.iteration(name, idx, value)` is the explicit single-iteration marker for
scripts that drive their own loop (or run one process per iteration). It used
to abort with a bare `raise` under `--replay_flor`; it now behaves like the
outermost `flor.loop`:

- restores that iteration's state on entry — enrolled `flor.checkpointing`
  objects first, then the AST-detected torch resume block;
- never *overwrites* the object store (`obj_store.get_shelf()` is keyed on the
  *historical* tstamp during replay, so an unconditional write would clobber
  the mirror it is reading from). It does warm the gaps — see "Replay warms the
  object store" above;
- narrows by suppressing logs rather than skipping work. Flor doesn't own the
  iteration space here — it can't enumerate the iterations ahead of time, and
  can't skip the body of a `with` block — so `--iter name=0,2` decides which
  iterations *record*, and the body always runs. `--iter name=last` isn't
  decidable in this mode and logs everything with a one-line notice.

Mirror addressing (`_layer_for`) is now shared between `flor.loop` and
`flor.iteration`, and reproduces the forward run's `layers` entry exactly:
index `k`, value stringified only when jsonable. It previously wrote `k + 1`,
which silently mismatched the forward filename for any outer loop over
non-jsonable values (a dataloader, say).

## Tests

`tests/` holds a pytest suite: unit coverage for the CLI verbs, `layers` /
filename addressing, the sqlite schema and its migrations, the AST visitors,
and the replay narrowing rules; plus `-m slow` end-to-end tests that spawn real
forward runs and replays in throwaway git repos (`make test` / `make
test-fast`). The torch tests assert that a narrowed replay recovers the *same*
value the forward run logged, which is the actual contract of mirror
addressing and of logical replay's fast-forward.

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