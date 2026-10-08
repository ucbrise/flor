# Revision log

## 2026-10-08 — Concurrent recording and distributed replay

Status: design and test plan proposed; implementation pending.

Support concurrent training processes without changing `flor.log`, `flor.arg`,
`flor.loop`, `flor.iteration`, or automatic output capture. Introduce UUID run
identity and a read-only `flor.run_id(...)` selection helper. Checkpoint access
accepts run IDs and preserves unambiguous timestamp lookups.

Implement in three stages: recording and identity migration, launcher
configuration, and distributed replay. Recording must preserve observations
and complete version mappings; hindsight replay must recover the original
job's per-rank numerical results.

### Problem and evidence

[tests/test_ddp_concurrency.py](tests/test_ddp_concurrency.py) reproduces two
failures with real subprocesses in temporary Git repositories:

- Concurrent ranks contend for Git's index. One rank's `git add -A` can stage
  other ranks' JSONLs, while its commit subject names only its own timestamp.
  Runs without a matching commit subject are absent from the mapping used by
  `Schedule.iter_dims()`, which can raise `KeyError`.
- Equal import-time microsecond timestamps share a run identity.
  `orm.to_jsonl()` opens the path with `"w"`, silently overwriting observations.
  Checkpoint shelves share the same collision risk.

`versions.git_commit()` prints and swallows Git errors, so launchers can report
success despite failed finalization. These tests isolate recording races;
actual PyTorch/DDP replay requires the CPU/Gloo tests below. GPU/NCCL coverage
is deferred.

### Give each run a UUID and retain timestamps as metadata

Generate a `uuid4` before the first record or captured checkpoint save and keep
it stable throughout the run. Persist its canonical string as `run_id` in every
new record and use it for `.flor/runs/<run_id>.jsonl`,
`.flor/obj_store/<run_id>/`, database grouping, commit mappings, and replay
selection. Copies retain their ID; new executions in independent clones get
new IDs.

Keep `tstamp` as the unadjusted start time in its existing datetime format,
allowing equal timestamps. Preserve `projid` as a project label and capture
optional Git author name/email at run start. Persist this attribution so later
Git configuration or directory renaming cannot change it. Project names and
Git identities are not uniqueness keys. One shared `job_id` identifies a
distributed launch; each rank has its own `run_id`.

UUIDs avoid timestamp reservation and Git-history scans, but publication must
still reject identity conflicts and preserve existing artifacts. Never rename
an ID after associating records or checkpoints with it. Retries retain the ID
and verify artifact ownership.

#### Selecting a run without copying UUIDs

Add the read-only helper:

```python
flor.run_id(run=None, *, tstamp=None, job_id=None,
            args=None, metadata=None, **arg_filters)
```

`run` accepts a full run ID, a timestamp string or datetime, or a dataframe row
with `run_id` (falling back to `tstamp` for legacy rows). `tstamp` explicitly
selects by timestamp and cannot be combined with `run`. `job_id` narrows the
selection to one distributed launch.

Additional keywords filter recorded Flor arguments, such as `lr=0.01` or
`seed=42`. The `args` mapping supports arbitrary names, including `flor::rank`
and application arguments that overlap helper parameters, such as `job_id`.
`metadata` filters recorded fields such as `projid`, `filename`, `author_name`,
and `author_email`. These separate namespaces preserve the meaning of user
arguments named `rank` or `author_email`.

Combine all criteria with AND and require at least one selector. Reject names
supplied in both `args` and `arg_filters`. Match exactly one distinct run:
rows in `flor.dataframe()` with the same `run_id` count as one run, including
rows for different loop iterations and rows with `source = 'forward'` or
`source = 'replay'`.

Argument filters use typed values from the original forward run with exact
equality, no string-to-number conversion, and no floating-point tolerance.
Repeated equal argument records count once; missing values do not match.
A filtered argument with multiple distinct values in a run being considered
raises `ValueError` identifying that run. Replay overrides and hindsight metrics
do not alter selection attributes. For ranges or evolving metrics, filter a
dataframe first and pass the selected row.

Return the canonical ID for one match; raise `LookupError` for none and
`ValueError` for ambiguity, listing matching IDs, timestamps, jobs, and relevant
argument values. Invalid combinations, unknown metadata fields, and malformed
values raise `ValueError`. Never choose the latest match implicitly or create
a run or ID through this helper; recording uses a separate context accessor.

```python
# Select a dataframe row and load its checkpoint.
rid = flor.run_id(row)
state = flor.load_checkpoint(rid, "ckpt.pth")

# Repeated trials may require additional filters.
rid = flor.run_id(lr=0.01, seed=42)
rid = flor.run_id(lr=0.01, metadata={"author_email": "alice@example.org"})

# Select a rank within a job.
rid = flor.run_id(job_id=job_id, args={"flor::rank": 0})

# Select by timestamp when it identifies one run.
rid = flor.run_id(tstamp=row.tstamp)
```

Expose `run_id`, `tstamp`, and `job_id` in `flor.dataframe()` and `flor.io()`.
Launcher arguments use normal dataframe argument columns; join IO rows to them
by `run_id`. Group runs by `run_id`, including equal timestamps, and use
`drop_duplicates("run_id")` to select one row per run. Retain timestamp
filtering and sorting.

Make `flor.checkpoints(...)` and `flor.load_checkpoint(...)` share these
resolution rules while preserving their positional call shape and `tstamp=`
keyword. Timestamp calls succeed when unique and report ambiguity otherwise.

#### Storage migration and historical runs

Add an explicit schema version and `run_id` to new records. Update ORM readers,
SQLite schema/grouping, `flor unpack`, checkpoint lookup/lineage, commit parsing,
and replay scheduling together. Cache rebuilding preserves persisted IDs;
UUID filenames have no temporal ordering, so sort by timestamp metadata.

Mark `flor.arg` and captured launcher argument records explicitly, retaining
their log-value representation and dataframe visibility. Currently `flor.arg`
calls `flor.log`, so unmarked records cannot reliably distinguish arguments
from metrics. Rebuild the typed argument index from marked JSONL records and
use this provenance for selection and replay initialization. The index is a
cache; JSONL remains the source of truth.

Read legacy timestamp-named files without rewriting historical commits.
Derive IDs in a `legacy:` namespace from the SHA-256 digest of original JSONL
bytes: identical copies share an ID, while different artifacts with equal
timestamps remain distinct. Keep explicit locators for legacy JSONL and
checkpoint paths. Preserve this mapping across branch changes, cloning, and
cache rebuilding.

Resolve old timestamp commit subjects through legacy artifacts, verifying
content when timestamps collide. Keep direct timestamp access to unambiguous
legacy checkpoint-only shelves; without a recorded run, the helper cannot
resolve a run ID. Ambiguous checkpoint ownership or commit mappings require
explicit association. This migration cannot recover already lost observations.

Recover legacy argument names from the matching auto-commit's argument body
and typed values from its JSONL only when they identify one value. Missing or
conflicting evidence makes argument selection unavailable for that run;
explicit ID/timestamp selection remains available. Do not infer arguments from
all run-level metrics.

#### Allocation timing and run boundaries

Use one lazy internal context accessor holding `run_id`, `tstamp`, and owning
PID across recording paths. Keep `Clock.get_datetime()` as a timestamp accessor
and profiling timers independent of identity. Import and historical queries
must not allocate a run ID.

Create the context before `_emit_io()` constructs its first `orm.Log`:
currently that record precedes `_register_run()` and `_deferred_init()`, so
deferred initialization is too late. Explicit logging, loop profiling, and
checkpoint capture share the accessor. For a captured file checkpoint, create
it before the original save, even without prior logging. Exclude initialization
diagnostics from automatic capture to prevent recursion.

Close the active identity after durable observation publication and saving
pending finalization metadata. Git retries use that saved identity. Clear the
active context at the run boundary; the next recorded event, including the next
interactive run, creates a new one. Replay uses the selected historical run ID
and timestamp without allocating a forward-run identity.

#### Fork policy

Initially support independent interpreters, including `torchrun` ranks and
Python spawn workers. Children inheriting Flor through `fork` have recording
and replay hooks disabled until a fresh interpreter starts, preventing copies
of parent records or use of its replay identity. Fork-launched training ranks
are outside the initial scope.

An after-fork handler discards inherited recording state and disables the
child. Check the importing PID at recording, checkpoint-hook, and finalization
entry points before touching inherited locks or buffers. Disabled children:

- Forward output and execute original checkpoint save/load functions.
- Preserve `flor.arg` return values, `flor.loop` iteration, and explicit
  historical checkpoint reads.
- Create no records, IDs, checkpoint copies, cache writes, or commits; cleanup
  returns immediately.
- Emit one uncaptured notice on the first attempted recording operation.

Fresh interpreters initialize their own contexts. Close inherited project-lock
descriptors without explicitly unlocking the parent's lock.

### Coordinate repository changes and preserve every version mapping

Serialize shadow-branch creation, ignore-file and command-file updates, and
finalization with a project-level interprocess lock at
`.flor/locks/project.lock`. Use an OS-managed advisory lock released on process
exit. File existence does not imply ownership; never unlink it during normal
operation. Run ID generation needs no lock. Hold the lock across Git's dirty
check, staging, and commit; leave Git's own `index.lock` to Git.

One rank can stage all completed JSONLs, leaving another with a clean tree.
Use one `FLOR::Auto-commit::run::<run_id>` subject per new run, allowing an empty
commit when another rank already committed its observations. Keep arguments
in the commit body and read historical timestamp subjects through the
compatibility reader. Build `run_id -> commit` mappings without date parsing;
retries must not duplicate finalization commits.

Keep SQLite transactions short and commit and close them before Git. Preserve
[tests/test_commit.py](tests/test_commit.py) interruption behavior, serialize
schema initialization, and bound write contention.

A future tracked job manifest could map several runs to one commit, requiring
coordinated mapping-reader and replay-selection changes. Rank-zero-only
committing still requires mappings for every rank; rank-zero-only recording
would discard observations.

### Publish complete records and recover from finalization failures

Write JSONL to a temporary file outside Git's tracked run set, then publish
atomically without replacing another run's file. Atomic replacement alone does
not prove ownership.

Git finalization must return a meaningful result or propagate an error. Persist
pending finalization metadata, including arguments, separately from the buffer
so retries finish versioning without duplicating observations or SQLite rows.
JSONL is the observation source of truth; SQLite is a rebuildable cache.

Explicit `flor.commit()` can propagate failures. Automatic shutdown requires
a tested mechanism that gives launchers a nonzero exit status: raising from
`atexit` is insufficient. Durable pending status enables recovery but does not
satisfy this exit-status requirement.

### Capture launcher configuration as Flor arguments

Rank identifies a worker in a distributed group. The torchrun integration maps
[`RANK`, `LOCAL_RANK`, and `WORLD_SIZE`](https://docs.pytorch.org/docs/2.14/elastic/run.html#environment-variables)
to proposed Flor arguments `flor::rank`, `flor::local_rank`, and
`flor::world_size`. The generic recorder, storage, and selection helper treat
them like other arguments; they are optional for ordinary runs.

At context creation, record available launcher values once as integers through
the internal argument path shared with `flor.arg`. Persist them in JSONL and
the commit's argument body. No PyTorch import or explicit logging is required.
Leave absent values absent and reject malformed or inconsistent configuration.

Reserve these names for Flor and reject application writes or CLI overrides.
A user argument or metric named `rank` stays independent. Initialize the
context before emitting arguments, without recursively calling public
`flor.arg`. Use argument records rather than separate rank fields or schemas.

Distribute one shared `job_id` per forward execution attempt. Use launcher
identity only if it distinguishes that attempt; otherwise a coordinator
creates and distributes a job UUID. Do not generate job IDs independently in
each rank or infer grouping from timestamps, project names, Git authors, or
rank numbers. Retain insufficiently unique launcher IDs as metadata. Replay
keeps the historical job ID and uses a separate attempt ID for completion and
retry tracking.

During replay, the launcher creates worker topology; Flor validates assigned
rank/world size against recorded arguments before training. Local-rank
placement must match the supported topology. Reading arguments alone cannot
configure workers. Separate checkpoint shelves protect Flor's copies but do
not prevent ranks overwriting a shared checkpoint source path.

### Select historical runs and replay distributed jobs together

The current orchestrator launches one process whose replay initialization
reads the latest JSONL. Instead, pass the selected historical run ID to each
child and load its exact arguments, starts, and observations. This is required
in stage 1 because one commit can contain several run files.

Schedule a distributed job as a unit and bind children to historical ranks.
Selecting one rank's results may still require all participants. Reconstruct
launcher settings with fresh rendezvous resources and preserve collective
participation when narrowing Flor loops; ranks cannot omit computation other
ranks depend on.

Bound rendezvous and execution time, report failures from any rank, and
terminate and reap remaining children after failure. Track completion for the
whole attempt, exclude partial failed results, avoid duplicate results on
retry, and restore the working branch on success and failure. Retain the
existing replay API with additive launcher settings and internal orchestration
based on job metadata.

### Implementation stages and acceptance criteria

1. **Recording and identity migration.** Implement UUID contexts, schema and
   legacy readers, argument provenance, `flor.run_id(...)`, atomic publication,
   coordinated finalization, failure recovery, and explicit run IDs for
   single-process replay. Preserve recording signatures and timestamp lookup.
   Pass the identity/selection tests below. Exercise Git contention with
   explicit synchronization and inspect completed commits; `git ls-files`
   proves only index membership. Verify launcher-visible finalization failures
   and recovery without duplicate observations, cache rows, or commits.
2. **Launcher arguments and rank binding.** Persist reserved arguments and job
   identity; bind ranks to historical IDs. Test job/argument selection and
   restoration of an older job's metadata and exact per-rank runs when the
   checked-out commit contains several JSONLs.
3. **Distributed replay.** Implement job scheduling, topology reconstruction,
   compatible loop narrowing, and bounded failure handling. Pass the numerical,
   selection, recovery, and isolation tests below.

In stage 1, retain equal timestamps in the same-microsecond fixture but assert
separate run IDs, logs, database groups, and checkpoints. Use run-ID-to-commit
assertions for new runs and dedicated timestamp coverage for legacy runs.
Remove strict `xfail` markers as requirements are implemented. Replace silent
collision expectations with preserved-observation assertions; successful exits
remain valid when data is preserved.

### Identity lifecycle and selection test plan

Add these cases to the concurrency tests. Identity, selection, and migration
belong to stage 1; launcher arguments and job binding belong to stage 2.

- **Identity and copies:** force equal, unchanged timestamps in simultaneous
  processes; verify distinct IDs consistent across records, filenames, database
  rows/groups, checkpoint shelves, and commit subjects. Repeat in independent
  clones sharing project name and Git author, then combine histories and check
  observations/mappings. Copies retain their ID after import/cache rebuilding.
- **Lifecycle:** record captured output first, including without explicit Flor
  calls, and separately capture a checkpoint before any output/logging. Verify
  the first artifact and later records share the final ID. Import, historical
  queries, and the helper create no identities/artifacts. Two interactive cycles
  with equal timestamps get separate stable IDs; a Git retry retains its ID.
- **Publication:** interrupt publication and verify no partial final JSONL
  appears. Retry under the saved ID; injected UUID destinations preserve
  conflicting logs and checkpoint copies with explicit failure. Successful
  retries verify ownership.
  UUID-generation failure aborts recording without timestamp fallback.
- **Migration:** exercise mixed formats, stable legacy IDs across branch changes,
  clones, and cache rebuilds, plus historical commits/checkpoints. Different
  legacy artifacts with equal timestamps get distinct IDs; report ambiguous
  ownership. Test checkpoint-only timestamp access separately and run existing
  forward, replay, and checkpoint workflows against both formats.
- **Selection:** cover full ID, dataframe row, timestamp, `lr`/`seed`, attribution,
  and job plus `args={"flor::rank": 0}`. Check identical trials, no matches,
  timestamp ambiguity, contradictory/duplicate filters, malformed values,
  unknown metadata, and no selector. Verify typed equality, missing/conflicting
  values, and names overlapping helper parameters. Repeated rows for one run
  cause no ambiguity; never choose the latest implicitly. Checkpoint APIs accept
  IDs and unambiguous legacy selectors, including `tstamp=`.
- **Argument provenance:** distinguish `flor.arg("lr", ...)` from a metric named
  `lr`; select only forward arguments regardless of replay overrides. Repeat
  after cache rebuilding. Test legacy recovery and unavailable-provenance errors.
- **Launcher arguments:** without explicit rank logging, emit reserved arguments
  once as integers, preserve user `rank`, leave absent values absent, and reject
  reserved writes/overrides. Reject replay rank/world-size mismatches before
  training.
- **Fork:** fork before/after context creation, with buffered records and while
  holding the project lock. Verify disabled recording, ordinary computation and
  output, no parent publication/ownership changes, and cleanup that neither
  releases the parent's lock nor retains it after parent exit. Spawned workers
  record under separate IDs. Skip only fork cases on platforms without `fork`.

### Distributed replay test plan

Stages 2–3 target single-host CPU jobs with real interprocess communication.
GPU/NCCL and multiple hosts follow later. Acceptance requires recovering the
forward job's per-rank values; successful exits, files, and mappings alone are
insufficient.

1. **Deterministic fixture:** launch two Gloo ranks with fixed seeds, synthetic
   inputs, and fresh rendezvous resources. Start with an exactly known
   `all_reduce`, then a tiny DDP model with fixed initialization and a short
   training loop. PyTorch remains optional elsewhere; the distributed test
   environment must install it and run these tests.
2. **Numerical equivalence:** save forward reference values for each rank and
   iteration in test-owned artifacts outside the repository. Add a hindsight
   `flor.log` and invoke the actual replay orchestrator. Compare with exact
   equality for the collective and explicit tolerance for training. Check
   historical job/rank ownership with no missing or duplicate ranks.
3. **Selection and narrowing:** record several jobs with different arguments,
   select an older one, and restore its exact arguments and checkpoint lineage.
   Cover full replay, selected-iteration logging, and one-rank results requiring
   all participants. Execute required computation/collectives; unselected
   iterations emit no hindsight rows. Reject incompatible selections before
   launch.
4. **Failure and recovery:** deliberately fail a rank and separately prevent
   rendezvous. Report failure and terminate/reap peers within a deadline. Check
   whole-attempt completion, exclude partial rows, retry without duplicates, and
   preserve historical observations. Git finalization failures belong to stage 1
   because replay creates no forward runs.
5. **Isolation and repeatability:** replay a job twice; also record two jobs
   concurrently and replay them sequentially in the shared tree. Keep identities,
   rank bindings, rendezvous resources, and results separate. Compare historical
   JSONL/checkpoints before and after, restore branches on success/failure, and
   clean up children and rendezvous artifacts.

Use explicit synchronization and controlled fault injection. Give every
subprocess group a deadline and `finally` cleanup that terminates and reaps
children. Retain per-rank stdout/stderr and report phase, job, and rank on error
or timeout. Allocate separate rendezvous resources for concurrent recording
jobs and fresh resources for every replay. Concurrent replay in one working
tree is deferred; it requires isolated checkouts or serialized checkout and
execution.

Run tests in order: collective, DDP equivalence, selection/narrowing,
failure/recovery, then concurrent-job isolation. Keep GPU-free recording tests
separate so regressions remain diagnosable without PyTorch or distributed
initialization.
