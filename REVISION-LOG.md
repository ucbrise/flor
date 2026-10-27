# Revision log

## 2026-10-08 — Concurrent recording and distributed replay

Status: design and test plan proposed; implementation pending.

Review and open decisions: [Multiprocessing design review](docs/multiprocessing-review.md).
This revision entry defines the proposed requirements and acceptance tests;
the review tracks readiness, unresolved design details, and decisions to make
before implementation. Recommendations in the review are not accepted changes
to this plan until recorded here.

These repository documents are the ground truth for subsequent review rounds;
resolved scope decisions carry forward unless the project owner revisits them.

Target workloads are PyTorch/Hugging Face model training and serving. The
[workload analysis](docs/model-workloads.md) maps natural multi-GPU alternatives
to current support and the accepted first-release scope below.

Support concurrent training processes without changing `flor.log`, `flor.arg`,
`flor.loop`, `flor.iteration`, or automatic output capture. Introduce UUID run
identity and a read-only `flor.run_id(...)` selection helper. Checkpoint access
accepts run IDs and preserves unambiguous timestamp lookups.

Implement in three stages: recording and identity migration, launcher
configuration, and distributed replay. Recording must preserve observations
and complete version mappings; hindsight replay must recover the original
job's per-rank numerical results.

### Scope decision — 2026-10-27

D1 is resolved: the first release targets single-host PyTorch DDP through
`torchrun`, Hugging Face Trainer/Accelerate using DDP, DataLoader-worker
compatibility, and independent concurrent training trials. CPU/Gloo is a
development milestone; actual two-GPU CUDA/NCCL validation is required before
claiming multi-GPU support.

Keep process-safety and observation-ownership work needed by supported helper
workers. Defer programmatic-spawn training integration, FSDP/DeepSpeed,
tensor/pipeline parallel training, multiple hosts, and online serving to separate
integrations. Finite model evaluation and
offline inference are the first serving-related follow-on direction.

Define supported helper start methods, lifecycle, observation ownership, and
replay behavior, plus Hugging Face instrumentation and model-save formats,
during implementation design. Unsupported operations must fail explicitly;
deferral does not permit silently dropping child observations. D2 is resolved
below; D3–D5 remain open in the review.

### Launch integration decision (D2) — 2026-10-27

D2 is resolved: require `flor run -- <launcher and arguments>` for supported
distributed recording in the first release. Add a `flor` console entry point
for the existing CLI and one new `run` command. Extend the existing `replay`
command for distributed hindsight execution, sharing supervision internally;
users do not manually relaunch historical ranks. Implementation is pending.

```sh
flor run -- torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py --batch-size 32
flor run -- accelerate launch --multi_gpu --num_processes=2 train.py
```

Keep recording calls and existing single-process commands unchanged. Preserve
concurrent independent-trial recording safety. Limit `run` options to Flor
supervision controls, such as an optional execution deadline; exact flags are
engineering choices. Pass the command after `--` unchanged while validating
supported launcher configurations. GPU count, strategy, rendezvous, and script
arguments remain in their existing interfaces. Preserve the working directory,
application environment, and output streams, adding Flor coordination metadata.
Do not treat delegated command arguments as Flor wrapper options.

Support tested fixed-size, single-host torchrun and Accelerate DDP
configurations. Reject unsupported configurations, elastic membership, and
automatic worker restarts explicitly before training. An unintegrated
distributed launch must report an actionable unsupported-launch error. The
command syntax does not promise support for arbitrary launchers or backends.

Flor's supervisor creates and distributes a fresh shared job UUID before
workers start and persists expected training-rank membership. Helpers follow
their defined ownership and completion rules without counting as extra ranks.
Success requires both a successful launcher exit and durable finalization of
every expected rank, including required helper observations and Git mappings.
Missing completion or fatal capture/finalization errors must produce a nonzero
Flor result even if the launcher exits zero. Worker status reporting and
completion validation are required; the outer wrapper alone is insufficient.

On failure, interruption, or timeout, terminate and reap remaining children
within a deadline, report phase/job/rank where available, and preserve pending
artifacts for recovery. A new training execution receives a fresh job UUID;
finalization retries retain the saved identities. Replay retains historical
job/run bindings and uses a fresh replay-attempt ID.

Keep the public surface to the new `run` command and distributed extensions to
`replay`, preserving `capture`, `query`, `dataframe`, `unpack`, and `stat`.
Separate `init`, `start`, `stop`, `worker`, and `commit` commands are not
required. Pending-finalization discovery and retry still need a documented,
testable interface during recovery design; `unpack` is not Git recovery.

Deliver minimum supervision/completion handling needed to prove recording
failure reporting in stage 1, the public CLI and launcher adapters/job
membership in stage 2, and distributed replay orchestration in stage 3.
See the launch integration acceptance tests below and the
[accepted review contract](docs/multiprocessing-review.md#d2--launch-integration).

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
actual PyTorch/DDP replay requires the CPU/Gloo development tests and GPU/NCCL
release validation below.

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
and replay scheduling together. Cache rebuilding preserves persisted IDs.

Mark `flor.arg` and captured launcher argument records explicitly, retaining
their log-value representation and dataframe visibility. Currently `flor.arg`
calls `flor.log`, so unmarked records cannot reliably distinguish arguments
from metrics. Rebuild the typed argument index from marked JSONL records and
use this provenance for selection and replay initialization. The index is a
cache; JSONL remains the source of truth.

Read legacy timestamp-named files without rewriting historical commits.
Keep explicit locators for legacy JSONL and checkpoint paths.
Preserve this mapping across branch changes, cloning, and cache rebuilding.

Recover legacy argument names from the matching auto-commit's argument body
and typed values from its JSONL. Missing or
conflicting evidence makes argument selection unavailable for that run;
explicit ID/timestamp selection remains available.


#### Allocation timing and run boundaries

Use one lazy internal context accessor holding `run_id`, `tstamp`, and owning
PID across recording paths. The PID identifies the OS process that owns the
in-memory context and detects forked children; rank identifies a worker within
a distributed job. Neither is a persistent run identity: `run_id` is the UUID
used for that purpose. Keep `Clock.get_datetime()` as a timestamp accessor and
profiling timers independent of identity. Import and historical queries must
not allocate a run ID.

Create the context before `_emit_io()` constructs its first `orm.Log`:
currently that record precedes `_register_run()` and `_deferred_init()`, so
deferred initialization is too late. Explicit logging, loop profiling, and
checkpoint capture share the accessor. For a captured file checkpoint, create
it before the original save, even without prior logging. Exclude initialization
diagnostics from automatic capture to prevent recursion.

When a recording run ends, save its observations, run ID, and the metadata
needed to finish its Git commit. Once these are durably saved, clear the
in-memory context for that run.

If the Git commit fails, retry it using the saved run ID.

Each new forward run gets a new UUID when recording begins, including in an
interactive session. Replay reuses the selected recorded run's ID and
timestamp; it does not allocate a new run ID.

#### Framework helper workers

Apply recording safety and ownership rules to supported DataLoader workers;
they are not additional DDP ranks. Define and test their supported start methods,
lifecycle, seeding, observation ownership, and replay behavior in stages 2–3.

Keep parent observations owned by the parent. Helpers must not republish
inherited buffered records, release the parent's project lock, or finalize the
parent's pending Git work. Check process ownership before using inherited
buffers or locks, and recreate process-local resources where needed. Preserve
application stream writes and define ownership for unfinished captured output.

Finalize helper observations on orderly worker shutdown, including
multiprocessing exit paths, and report failures to the training rank or launcher. Do not rely
only on `atexit`: [`os._exit()` bypasses cleanup handlers](https://docs.python.org/3/library/os.html#os._exit).
Preserve deterministic input delivery and replay's checkpoint write protection.
If safe initialization or supported replay behavior is unavailable, report an
explicit failure; disabling hooks or silently dropping observations does not
satisfy the contract. Independently launched training ranks initialize their
own contexts.

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
satisfy this exit-status requirement. Under D2, the Flor supervisor must check
durable worker completion as well as child exit status. Fatal automatic-capture
errors and missing completion must fail the supervised execution.

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

Under D2, the Flor supervisor creates and distributes one fresh shared `job_id`
UUID per forward execution attempt before workers start and records expected
training-rank membership. Do not generate job IDs independently in
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
based on job metadata. Extend the existing `replay` command and share the
supervision machinery with `run` internally, retaining historical bindings and
using a fresh replay-attempt ID.

### Implementation stages and acceptance criteria

1. **Recording and identity migration.** Implement UUID contexts, schema and
   legacy readers, argument provenance, `flor.run_id(...)`, atomic publication,
   coordinated finalization, failure recovery, supported helper-worker
   recording safety and lineage, and explicit run IDs for single-process replay.
   Implement the minimum supervision/completion mechanism needed to verify
   durable worker finalization and propagate recording failures under D2.
   Preserve recording signatures and timestamp lookup.
   Pass the identity/selection tests below. Exercise Git contention with
   explicit synchronization and inspect completed commits; `git ls-files`
   proves only index membership. Verify launcher-visible finalization failures
   and recovery without duplicate observations, cache rows, or commits. Include
   independent concurrent training trials and failure reporting despite a
   successful child/launcher exit with missing or failed finalization.
2. **Launcher arguments and rank binding.** Persist reserved arguments and job
   identity and expected membership; bind ranks to recorded IDs. Add the `flor`
   console entry point and `run` command with tested torchrun/Accelerate DDP
   adapters, unchanged argument forwarding, supported-configuration validation,
   and completion-aware supervision. Define supported DataLoader-worker
   start methods, lifecycle, observation ownership, and replay behavior without
   treating helpers as training ranks. Specify Hugging Face instrumentation,
   resolved configuration capture, and model-save/restore protection. Test
   job/argument selection and restoration of an older job's metadata and exact
   per-rank runs when the checked-out commit contains several JSONLs.
3. **Distributed replay.** Implement job scheduling, topology reconstruction,
   compatible loop narrowing, and bounded failure handling through the existing
   `replay` command, sharing supervision internally with `run`. Pass the numerical,
   selection, recovery, and isolation tests below. Validate PyTorch and Hugging
   Face DDP with helper workers, and pass actual two-GPU CUDA/NCCL tests before
   claiming multi-GPU support.

In stage 1, retain equal timestamps in the same-microsecond fixture but assert
separate run IDs, logs, database groups, and checkpoints. Use run-ID-to-commit
assertions for new runs and dedicated timestamp coverage for legacy runs.
Remove strict `xfail` markers as requirements are implemented. Replace silent
collision expectations with preserved-observation assertions; successful exits
remain valid when data is preserved.

### Identity lifecycle and selection test plan

Add these cases to the concurrency tests. Identity, selection, migration, and
supported helper-worker recording safety belong to stage 1; launcher arguments,
job binding, and helper lifecycle/replay design belong to stage 2.
Programmatic-spawn training integration is deferred.

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
- **Helper-worker safety:** exercise supported DataLoader start methods,
  including fork where supported, with buffered records and process-local
  resources. Verify no parent-record duplication, changed ownership, inherited
  lock deadlock, or lost worker output; cover orderly worker shutdown and failures.

### Launch integration acceptance tests (D2)

Stage 1 proves the minimum supervision/completion contract with GPU-free
subprocess fixtures; stage 2 tests the public CLI and supported real launcher
adapters. Stage 3 exercises the same supervision through distributed replay.

- **CLI and forwarding:** verify the installed `flor` entry point and module
  entry point expose the same commands. Exercise `run -- <command>` with spaces,
  quoted values, and script arguments named like Flor commands/options. Preserve
  argument boundaries, working directory, application environment, and output.
  Keep existing commands and single-process invocation behavior compatible.
- **Identity and membership:** launch two supported jobs simultaneously and
  verify fresh separate job UUIDs, exact expected rank sets, and correct
  per-worker run bindings. Helpers are not counted as training ranks. New
  executions get new job IDs; retrying saved finalization retains its IDs.
- **Success gate:** require all expected durable completion records and Git
  mappings. A zero launcher exit with a missing rank completion, failed Git
  finalization, or a fatal capture error must still return nonzero from Flor
  and leave the job explicitly incomplete. A failed helper finalization must
  reach the owning rank or supervisor and prevent success.
- **Cleanup and retry:** inject worker failure, interruption, and timeout;
  verify bounded termination/reaping, useful diagnostics, and preserved pending
  artifacts. Exercise the documented finalization retry path without duplicate
  observations, cache rows, or commits.
- **Unsupported execution:** reject unintegrated distributed launches with an
  actionable error. Reject unsupported launcher/topology configurations,
  elastic membership, and automatic restarts before training. Forwarding an
  arbitrary command does not establish its distributed compatibility.
- **Distributed replay:** use the existing replay command to launch historical
  workers internally with a fresh attempt ID and the recorded run bindings;
  no manual per-rank launches are required. Apply the existing whole-attempt
  success, partial-result hiding, and cleanup acceptance tests.

### Distributed replay test plan

Stages 2–3 start with single-host CPU jobs and real interprocess communication,
then require actual two-GPU CUDA/NCCL validation for release. Multiple hosts
remain deferred. Acceptance requires recovering the forward job's per-rank
values; successful exits, files, and mappings alone are insufficient.

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
6. **Supported training integrations:** record through `flor run --` using
   PyTorch DDP via `torchrun` and a documented Hugging Face Trainer/Accelerate
   DDP fixture, then use the existing replay command. Include DataLoader
   workers, deterministic transforms and seeding, historical configuration
   restoration, rank-zero checkpoint saves, and supported Hugging Face model
   saves. Test actual two-GPU CUDA/NCCL forward and hindsight numerical
   equivalence, selection, failure cleanup, and checkpoint write protection.

Use explicit synchronization and controlled fault injection. Give every
subprocess group a deadline and `finally` cleanup that terminates and reaps
children. Retain per-rank stdout/stderr and report phase, job, and rank on error
or timeout. Allocate separate rendezvous resources for concurrent recording
jobs and fresh resources for every replay. Concurrent replay in one working
tree is deferred; it requires isolated checkouts or serialized checkout and
execution.

Run tests in order: collective, CPU DDP equivalence, selection/narrowing,
failure/recovery, concurrent-job isolation, then supported training integrations
and GPU/NCCL validation. Keep GPU-free recording tests separate so regressions
remain diagnosable without PyTorch or distributed initialization.
