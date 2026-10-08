# Revision log

## 2026-10-08 — Concurrent recording and distributed replay

Status: design and test plan proposed; implementation pending.

FlorDB can support concurrent training processes while keeping the recording
API fixed. `flor.log`, `flor.arg`, `flor.loop`, `flor.iteration`, and automatic
output capture continue to work without rank arguments. Introduce a UUID per
run and a `flor.run_id(...)` selection helper; checkpoint access accepts run IDs
and retains unambiguous timestamp lookups. The changes belong in run identity,
repository coordination, persistence, and replay orchestration.

The plan has three stages: reliable recording with identity migration and
general run selection, distributed launch configuration, and distributed
replay. Each stage has its own acceptance tests. Recording is validated by
preserved observations and
complete version mappings; distributed replay is validated by recovering the
original job's per-rank numerical results. The detailed replay test plan below
covers that second contract, including selection, narrowing, failure handling,
and isolation. Both the implementation and the additional tests are planned.

### Problem and evidence

[tests/test_ddp_concurrency.py](tests/test_ddp_concurrency.py) exercises two
independent failures using real subprocesses in temporary Git repositories:

- Concurrent ranks finalize in one working tree and contend for Git's index.
  A successful `git add -A` can stage other ranks' run files, but its commit
  subject names only its own timestamp. A surviving JSONL without a commit
  naming it has no entry in the timestamp-to-commit mapping used by
  `Schedule.iter_dims()`, which can then raise `KeyError`.
- Two processes with the same import-time microsecond timestamp share a run
  identity. `orm.to_jsonl()` opens the corresponding path with `"w"`, so the
  later process overwrites the earlier process's observations. Checkpoint
  shelves use the same identity and are also exposed to collisions.

Git errors are printed and swallowed by `versions.git_commit()`, leaving
launchers with a successful exit status despite missing finalization. The
timestamp overwrite is silent. Both failures can be reproduced with ordinary
Python subprocesses. These tests isolate Flor's recording races; actual
PyTorch/DDP integration requires the CPU/Gloo tests planned below, with
GPU/NCCL coverage deferred.

### Give each run a UUID and retain timestamps as metadata

Generate a random UUID (`uuid4`) before constructing the first record or saving
the first checkpoint, and keep it stable throughout the run. Persist the
canonical UUID string as `run_id` in every new record and use it in
`.flor/runs/<run_id>.jsonl`, `.flor/obj_store/<run_id>/`, database grouping,
commit mappings, and replay selection. Copies of a recorded run retain its ID;
new executions in independently recorded clones generate new IDs.

Keep `tstamp` as the unadjusted start time, with its existing datetime format.
Several runs may have the same timestamp. Preserve `projid` as a project label
and capture Git author name/email as optional attribution at run start. Neither
directory names nor Git identities are uniqueness keys: clones can share both,
and the same person can launch multiple jobs. A shared `job_id` identifies one
distributed launch; each rank has its own `run_id`.
Persist this metadata with the run so later Git configuration or directory
renaming does not change its recorded attribution.

UUIDs provide practical uniqueness across processes and clones without a
timestamp reservation registry or a scan of Git history before recording.
Protect every publication against overwriting an existing run nevertheless.
An identity conflict must preserve existing artifacts and report an error;
never silently rename a run after records or checkpoints have been associated
with it. Retries retain the same ID and verify that existing artifacts belong
to that run.

#### Selecting a run without copying UUIDs

Add the proposed read-only helper:

```python
flor.run_id(run=None, *, tstamp=None, job_id=None,
            args=None, metadata=None, **arg_filters)
```

`run` accepts a full run ID, a timestamp string or datetime, or a dataframe
row containing `run_id` (falling back to `tstamp` for legacy rows). `tstamp`
is the explicit timestamp form and cannot be supplied together with `run`.
`job_id` narrows the selection to a distributed execution. Additional keyword
arguments select recorded Flor arguments, for example `lr=0.01` or `seed=42`.
The `args` mapping supports arbitrary argument names, including reserved names
such as `flor::rank` and names that overlap helper parameters, such as an
application argument named `job_id`. `metadata` selects recorded fields such
as `projid`, `filename`, `author_name`, and `author_email`. Keep these namespaces
explicit so a user argument named `rank` or `author_email` retains its meaning.

All criteria are combined with AND. Reject an argument name supplied in both
`args` and `arg_filters` instead of assigning precedence. Require at least one
selector and match exactly one distinct recorded run after filtering. Multiple
metric or replay rows for the same run count as one candidate. Rank has no
special helper parameter; it is selected through its recorded argument like
any other execution parameter.

Argument filters use typed values from the original forward run, with exact
equality and no implicit string-to-number conversion or floating-point
tolerance. Repeated equal argument records count once. A filtered argument
with several distinct values in a candidate run raises `ValueError` identifying
that run; never choose its first or last value silently. Missing argument
values do not match. Replay overrides and hindsight metrics do not change the
original run's selection attributes. For range predicates or evolving metrics,
filter a dataframe first and pass a selected row to the helper.

Return the canonical ID on one match. Raise `LookupError` on no match and
`ValueError` on ambiguity, with candidate IDs, timestamps, jobs, and relevant
argument values to help refine the selection. Invalid selector combinations,
unknown metadata fields, or malformed values raise `ValueError`. Never select
the latest match implicitly. The helper neither creates a run nor generates an
ID; internal recording uses a separate context accessor.

```python
# A row already chosen through ordinary dataframe filtering.
rid = flor.run_id(row)
state = flor.load_checkpoint(rid, "ckpt.pth")

# Select by ordinary recorded hyperparameters; repeated trials may need
# further criteria to identify one run.
rid = flor.run_id(lr=0.01, seed=42)

# Combine recorded attribution with hyperparameters.
rid = flor.run_id(lr=0.01, metadata={"author_email": "alice@example.org"})

# Select a rank using the same argument-filtering mechanism.
rid = flor.run_id(job_id=job_id, args={"flor::rank": 0})

# Preserve timestamp selection when it resolves to one run.
rid = flor.run_id(tstamp=row.tstamp)
```

Expose `run_id` in `flor.dataframe()` and `flor.io()` alongside `tstamp` and
`job_id`. Launcher configuration appears through the existing argument columns
in `flor.dataframe()`; IO rows can be joined to those arguments by `run_id`.
Group and deduplicate runs by `run_id`, including when timestamps coincide.
Users selecting one row per run should use
`drop_duplicates("run_id")`. Keep timestamp filtering and sorting available.
Make `flor.checkpoints(...)` and `flor.load_checkpoint(...)` share the helper's
resolution rules while preserving their existing positional call shape and
the accepted `tstamp=` keyword. Existing timestamp calls work when unique and
report ambiguity when another run has the same timestamp.

#### Storage migration and historical runs

New records carry an explicit schema version and `run_id`. Update ORM readers,
SQLite schema and grouping, `flor unpack`, checkpoint lookup and lineage,
commit parsing, and replay scheduling together. Rebuilding a cache must preserve
IDs read from persisted artifacts. UUID filenames have no temporal ordering;
select runs explicitly and sort display results using timestamp metadata.

Mark records produced by `flor.arg` explicitly in the new schema, while
retaining their existing log-value representation and dataframe visibility.
Use this same marker for automatically captured launcher arguments. Currently
`flor.arg` calls `flor.log`, so ordinary log records alone cannot reliably
distinguish arguments from metrics. Reconstruct the typed argument index from
marked JSONL records; it is a derived cache, not a second source of truth.
Use argument provenance during replay initialization as well as selection.

Read legacy timestamp-named files without rewriting historical Git commits.
For legacy JSONL, derive a deterministic ID in a distinct `legacy:` namespace
from the SHA-256 digest of the original file bytes. Identical copies of an
artifact get the same legacy ID; different artifacts sharing a timestamp
remain distinct. Retain explicit locators for their old JSONL and checkpoint
paths instead of constructing UUID paths for legacy data. Branch changes,
cloning, and cache rebuilding must preserve this mapping.

Resolve old timestamp commit subjects through legacy artifacts and verify
their content when several historical runs share a timestamp. Preserve direct
timestamp access to an unambiguous legacy checkpoint-only shelf; without a
recorded run, the selection helper reports that no run ID can be resolved.
Ambiguous legacy checkpoint ownership or commit mappings require an explicit
association, rather than guessing from a directory name. Replacing an already
lost legacy observation is outside this migration.

Legacy argument filtering uses argument names from the matching auto-commit's
argument body and typed values from its associated JSONL when those records
identify one value. If that evidence is absent or conflicting, report that
argument selection is unavailable for that legacy run and allow explicit
ID/timestamp selection. Never infer that every run-level metric is an argument.

#### Allocation timing and run boundaries

Introduce one lazy internal run-context accessor holding `run_id`, `tstamp`,
and the owning PID, and return that same context to every recording path.
Keep `Clock.get_datetime()` as a timestamp accessor and keep profiling timers
independent of identity. Importing Flor or selecting historical data alone
must not generate a new run ID.

This accessor must run before `_emit_io()` constructs its first `orm.Log`.
Currently output capture appends that record before `_register_run()` calls
`_deferred_init()`, so creating the context only in deferred initialization is
too late. Explicit logging, loop profiling, and checkpoint capture share the same
accessor. The checkpoint hook must create the context before the original save
for a captured file checkpoint, even when no logging call has occurred. Keep
context-initialization diagnostics out of automatic capture to prevent recursion.

Close the active identity once its observations are durably published and its
pending finalization metadata is saved. A Git retry uses that saved identity;
it must not create a new run. Clear the process's active identity at the run
boundary and create a new context on the next recorded event, including the next
interactive run. During replay, return the explicitly selected historical
run ID and timestamp without generating a new forward-run identity.

#### Fork policy

Initially support recording in independently launched interpreters, including
`torchrun` ranks and workers started with Python's spawn method. A child that
inherits Flor through `fork` has recording and Flor's replay hooks disabled
until it starts a fresh interpreter. This keeps auxiliary workers from
publishing copies of their parent's buffered records or using its historical
replay identity. Fork-launched training ranks are outside the initial scope.

Register an after-fork handler to discard inherited recording state and mark
the child disabled. Also check the importing PID at recording, checkpoint-hook,
and finalization entry points before touching inherited locks or buffers.
Disabled children forward output normally, execute the original checkpoint
save/load functions, and retain value-producing API behavior such as returning
`flor.arg` values and iterating `flor.loop`. They create no Flor records,
new run IDs, checkpoint copies, cache writes, or commits. Their cleanup hook
returns immediately. Explicit historical checkpoint reads remain available.
Emit one uncaptured notice on the first attempted recording operation in such
a child so the unsupported launch mode is visible. Fresh interpreters initialize
their own state and generate their own run contexts.

### Coordinate repository changes and preserve every version mapping

Introduce a project-level interprocess lock for Flor's repository mutations,
including shadow-branch creation, ignore-file updates, command-file updates,
and finalization. Run ID generation does not require this lock. Store the
lock at `.flor/locks/project.lock` and use an OS-managed advisory lock released
when its owning process exits; the presence of the file alone does not signal
ownership. Never unlink the lock file during normal operation. Close inherited
lock descriptors in fork children without explicitly unlocking the parent's
lock. The Git critical section must cover the dirty check, staging, and commit
together. Use this Flor lock rather than taking Git's own `index.lock`, which
Git needs for its operations.

A lock alone does not guarantee one version mapping per run. One rank can
stage all completed JSONLs, leaving the next rank with a clean tree. For the
initial implementation, use one `FLOR::Auto-commit::run::<run_id>` subject per
new run and allow an empty commit when that run's observations were already
committed by another rank. Keep the run's arguments in its commit body and
resolve historical timestamp subjects through the compatibility reader.
Build a `run_id -> commit` mapping without parsing IDs as dates. Make retries
idempotent so an already finalized run does not acquire duplicate finalization
commits.

Keep SQLite transactions short, commit and close them before invoking Git,
and preserve the interruption behavior covered by
[tests/test_commit.py](tests/test_commit.py). Serialize schema initialization
and handle bounded write contention as part of the concurrent write path.

A future tracked job manifest could map several runs to one commit explicitly.
That requires updating the mapping reader and replay selection together.
Electing rank zero to commit is insufficient unless it publishes mappings
for all participating runs. Recording only rank zero would discard the
observations these tests require preserving.

### Publish complete records and make finalization failures recoverable

Write JSONL to a temporary file and publish it atomically at the run ID's
path. Keep temporary artifacts outside Git's tracked run set, and ensure that
publication cannot replace another run's file. Atomic replacement alone does
not establish ownership of a destination.

Make Git finalization return a meaningful result or propagate an error.
Persist pending finalization information, including the run's arguments,
separately from the recording buffer so a retry can finish versioning without
duplicating observations or SQLite rows. JSONL remains the observation source
of truth and SQLite remains a rebuildable cache.

Explicit `flor.commit()` can propagate failure to its caller. Automatic
shutdown needs a deliberate launcher-visible failure mechanism: simply
raising from an `atexit` callback does not reliably change the process exit
status. Choose and test that mechanism before claiming that launchers can
detect failed finalization. Durable pending status supports recovery but does
not by itself satisfy the nonzero-exit requirement.

### Capture launcher configuration as Flor arguments

Rank is a standard distributed-training concept: it identifies a worker within
its group. PyTorch documents `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` as
[environment variables supplied by torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html#environment-variables).
The `flor::` argument names below are Flor's proposed storage convention.
Keep the mapping from these environment variables in the torchrun integration;
the generic recorder, storage, and selection helper handle arbitrary arguments
such as `lr`, `seed`, or rank. Rank is optional distributed configuration and
is not a required field for every Flor run.

At run-context creation, the torchrun integration reads `RANK`, `LOCAL_RANK`,
and `WORLD_SIZE` from the launcher environment. Record available values as
integer arguments named `flor::rank`, `flor::local_rank`, and `flor::world_size`,
using the internal
argument-recording path shared with `flor.arg`. Emit each once per run and
include it in persisted arguments and the commit's argument body. No PyTorch
import or user logging call is required. Ordinary single-process runs with no
launcher environment leave these arguments absent. Validate provided values
and report malformed or inconsistent launch configuration.

Reserve these names for Flor. A user argument or metric named `rank` remains
independent; application writes or CLI overrides to the reserved launcher names
are rejected. Initialize the context before emitting these arguments and avoid
calling the public `flor.arg` recursively from context creation. Rank, local
rank, and world size do not need separate fields on every record or a separate
rank schema: their existing argument records carry the configuration.

Assign one shared `job_id` per forward execution attempt and distribute it
to all ranks. Use a launcher-provided identity only when it distinguishes that
attempt; otherwise a coordinator generates and distributes a job UUID. Never
let each rank independently generate its job ID or infer grouping from nearby
timestamps, project names, Git authors, or rank numbers. Retain launcher IDs as
metadata when they are insufficient for uniqueness.

Replay retains that historical job ID and uses a separate replay-attempt ID
for completion and retry tracking.

Expose launcher arguments through the normal dataframe argument columns and
keep `job_id` as an identity field. During replay, the launcher creates the
worker topology; Flor then validates each child's assigned rank and world size
against its recorded arguments before executing training. Local-rank placement
must agree with the supported replay topology. Retrieving a saved argument does
not itself configure distributed execution. Unique checkpoint shelves protect
Flor's copies; they do not resolve a training script in which all ranks
independently overwrite the same checkpoint source path.

### Select historical runs explicitly and replay distributed jobs together

The current orchestrator launches one Python process, and the replay child
initializes from the latest JSONL. Pass the selected historical run ID explicitly
to the child and load its exact arguments, starts, and observations. This is
necessary even before adding distributed orchestration because a commit can
contain several ranks' files.

For real DDP replay, schedule the historical job as a unit and bind each child
to its historical rank. Selecting results for one rank may still require
launching all ranks that participate in its computation. Restoring `RANK` in
one process cannot reproduce communication among workers. Reconstruct the
relevant launcher settings while creating fresh rendezvous resources for the
replay. Preserve required participation in collective operations when narrowing
Flor loops; ranks cannot independently omit work that other ranks require them
to perform.

The orchestrator must bound rendezvous and execution time, report failures
from any rank, and terminate and reap remaining children after failure. Track
completion for the whole replay attempt so partial rows from a failed attempt
are not presented as a successful job result. Define retry behavior that avoids
duplicate results and restore the working branch on both success and failure.

The existing replay API can remain the entry point, with job metadata driving
orchestration internally. Any settings needed to reconstruct the launcher
should be additive.

### Implementation stages and acceptance criteria

1. **Reliable recording and identity migration.** Implement UUID run contexts,
   the schema and legacy-reader changes, argument provenance, the general
   `flor.run_id(...)` helper, atomic JSONL publication, coordinated finalization,
   and explicit failure handling.
   Pass explicit run IDs to single-process replay children as part of the
   migration so UUID filenames cannot break historical selection.
   Preserve recording signatures and unambiguous timestamp lookups. Force
   identical timestamps and verify separate run IDs, logs, database groups,
   and checkpoints. Exercise Git contention deterministically and inspect files
   in completed commits; `git ls-files` alone proves index membership. Force
   finalization failure and verify launcher-visible status and recovery without
   duplicate observations, cache rows, or commits. Apply the identity and
   selection tests below, including the policy for forked children.
2. **Launcher arguments and rank binding.** Persist reserved launcher arguments
   and job identity, and bind each selected rank to its historical run ID.
   Extend general argument-filtering tests to launcher arguments and job IDs.
   Record several jobs with distinct arguments and verify that selecting an
   older job restores its metadata and
   each participating rank's exact run, even when several JSONLs are present
   in the checked-out commit.
3. **Distributed replay.** Implement job-level scheduling, topology
   reconstruction, compatible loop narrowing, and bounded failure handling.
   The detailed test plan below establishes numerical equivalence, correct
   historical selection, safe narrowing, failure/retry behavior, and isolation.

Update existing recording tests as part of stage 1. The same-microsecond
fixture should continue to force equal start times while assertions distinguish
runs by `run_id`. Change timestamp-to-commit assertions to run-ID-to-commit
assertions for new runs and retain dedicated legacy-reader coverage. Remove
strict `xfail` markers as their individual requirements are implemented.
Replace the test documenting silent collisions with assertions about preserved
observations; successful exits remain appropriate when data is preserved.

### Identity lifecycle and selection test plan

Add these tests alongside the existing concurrency assertions. Basic identity,
metadata/hyperparameter selection, and migration cases belong to stage 1;
automatic launcher arguments and job binding follow in stage 2:

- Force equal timestamps in simultaneous processes and verify distinct run IDs,
  with each ID consistent across records, filenames, database rows, checkpoint
  shelves, and commit subjects. Assert timestamps remain equal and unchanged.
  Ensure dataframe grouping preserves both runs.
- Record independently in two clones with the same project directory name,
  Git author, and timestamp. Combine their histories and verify distinct run
  IDs, intact observations, and correct version mappings. Copies of the same
  recorded run must retain one identity after import and cache rebuilding.
- Make captured output the first recorded event, including a script with no
  explicit Flor logging calls. Verify its first record already has the final
  run ID. Separately save a checkpoint before any log or captured output and
  verify that the later records use that checkpoint's run ID.
- Verify import, historical queries, and `flor.run_id(...)` generate no new
  identities or recording artifacts. Run two interactive recording cycles
  with an equal timestamp and verify separate IDs, stable within each cycle.
  Force a Git retry and verify it retains the completed run's ID.
- Interrupt publication and verify there is no partial final JSONL. Retry
  recoverable work under the saved ID. Inject an existing UUID destination and
  verify conflicting logs and checkpoint copies are preserved with an explicit
  failure; successful retries must verify ownership instead of overwriting
  another run. Inject UUID-generation failure and verify recording aborts
  without falling back to a timestamp key.
- Read mixed legacy and UUID runs. Verify stable legacy IDs after branch
  changes, cloning, and cache rebuilding, plus correct old commit and checkpoint
  lookup. Different legacy JSONL artifacts with the same timestamp must get
  different IDs and ambiguous ownership must be reported. Test checkpoint-only
  legacy timestamp access separately from run-ID selection. Run the existing
  single-process forward, replay, and checkpoint workflows against both formats.
- Exercise helper selection by full ID, dataframe row, unique timestamp,
  `lr`/`seed`, attribution metadata, and job plus `args={"flor::rank": 0}`.
  Check repeated trials with identical hyperparameters, missing matches,
  timestamp ambiguity, contradictory filters, duplicate argument filters,
  malformed values, and calls with no selector. Verify typed equality, missing
  arguments, conflicting argument values, and argument names that overlap
  helper parameters. Ensure duplicate records for one run do not cause
  ambiguity and no call implicitly chooses the latest run. Verify checkpoint
  APIs accept IDs and unambiguous legacy selectors, including `tstamp=`.
- Distinguish a `flor.arg("lr", ...)` record from a metric named `lr` and verify
  argument filters consult only forward argument records. Replay overrides
  must not change which original run matches. Rebuild the cache and repeat
  selection to verify argument provenance survives. Cover legacy argument
  recovery and explicit errors when historical provenance is unavailable.
- Run launcher-configured processes without explicit rank logging and verify
  reserved arguments are emitted once with integer values. Check that an
  ordinary user argument named `rank` stays independent, absent launcher values
  are not invented, and reserved-name writes/overrides are rejected. Replay
  must reject assigned rank/world-size mismatches before training begins.
- Fork before and after the parent creates a run context, including a parent
  with buffered records. Verify the child follows the disabled-recording
  policy, preserves ordinary computation and output, and neither publishes
  parent records nor changes parent ownership. Verify a fresh spawned worker
  records normally under a separate identity. Include a fork while the project
  lock is held and verify that child cleanup does not release the parent's lock
  or retain ownership after the parent exits. On platforms without `fork`, skip
  only the fork-specific cases.

### Detailed distributed replay test plan

This planned suite supplies the distributed acceptance tests for stages 2
and 3, building on stage 1's recording tests. It targets single-host CPU jobs
with real interprocess communication. GPU/NCCL and multiple-host coverage are
subsequent work.

The central acceptance criterion is numerical: hindsight replay must recover
the same per-rank values as the original distributed execution. Successful
process exits, complete files, and version mappings support that assertion;
they do not replace it.

1. **Build a deterministic distributed fixture.** Launch two ranks with
   PyTorch's Gloo backend, fixed seeds, synthetic inputs, and a fresh
   rendezvous for each invocation. Begin with an `all_reduce` whose result
   is known exactly, then add a tiny DDP model with fixed initialization and
   a short training loop. Keep PyTorch optional for the rest of the suite;
   the distributed test environment must install it and run these tests.
2. **Compare forward execution with hindsight replay.** During the forward
   job, compute a reference value for each rank and iteration in separate
   test-owned artifacts outside the project repository. Add a hindsight
   `flor.log` to the training script and invoke Flor's actual replay
   orchestrator. Compare recovered values with the forward references,
   using exact equality for the simple collective and an explicit numerical
   tolerance for model training. Verify historical job and rank ownership for
   every result, with no missing or duplicate ranks.
3. **Exercise historical selection and loop narrowing.** Record several jobs
   with different arguments, select an older job, and verify that its exact
   arguments and checkpoint lineage are restored. Cover full replay,
   selected-iteration logging, and requests for one rank's results that still
   require all participants. All participating ranks must execute the
   computation and collectives required to reach selected iterations.
   If a requested selection is incompatible with distributed execution,
   require a clear rejection before launch rather than a hang or incorrect
   result. Verify that unselected iterations emit no hindsight metric rows.
4. **Inject replay failures and verify recovery.** Fail one rank deliberately
   and prevent one rank from reaching rendezvous. The orchestrator must report
   unsuccessful execution and terminate remaining children within a deadline.
   A missing rank must not become an apparently successful partial replay.
   Verify attempt completion status, ensure partial rows are excluded from
   completed results, and retry without accumulating duplicate result rows.
   Historical observations must survive these failures. Git finalization
   failure belongs to stage 1's recording tests because replay does not
   finalize new forward runs.
5. **Verify isolation and repeatability.** Replay the same historical job
   twice. Also record two separate jobs concurrently, then replay each job
   sequentially in the shared working tree. Check that identities, rank
   bindings, rendezvous resources, and results remain separate. Compare
   historical JSONL and checkpoint contents before and after replay to verify
   that replay leaves them unchanged. Check branch restoration on success
   and failure as well as cleanup of child processes and rendezvous artifacts.

Use explicit synchronization and controlled fault injection instead of timing
guesses. Give every subprocess group a deadline and cleanup in `finally`,
including termination and reaping of remaining children. Keep stdout/stderr
for each rank and report the failing phase, job, and rank on timeout or error.
Concurrent recording jobs must use independently allocated rendezvous
resources, and every replay invocation must get fresh resources. Concurrent
replays in a shared working tree are outside the initial scope: supporting
them would require isolated checkouts or serialized checkout and execution.

Run the smallest collective test first, then DDP numerical equivalence,
selection/narrowing, failure/recovery, and concurrent-job isolation. Keep the
existing GPU-free recording tests as a separate foundation so that recording
regressions remain diagnosable without PyTorch or distributed initialization.
