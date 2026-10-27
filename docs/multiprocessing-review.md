# Multiprocessing design review

Reviewed: 2026-10-27. Status: proposed; implementation pending.

The [revision plan](../REVISION-LOG.md#2026-10-08--concurrent-recording-and-distributed-replay)
defines the requirements and acceptance tests. This document tracks the review,
open decisions, and design work needed to make implementation concrete. Keep
the technical discussion in these repository documents; use chat for brief
updates and decisions. Update the plan when a decision changes its requirements
or stages, and record the resolution here.

## Assessment

The plan is well developed as a requirements and acceptance-test specification,
but is not yet a complete implementation design. It separates three necessary
capabilities: preserving concurrent recordings, identifying related workers,
and replaying their computation together.

Identity, compatibility, and data preservation are the strongest parts.
Launcher coordination, fork replay, shutdown failure reporting, and recovery
across partially completed writes still need concrete protocols. Stage 1 is
already a substantial migration, especially with full forward fork support.

## Proposed delivery plan

These are the existing stages, not a revised scope.

| Stage | Deliverables | Acceptance |
| --- | --- | --- |
| 1. Reliable recording and identity migration | UUID contexts; schema and legacy readers; argument provenance; historical selection; atomic publication; serialized Git finalization; recovery; forward fork recording and lineage; explicit run selection for single-process replay. | Equal timestamps lose no observations or checkpoints. Every run has a valid commit mapping. Failures reach the launcher. Retries duplicate neither observations, cache rows, nor commits. Existing workflows remain compatible. |
| 2. Launcher arguments and worker binding | Reserved rank arguments; a shared job ID; exact historical rank and fork-child bindings. | An older job restores its actual arguments, workers, and checkpoint lineage even when its commit contains several run files. |
| 3. Distributed replay | Job scheduling; topology and rendezvous reconstruction; compatible loop narrowing; bounded failure handling and attempt completion. | CPU/Gloo collective, DDP, and fork tests reproduce forward values. Failed attempts expose no partial results. Processes are reaped and the working branch is restored. |

The proposed contract preserves the recording API, uses JSONL as observation
truth and SQLite as a cache, and rejects ambiguous historical selection.
Every rank retains its own observations, checkpoint shelf, and commit mapping.
Initial distributed replay targets one host with CPU/Gloo. GPU/NCCL, multiple
hosts, and concurrent replay in one working tree are deferred.

## Strengths to retain

- UUID migration covers database grouping, checkpoint lineage, commit parsing,
  replay initialization, and interactive run boundaries as well as filenames.
- Git locking is paired with a mapping for every run. A worker whose JSONL was
  committed by another worker still needs its own finalization mapping; the
  proposed empty commit handles that case.
- Marked argument records distinguish `flor.arg` from metrics and preserve the
  original forward selection attributes despite replay overrides.
- Replay receives an explicit historical run ID. Checking out a commit does
  not identify a run when that commit contains several JSONLs.
- Acceptance requires numerical agreement with forward execution, not just
  successful processes and intact artifacts.

## Decisions for the project owner

All decisions below are open. Recommendations are proposals, not approvals.

| ID | Decision | Recommendation and consequence |
| --- | --- | --- |
| D1 | Must full fork support ship with the first distributed release, including nested forks and forks inside active Flor loops? | Deliver independent-interpreter recording and CPU/Gloo replay first, then full fork support, unless fork workloads are essential immediately. Accepting this recommendation changes the current stage boundaries and requires an explicit interim fork policy. |
| D2 | May multiprocessing support require a Flor launch wrapper or explicit launcher integration? | Allow one to own job-ID distribution, shutdown status, and supervision. Logging calls remain unchanged. If ordinary `torchrun` must work with only `import flordb`, state that constraint before selecting the coordination mechanism. |
| D3 | What numerical fidelity does the first release guarantee? | Guarantee equivalence for deterministic workloads with supported topology and matching inputs and environment. Define user responsibilities for randomness and external state; broader automatic capture is additional scope. |
| D4 | May replay overlap an active recording job in the same checkout? | Initially reject the overlap, or use isolated worktrees. Choose one explicitly; short Git locks do not protect running code from later checkout changes. |
| D5 | Is one finalization commit per run/rank acceptable? | Retain the proposed per-run commits initially. A tracked job manifest could reduce commit volume later but would expand mapping and migration work now. |

## Engineering design still required

These details need written implementation choices. They need not each become
a separate owner approval unless they change the supported behavior.

1. **Job coordination and completeness (D2).** Specify who creates the shared
   job UUID, how workers receive it, and how concurrent launches and restarts
   remain separate. Persist expected membership and distinguish incomplete
   recording jobs from replayable jobs. Define ownership and cleanup of the
   coordination resources.
2. **Shutdown and failure propagation (D2).** Choose how ordinary interpreter
   shutdown and orderly multiprocessing worker shutdown finalize observations
   and produce launcher-visible failure. Explicit `flor.commit()` exceptions,
   diagnostics, and pending status alone do not establish this behavior.
   Include fatal recording errors originating under automatic capture, where
   exceptions currently can be swallowed.
3. **Fork identity and replay binding (D1).** Define the stable child selector
   for repeated forks at one location, nested forks, and worker replacement.
   Specify how creation order affects matching. Preserve the inherited state
   required for computation while separating publication ownership. Cover
   loop scopes, arguments, checkpoints, partial output, locks, and completion.
4. **Publication and recovery state machine.** Specify ordering and durable
   markers for pending metadata, JSONL publication, cache insertion, Git
   finalization, and completion. For every interruption point, explain how
   retry verifies artifact ownership and recognizes work already completed.
   Include the case where Git committed successfully before local completion
   status was saved. Define how pending work is discovered and retried.
5. **Legacy identity.** Choose a reproducible derivation or persisted mapping
   that survives cloning, branch changes, and cache rebuilding without
   rewriting history. Distinguish different artifacts with equal timestamps,
   preserve copies, and define errors for ambiguous checkpoint ownership.
6. **Replay attempt publication.** Specify how per-worker results remain
   hidden until the whole attempt succeeds, how failures discard partial
   results, and how retry and repeated successful replay avoid duplicates.
7. **Execution fidelity and isolation (D3, D4).** State which initialization,
   random state, input ordering, dependencies, and external data are captured
   versus supplied by users. Define the source-snapshot assumption for
   concurrent recording, the recording/replay exclusion rule, and recovery of
   the checkout after failure. Distributed narrowing must preserve collective
   participation; suppressing selected outputs must not skip required work.
8. **Storage and platform details.** Write the concrete schema and argument
   encoding, legacy-reader behavior, lock implementation and supported
   platforms, and SQLite contention deadlines. Check checkpoint lineage code
   for timestamp-ordering assumptions when introducing unordered UUIDs.

## Evidence and validation

The review checked the plan against recording, Git, replay, storage, and
checkpoint paths. Current behavior supports the diagnosis:

- [JSONL publication](../flordb/orm.py) uses timestamp filenames and opens them
  with `"w"`.
- [Git finalization](../flordb/versions.py) swallows commit errors and produces
  no commit for a clean tree.
- [Replay initialization](../flordb/cli.py) selects the lexically latest JSONL
  and treats top-level metric records as historical arguments.
- [Scheduling](../flordb/repl.py) maps timestamps to commits and launches one
  replay process at a time.
- [Commit handling](../flordb/api.py) clears buffered observations and argument
  state before Git finalization; it does not yet persist a retry protocol.

Focused baseline run on 2026-10-27:

```sh
python -m pytest tests/test_ddp_concurrency.py tests/test_commit.py -q -ra
```

Result: **7 passed, 4 expected failures**. The expected failures cover missing
per-run commit mappings, successful exit despite failed finalization, and
timestamp collisions losing run files and observations. This is a recording
baseline; it does not establish distributed numerical equivalence.

The existing concurrency fixtures still need the stronger synchronization,
completed-commit inspection, and timeout cleanup required by the plan.
Implement the deterministic collective before DDP, then exercise selection,
failure recovery, and concurrent-job isolation. Add fork equivalence when
fork replay enters the supported scope. Keep recording tests independent of
PyTorch.

## Decision record

No decisions resolved yet. Record each resolution by ID, with its date,
chosen behavior, and corresponding revision-plan change.
