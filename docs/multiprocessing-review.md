# Multiprocessing design review

Reviewed: 2026-10-27. Status: proposed; implementation pending.

The [revision plan](../REVISION-LOG.md#2026-10-08--concurrent-recording-and-distributed-replay)
defines the requirements and acceptance tests. This document tracks the review,
open decisions, and design work needed to make implementation concrete. Keep
the technical discussion in these repository documents; use chat for brief
updates and decisions. Update the plan when a decision changes its requirements
or stages, and record the resolution here.

These repository documents are the ground truth for subsequent review rounds.
Carry resolved decisions forward unless the project owner revisits them.

## How to finish this review

Your goal is to approve a useful first release and its proof of correctness.
Read the delivery table, resolve the owner decisions, and check the acceptance
scenarios below. The engineering details and evidence are reference material;
you do not need to design each mechanism yourself.

### 1. Check the intended user experience

Read **Proposed delivery plan** and evaluate this workflow:

> I launch a supported multiprocessing training job. Flor preserves each
> worker's observations and checkpoint copies. Later I select that historical
> job, add a hindsight log, and replay the cooperating workers. I get values
> matching the original computation. If recording or replay fails, I can tell
> that it failed and retry without corrupting history or duplicating results.

The first release targets single-host PyTorch and Hugging Face DDP training,
including DataLoader-worker compatibility and concurrent independent trials.
CPU/Gloo is the development milestone; actual GPU/NCCL validation is required
before claiming multi-GPU support. Sharded/model-parallel training and online
serving are follow-on work.

This is the resolved D1 scope. The [workload analysis](model-workloads.md)
records the rationale. Write down any missing essential workflow; the remaining
owner decisions concern launch integration, fidelity, isolation, and versioning.

### 2. Resolve the remaining design decisions

Read **Decisions for the project owner** in this order:

1. **D1 — Initial workloads: resolved.** PyTorch/Hugging Face DDP, helper-worker compatibility, and GPU validation as described above.
2. **D2 — How much may the launch command change?** This constrains coordination and failure reporting.
3. **D3 — Under what conditions must replay match?** This defines the correctness promise.
4. **D4 — Can recording and replay share a checkout at the same time?** This defines isolation behavior.
5. **D5 — Are per-worker Git commits acceptable?** This confirms the versioning approach.

For each, record a choice or explicitly delegate it with constraints. You can
accept a recommendation, change it, or ask for a concrete example before
deciding. Fill in the **Decision record** at the end; recommendations alone
do not resolve a decision. A deferred feature needs a stated interim behavior,
such as an explicit unsupported-operation error.

### 3. Agree on what would demonstrate success

Check these scenarios against your intended workloads. The revision plan has
the detailed tests; add or change scenarios there if this list is insufficient.

| Scenario | Required outcome |
| --- | --- |
| Workers start simultaneously, including with equal timestamps. | Every worker's observations, run identity, checkpoint copies, and version mapping survive. |
| Several jobs exist and I select an older one. | Replay restores that job's exact workers, recorded arguments, and checkpoint lineage. |
| I add a hindsight log to a supported collective or DDP job. | Each selected worker/iteration matches the forward reference within the agreed numerical tolerance; required peers still participate. |
| A worker, rendezvous, or Git finalization fails. | Failure is visible, children are cleaned up, partial replay results stay hidden, and retry creates no duplicates. |
| I use existing single-process history or checkpoints. | Supported legacy lookups still work; ambiguity is reported rather than silently resolved. |
| I train and replay with DataLoader workers enabled. | Supported helpers preserve parent/child observation ownership and deterministic inputs without being mistaken for DDP ranks. |

### 4. Close the review

You can declare **scope and acceptance review complete** when:

- [x] The first supported workloads and explicit deferrals are recorded (D1).
- [ ] D1–D5 have choices or explicit delegation with constraints.
- [ ] The acceptance scenarios would convince you the capability works.
- [ ] The revision plan reflects those decisions without contradictory promises.
- [ ] Each remaining engineering item is assigned to a delivery stage, or deferred with its dependent feature.

At that point, record the review outcome and date below. This approves the
target and the work to reach it; it does not claim implementation is complete.
The implementation is successful only when the agreed acceptance tests pass.

## Current assessment

The requirements and tests are thorough. Concrete protocols are still needed
for job coordination, shutdown failures, helper-worker handling, and interrupted
writes. Those are engineering work items below. D1 resolves the workload scope;
the remaining design must make DDP and framework-worker support concrete.

## Proposed delivery plan

These stages reflect the resolved D1 scope.

| Stage | Deliverables | Acceptance |
| --- | --- | --- |
| 1. Reliable recording and identity migration | UUID contexts; schema and legacy readers; argument provenance; historical selection; atomic publication; serialized Git finalization; recovery; helper-worker recording safety and lineage; explicit run selection for single-process replay. | Equal timestamps lose no observations or checkpoints. Every run has a valid commit mapping. Failures reach the launcher. Retries duplicate neither observations, cache rows, nor commits. Independent trials and supported helper workers preserve parent ownership. |
| 2. Launcher arguments and worker binding | Reserved rank arguments; a shared job ID; exact historical rank bindings; helper-worker lifecycle and replay ownership; Hugging Face instrumentation/configuration and model-save contract. | An older job restores its actual arguments, workers, and checkpoint lineage even when its commit contains several JSONLs. Supported helper workers are distinguished from training ranks. |
| 3. Distributed replay | Job scheduling; topology and rendezvous reconstruction; compatible loop narrowing; bounded failure handling; PyTorch and Hugging Face DDP acceptance. | CPU/Gloo development tests and actual two-GPU CUDA/NCCL tests reproduce forward values, including helper-worker configurations. Failed attempts expose no partial results. Processes are reaped and the working branch is restored. |

The proposed contract preserves the recording API, uses JSONL as observation
truth and SQLite as a cache, and rejects ambiguous historical selection.
Every rank retains its own observations, checkpoint shelf, and commit mapping.
Initial distributed replay targets one host, starting with CPU/Gloo and requiring
GPU/NCCL validation for release. Multiple hosts, concurrent replay in one working
tree, sharded/model-parallel training, and online serving are deferred.
Finite model evaluation/offline inference is the first serving-related direction;
its integration is follow-on work.

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

D1 is resolved. D2–D5 remain open; their recommendations are proposals.

| ID | Decision | Recommendation and consequence |
| --- | --- | --- |
| D1 | Which model-training and serving workflows must ship first? | **Resolved 2026-10-27:** single-host PyTorch/Hugging Face DDP, DataLoader-worker compatibility, independent concurrent trials, and actual GPU/NCCL validation. Sharded/model-parallel training and online serving are follow-on work. See the [workload analysis](model-workloads.md). |
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
3. **Helper-worker identity and replay (D1).** Define supported DataLoader start
   methods, observation ownership, lifecycle, seeding, and replay behavior.
   Preserve parent records and application stream writes without silently
   dropping child observations. Cover worker replacement, partial output,
   process-local resources, and completion within the supported DataLoader
   lifecycle.
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
9. **Hugging Face integration (D1).** Define the instrumentation surface,
   recoverable hindsight values, resolved configuration, and supported model-save
   formats in stage 2. Prove recording, restore, and replay write protection in
   stage 3, including GPU/NCCL coverage.

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
failure recovery, and concurrent-job isolation. Include DataLoader compatibility,
Hugging Face integration, and GPU/NCCL release validation. Keep recording tests
independent of PyTorch.

## Decision record

Review outcome: **In progress**. D1 is resolved; D2–D5 remain open.

Replace each open entry with the chosen behavior or delegated constraints,
date it, and link to the corresponding revision-plan change.

| ID | Resolution or delegated constraints | Date / plan reference |
| --- | --- | --- |
| D1 — Initial worker support | Resolved: single-host PyTorch/Hugging Face DDP, DataLoader-worker compatibility, independent trials, CPU/Gloo development and GPU/NCCL release validation. Sharded/model-parallel training and online serving deferred. | 2026-10-27 / [revision-plan scope](../REVISION-LOG.md#scope-decision--2026-10-27), [workload analysis](model-workloads.md#accepted-scope-d1) |
| D2 — Launch integration | Open | — |
| D3 — Numerical fidelity | Open | — |
| D4 — Checkout isolation | Open | — |
| D5 — Commit granularity | Open | — |
