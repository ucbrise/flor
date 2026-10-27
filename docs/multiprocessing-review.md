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

The [workload analysis](model-workloads.md) records the rationale. 
Write down any missing essential workflow; 
the remaining owner decisions concern fidelity, isolation, and versioning.

### 2. Resolve the remaining design decisions

Read **Decisions for the project owner** in this order:

1. **D1 — Initial workloads: resolved.** PyTorch/Hugging Face DDP, helper-worker compatibility, and GPU validation as described above.
2. **D2 — Launch integration: resolved.** Require `flor run -- <launcher and arguments>` for supported distributed recording; extend `flor replay` for distributed hindsight execution.
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
- [x] The launch integration and CLI boundary are recorded (D2).
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
D2 establishes the supervising `flor run` integration. Internal protocols must
still make DDP and framework-worker support concrete.

## Proposed delivery plan

These stages reflect the resolved D1 scope and D2 launch integration.

| Stage | Deliverables | Acceptance |
| --- | --- | --- |
| 1. Reliable recording and identity migration | UUID contexts; schema and legacy readers; argument provenance; historical selection; atomic publication; serialized Git finalization; recovery; helper-worker recording safety and lineage; minimum supervision/completion mechanism for failure reporting; explicit run selection for single-process replay. | Equal timestamps lose no observations or checkpoints. Every run has a valid commit mapping. Recording/finalization failures and missing completion cause a nonzero supervised result. Retries duplicate neither observations, cache rows, nor commits. Independent trials and supported helper workers preserve parent ownership. |
| 2. Launcher arguments and worker binding | `flor` console entry point and `run` command; tested torchrun/Accelerate DDP adapters; reserved rank arguments; shared job ID and expected membership; exact historical rank bindings; helper-worker lifecycle and replay ownership; Hugging Face instrumentation/configuration and model-save contract. | Launcher/script arguments pass unchanged. Separate launches receive separate identities. Unsupported launches/restarts fail explicitly. An older job restores its actual arguments, workers, and checkpoint lineage even when its commit contains several JSONLs. Supported helper workers are distinguished from training ranks. |
| 3. Distributed replay | Existing `replay` command extended with shared supervision; job scheduling; topology and rendezvous reconstruction; compatible loop narrowing; bounded failure handling; PyTorch and Hugging Face DDP acceptance. | CPU/Gloo development tests and actual two-GPU CUDA/NCCL tests reproduce forward values, including helper-worker configurations. Failed attempts expose no partial results. Processes are reaped and the working branch is restored. |

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

D1 and D2 are resolved. D3–D5 remain open; their recommendations are proposals.

| ID | Decision | Recommendation and consequence |
| --- | --- | --- |
| D1 | Which model-training and serving workflows must ship first? | **Resolved 2026-10-27:** single-host PyTorch/Hugging Face DDP, DataLoader-worker compatibility, independent concurrent trials, and actual GPU/NCCL validation. Sharded/model-parallel training and online serving are follow-on work. See the [workload analysis](model-workloads.md). |
| D2 | May multiprocessing support require a Flor launch wrapper or explicit launcher integration? | **Resolved 2026-10-27:** require `flor run -- <launcher and arguments>` for supported distributed recording; extend the existing `replay` command for distributed hindsight execution. Preserve launcher arguments, recording APIs, and existing single-process commands. Flor owns job identity, completion checks, and supervision. See the [accepted contract](#d2--launch-integration). |
| D3 | What numerical fidelity does the first release guarantee? | Guarantee equivalence for deterministic workloads with supported topology and matching inputs and environment. Define user responsibilities for randomness and external state; broader automatic capture is additional scope. |
| D4 | May replay overlap an active recording job in the same checkout? | Initially reject the overlap, or use isolated worktrees. Choose one explicitly; short Git locks do not protect running code from later checkout changes. |
| D5 | Is one finalization commit per run/rank acceptable? | Retain the proposed per-run commits initially. A tracked job manifest could reduce commit volume later but would expand mapping and migration work now. |

### D2 — launch integration

Accepted 2026-10-27. Require `flor run -- <launcher and arguments>` for supported
distributed recording in the first release, with unchanged logging calls and
launcher arguments. Distributed hindsight execution uses the existing `replay`
command, extended to supervise the cooperating ranks internally. Existing
single-process commands retain their current form. The
[revision-plan decision](../REVISION-LOG.md#launch-integration-decision-d2--2026-10-27)
records the requirements, stages, and acceptance checks.

The objective is a reliable job-level success/failure result backed by shared
identity and verified completion of every expected worker's recording. A
supervising Flor process is the accepted mechanism for meeting that objective.

Accepted CLI shape, **implementation pending**:

```sh
# Existing launch.
torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py --batch-size 32

# Required Flor launch for supported distributed recording.
flor run -- torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py --batch-size 32

# The same boundary for the supported Hugging Face DDP integration.
flor run -- accelerate launch --multi_gpu --num_processes=2 train.py
```

The existing launcher continues to create training ranks and configure their
distributed environment. [PyTorch documents this role for torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html);
[Accelerate documents both launch paths](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch).
Flor adds coordination around tested launcher configurations. Supervision flags
and the internal status protocol remain engineering choices.

#### Accepted CLI surface

Add one new public command, `run`, for supervised forward execution.
Keep distributed hindsight execution in the existing `replay` command, sharing
the supervision machinery internally. A small public interface still requires
substantial lifecycle and failure-handling implementation.

The repository currently exposes these commands through `python -m flordb`;
[`setup.py`](../setup.py) does not install a `flor` console entry point.
Add a `flor` console entry point exposing the same CLI:

| Command | Role | Status |
| --- | --- | --- |
| `flor run [Flor options] -- <launcher and arguments>` | Start and supervise a supported recording job. | Accepted new command; implementation pending. |
| `flor replay ...` | Select history and orchestrate hindsight execution, extended to distributed jobs. | Existing command; distributed extension accepted, implementation pending. |
| `flor capture ...` | Inspect captured output, preview extraction, or extract metrics. | Existing command via `python -m flordb`. |
| `flor query ...`, `flor dataframe ...` | Query recorded observations. | Existing commands via `python -m flordb`. |
| `flor unpack` | Rebuild the local cache from recorded history. | Existing command via `python -m flordb`. |
| `flor stat` | Print platform/Python environment information. | Existing command via `python -m flordb`. |

Keep `run` options limited to Flor supervision controls, such as an optional
execution deadline; exact flags remain to be designed. Pass the command after
`--` unchanged, while validating supported launcher configurations. Keep GPU
count, distributed strategy, rendezvous, and training arguments in the existing
launcher/script interface. Job/run IDs and worker completion reporting are
managed internally. Parse the delegated command separately so its arguments
cannot be mistaken for Flor wrapper options.

This decision does not require separate public `init`, `start`, `stop`, `worker`,
or `commit` commands. Pending-finalization recovery still needs a documented,
testable retry path; decide its interface during recovery design rather than
assuming `unpack` repairs Git finalization. Existing single-process commands
retain their current form; optional supervised single-process execution can
share `run` if included in the implementation scope.

Implementation constraints:

- Create and distribute one fresh job UUID before workers start; record expected
  training-rank membership. Helpers retain their defined ownership and are not
  counted as extra ranks. Concurrent launches receive separate identities.
- Return success only after the launcher succeeds and all expected ranks have
  durably finalized their recording, including required helper output and Git
  mappings. Missing completion or a fatal capture/finalization error must fail
  the Flor command even if the underlying launcher returns zero. An outer
  wrapper alone cannot supply this guarantee; worker status reporting and
  completion validation are required.
- On failure, interruption, or timeout, terminate and reap remaining children
  within a deadline, identify the failing phase/job/rank where available, and
  retain recoverable pending artifacts. Ordinary shutdown and helper exit
  paths still need the engineering protocols below; Python documents that
  [`os._exit()` and some other exits bypass atexit](https://docs.python.org/3/library/atexit.html).
- Preserve launcher and script arguments, working directory, application
  environment, and output streams, adding Flor coordination metadata. Validate
  supported fixed-size, single-host configurations before training. Defer
  elastic membership and automatic worker restarts, rejecting those
  configurations explicitly; a new training execution gets a fresh job UUID.
  Retrying finalization of saved work retains its existing identities.
- Preserve existing single-process commands and concurrent independent-trial
  recording safety. Require the supported integration for the new distributed
  contract; an unintegrated distributed launch must report an actionable
  unsupported-launch error rather than silently claim a replayable job.
- Use the same integration boundary internally for distributed replay, retaining
  historical job/run bindings with a fresh replay-attempt ID. Users do not
  manually relaunch historical ranks.

The revision plan places the minimum supervision/completion mechanism needed
to prove stage-1 failure reporting in stage 1, the public CLI, launcher adapters,
and job membership in stage 2, and replay orchestration in stage 3.
Acceptance covers unchanged argument forwarding, simultaneous launches,
missing rank completion despite launcher exit zero, failed helper finalization,
interruption cleanup, and explicit unsupported-launch/restart errors.

## Engineering design still required

These details need written implementation choices. They need not each become
a separate owner approval unless they change the supported behavior.

1. **Job coordination and completeness (D2).** Specify how the supervisor
   distributes the shared job UUID to workers and keeps concurrent launches
   separate. Reject unsupported automatic restarts. Persist expected membership
   and distinguish incomplete recording jobs from replayable jobs. Define
   ownership and cleanup of the coordination resources.
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

Review outcome: **In progress**. D1 and D2 are resolved; D3–D5 remain open.

Replace each open entry with the chosen behavior or delegated constraints,
date it, and link to the corresponding revision-plan change.

| ID | Resolution or delegated constraints | Date / plan reference |
| --- | --- | --- |
| D1 — Initial worker support | Resolved: single-host PyTorch/Hugging Face DDP, DataLoader-worker compatibility, independent trials, CPU/Gloo development and GPU/NCCL release validation. Sharded/model-parallel training and online serving deferred. | 2026-10-27 / [revision-plan scope](../REVISION-LOG.md#scope-decision--2026-10-27), [workload analysis](model-workloads.md#accepted-scope-d1) |
| D2 — Launch integration | Resolved: require `flor run` around supported distributed recording launchers; add the console entry point, extend `replay`, preserve launcher/script arguments and existing single-process commands. Verify all expected worker finalization before success; reject unsupported launches, elastic membership, and automatic restarts. Internal protocols and the recovery retry interface remain engineering work. | 2026-10-27 / [revision-plan launch integration](../REVISION-LOG.md#launch-integration-decision-d2--2026-10-27), [accepted CLI surface](#accepted-cli-surface) |
| D3 — Numerical fidelity | Open | — |
| D4 — Checkout isolation | Open | — |
| D5 — Commit granularity | Open | — |
