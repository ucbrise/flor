# Model training and serving: multiprocessing scope

Researched: 2026-10-27. Supports decision D1 in the
[multiprocessing review](multiprocessing-review.md#decisions-for-the-project-owner).
Target users train PyTorch or Hugging Face models, possibly on several GPUs,
and evaluate or serve those models. D1 scope is accepted below; detailed
integration mechanisms remain to be designed.

## What the evidence establishes

Official documentation establishes the supported and recommended workflows;
it does not establish their percentage of real deployments. We have no adoption
survey or Flor-user workload inventory from which to quantify fork usage.
Priorities below are engineering judgments based on those documented workflows.

For training, distinguish **parallelism strategy**, **launcher**, and **helper
workers**. DDP, FSDP, and DeepSpeed are strategies; `torchrun` and
`accelerate launch` start workers. Hugging Face Trainer delegates distributed
execution to Accelerate rather than defining a separate process architecture.
See [Transformers' Accelerate integration](https://huggingface.co/docs/transformers/accelerate).

## Natural multi-GPU training alternatives

| Workload | Natural alternative | Process implications and relevance |
| --- | --- | --- |
| Training state fits on one GPU; train one model faster | DDP: replicate the model, partition batches, synchronize gradients. | One training process per GPU is the documented PyTorch pattern. Prioritize `torchrun`; programmatic `torch.multiprocessing.spawn` is another launch path and needs its own lifecycle tests. [PyTorch DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html), [Transformers DDP](https://huggingface.co/docs/transformers/en/ddp). |
| Hugging Face training using Trainer or a custom Accelerate loop | Launch the selected strategy with `accelerate launch` or an appropriate `torchrun` invocation. | DDP is a natural first integration. FSDP or DeepSpeed configuration changes the requirements; supporting the launcher does not establish support for every backend. [Accelerate launching](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch), [Trainer integration](https://huggingface.co/docs/transformers/accelerate). |
| Training state exceeds one GPU's memory | FSDP or DeepSpeed ZeRO. | Shard state across cooperating GPU ranks. FSDP shards parameters, gradients, and optimizer state; ZeRO does so progressively by stage. These are important large-model workflows with additional configuration and checkpoint requirements. [FSDP2](https://huggingface.co/docs/transformers/en/fsdp), [ZeRO](https://huggingface.co/docs/transformers/en/deepspeed). |
| Large-model training needing additional partitioning | Tensor parallelism, pipeline parallelism, or combinations with data parallelism. | Split tensor operations or model stages across devices. Flor must preserve the process groups and dependencies, not merely rank/world size. Specialized follow-on integration. [Parallelism methods](https://huggingface.co/docs/transformers/en/perf_train_gpu_many). |
| Existing single-process multi-GPU code | `torch.nn.DataParallel`. | Uses model replicas and threads in one process. PyTorch recommends DDP instead, even on one host. Avoid making this the primary new target; logs inside threaded forwards require separate scrutiny. [DataParallel](https://docs.pytorch.org/docs/2.14/generated/torch.nn.DataParallel.html). |
| Independent experiments, one per GPU | Separate single-GPU training commands, each assigned a device. | No gradient collectives or shared distributed job required. Concurrent recording safety must cover this accepted workload even without distributed replay. This is a scope choice, not a measured usage claim. |

LoRA and other parameter-efficient methods change what is trained; they do not
by themselves choose the worker topology. Classify those jobs by their actual
launcher, strategy, and save format rather than granting blanket support.

## Where fork matters in these workflows

**GPU ranks:** ordinary CUDA multiprocessing should not depend on inheriting an
already initialized CUDA runtime through `fork`. PyTorch documents `spawn` or
`forkserver` for CUDA subprocesses. Thus full replay of arbitrary inherited
application state after fork is not a prerequisite for the documented DDP
training path. This is a scope inference, not a frequency estimate.
[PyTorch multiprocessing guidance](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html#cuda-in-multiprocessing).

**DataLoader workers:** `num_workers > 0` creates data-loading processes under
each training rank, even if ranks were independently launched. These workers
fetch data and run transforms/collation; they are not additional DDP ranks.
Their start method depends on configuration and platform. Worker RNG and input
ordering matter to replay. A multi-GPU claim tested only with `num_workers=0`
would leave an important documented configuration untested.
[PyTorch data loading](https://docs.pytorch.org/docs/2.14/data.html#single-and-multi-process-data-loading).

**Serving engines and notebooks:** process ownership and launch method vary by
engine and entrypoint. vLLM documents tradeoffs among fork, spawn, and forkserver;
some sections describe older engine behavior, so do not infer every current
entrypoint's defaults from them. Pin versions and test the chosen entrypoint.
[vLLM multiprocessing design](https://docs.vllm.ai/en/latest/design/multiprocessing/).

Accepted scope separates **compatibility with framework-created helper
workers** from **general fork-child hindsight replay**. The former belongs in
release validation; the latter is deferred. The helper-worker
contract must explicitly define helper observation ownership, finalization,
and replay behavior. Silently dropping child records or disabling hooks would
contradict the current revision plan and is not an accepted shortcut.

## What Flor supports today, and what the plan would add

Here, support means tested recording, applicable checkpoint preservation, and
faithful hindsight replay. Importing Flor successfully establishes less.

| Workload | Current repository | Existing revision plan |
| --- | --- | --- |
| Single-process PyTorch custom training loop | Existing logging, loop replay, and file-path `torch.save`/`torch.load` checkpoint workflows have tests. CUDA behavior is not established by those CPU examples. | Preserve compatibility while migrating identity and lookup. |
| Concurrent independent runs in one repository | Unsafe: timestamp collisions and missing Git mappings are reproduced by concurrency tests. | Stage 1 directly addresses recording; test independent trials explicitly. |
| Multi-process DDP via `torchrun` | No reliable distributed recording/replay contract. Current replay starts one process. | Direct target: CPU/Gloo development tests, followed by required GPU/NCCL release validation. |
| DDP via programmatic spawn | No verified worker lifecycle or distributed replay support. | Separate contexts are intended, but parent entrypoint, worker targeting, and shutdown need end-to-end coverage. |
| Hugging Face Trainer / Accelerate DDP | No dedicated integration or tests found. General text capture and explicit Flor calls provide partial building blocks. | DDP infrastructure helps, but Trainer metrics, resolved configuration, internal loop access, and model saving still need explicit integration and acceptance tests. |
| FSDP, DeepSpeed, tensor/pipeline parallel training | No verified integration. Capturing individual save calls does not establish a complete distributed checkpoint. | Generic identity helps; strategy-specific configuration, process groups, checkpoint bundles, and replay are not yet specified. |
| DataParallel | Outer-loop logging may use the existing single-process path; threaded forward logging and GPU equivalence are unverified. | No dedicated acceptance case. |
| Distributed inference or online serving | No serving-engine integration, request replay, or bounded recording-window contract found. | Not delivered by the training replay plan. Requires separate workload and acceptance definitions. |

Local evidence: [recording/checkpoint hooks](../flordb/api.py),
[replay launcher](../flordb/repl.py), [AST loop/log discovery](../flordb/hlast/visitors.py),
[checkpoint tests](../tests/test_torch_checkpointing.py),
[recording race tests](../tests/test_ddp_concurrency.py), and
[current replay contract](replay.md). This table is code inspection plus the
previous focused baseline, not a new runtime compatibility certification.

### Hugging Face support needs more than process safety

Trainer owns its training loop. Flor's current script analysis recognizes
explicit `flor.loop` and `flor.log` calls; it does not automatically expose
Trainer's internal steps or intermediate tensors. Define whether the first
integration supports callbacks, custom Trainer methods, or a custom training
loop, and exactly which hindsight values users can recover.

Flor currently hooks `torch.save`/`torch.load` and copies individual path-based
saves. Transformers can save safetensors and sharded model artifacts; those
writes do not all pass through the hooks. Distributed Checkpoint also creates
multiple files across ranks. A model bundle needs complete ownership,
publication, restore, and replay write protection; intercepting an optimizer
file alone does not preserve the model.
[Transformers model saving](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.save_pretrained),
[PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html).

## Serving alternatives and additional requirements

| Need | Natural serving alternative | Consequence for Flor |
| --- | --- | --- |
| More request throughput; each model replica fits on one GPU | Independent replicas with request routing or engine data parallelism. | Track model/version and request ownership across replicas; requests are not gradient-synchronized training ranks. [vLLM data parallel deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/). |
| A model replica spans several GPUs | Tensor or pipeline parallel inference, for example vLLM. | Engine workers cooperate for one request; recording and replay need engine topology and integration. vLLM documents native multiprocessing and Ray execution. [vLLM scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/). |
| Evaluate historical models or run offline batch inference | Finite inference/evaluation scripts, optionally distributed. | Closest to existing Flor runs. A useful first serving-related target before long-lived servers; distributed engines still need their own integration. This is a scope recommendation. |

Online serving adds design questions beyond forking: what bounds a recording
window, when observations become durable without shutdown, and what request
inputs, generation settings, model revisions, and random state enable replay.
Dynamic batching, concurrency, and engine caches complicate equivalence.
Specify output/metric reproducibility separately from reproducing latency or
request scheduling. Generic child-run IDs alone do not supply this contract.

## Accepted scope (D1)

Resolved 2026-10-27. These are release priorities and boundaries, not a claim
that the capability is implemented.

1. **First training target:** single-host PyTorch DDP through `torchrun`, then
   Hugging Face Trainer/Accelerate using DDP with a documented instrumentation
   surface and supported model-save format. Include independent GPU trials.
2. **Validation:** CPU/Gloo first for development, followed by actual two-GPU
   CUDA/NCCL tests before claiming multi-GPU support. Exercise helper workers,
   supported start methods, worker output, failure cleanup, deterministic data
   loading, historical selection, and rank-zero checkpoint saves.
3. **Separate integrations:** programmatic spawn, FSDP/DeepSpeed, and
   tensor/pipeline parallelism each require named fixtures and artifact rules.
   Do not claim them as automatic consequences of DDP support.
4. **Serving follow-on:** finite model evaluation/offline inference first. Choose
   an engine and request/window contract before promising online serving replay.
5. **Fork decision:** retain the process-safety work needed by supported
   framework workers; defer arbitrary nested-fork hindsight replay.

D1 is resolved. D2–D5 in the review still govern launch integration, numerical
fidelity, checkout isolation, and commit granularity. The Hugging Face
instrumentation/model-save surface and helper-worker behavior need concrete
engineering designs within the accepted scope.
