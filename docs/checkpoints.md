# Checkpoints

FlorDB checkpoints PyTorch models, optimizers, and other training objects so
replay can restore them without repeating expensive GPU training. Checkpoints
are stored under `.flor/obj_store/<tstamp>/`, indexed by loop iteration. When
your script calls `torch.save`, FlorDB's own copy of that file is called a
**mirror**.

## How do I save checkpoints?

Choose based on how your training script saves checkpoints today:

| Your script | What to use | When state is captured |
|---|---|---|
| Calls `torch.save` inside `flor.loop` | Keep the existing save | At the first save in an eligible outer iteration |
| Does not save inside the loop | Wrap the loop in `flor.checkpointing(...)` | At the end of eligible outer iterations, plus a final snapshot at loop exit |

### Keep an existing save

```python
for epoch in flor.loop("epoch", range(epochs)):
    ...
    torch.save({"model": net.state_dict(), "optimizer": opt.state_dict()}, "ckpt.pth")
```

During a forward run, your `ckpt.pth` is written as usual. FlorDB mirrors it into
its object store (`.flor/obj_store/<tstamp>/`), once per eligible iteration.
During replay, FlorDB suppresses writes to your checkpoint file. Recomputation
may still fill missing snapshots directly in the object store, as described
below.

### Name the objects to save

```python
with flor.checkpointing(model=net, optimizer=opt):
    for epoch in flor.loop("epoch", range(epochs)):
        ...
```

The objects named here are **enrolled**: FlorDB saves them at every eligible
iteration and restores them during replay. The block declares what to
checkpoint; it does not load anything during a normal run.

Enroll the model, optimizer, and any schedulers or gradient scalers your loop
uses. FlorDB saves and restores them through `state_dict()` and
`load_state_dict()`, updating the existing objects in place.

### Control how often state is saved

```python
flor.set_ckpt_interval(60)  # default, in seconds; use 0 for every iteration
```

The first outer iteration is always eligible. At the start of each later outer
iteration, FlorDB checks whether the interval has elapsed since the last
eligible one; it does not save on a background timer.

The interval reduces how often snapshots are taken, but it does not cap disk
usage — total size depends on the objects and the number of snapshots.
Checkpoints stay out of git; see [Storage](storage.md).

## How does replay know what to restore?

Enrolled objects are restored using the references passed to
`flor.checkpointing`. For a mirrored `torch.save` file, FlorDB uses these rules:

| Priority | Source | Meaning |
|---|---|---|
| 1 | `flor.restore(path, ...)` | You explicitly name the destination objects. |
| 2 | A recognized resume block | FlorDB reads the file-to-object mapping from your load calls. |
| 3 | A recognized `torch.save` | FlorDB infers the destinations from the objects being saved and announces the mapping. This fallback is skipped when objects are explicitly enrolled. |

### Declare the mapping explicitly

After constructing your objects, before the loop, use one of these forms:

```python
flor.restore(path, net)                       # whole file is net's state_dict
flor.restore(path, model=net, optimizer=opt)  # dict keys -> destination objects
```

These are alternatives, depending on the file format. The path may be computed.
Targets must support `load_state_dict()`.

`flor.restore` is a declaration for replay: it registers the objects, and the
loop restores their historical checkpoints. You can add the declaration to an
existing script and replay without training again.

### Let FlorDB read your resume block

FlorDB recognizes these resume patterns at module scope or inside functions
and methods, using literal paths and simple variable names:

```python
_resume = torch.load("ckpt.pth")
net.load_state_dict(_resume["model"])
opt.load_state_dict(_resume["optimizer"])
```

The load and apply calls must be in the same scope. Replay keeps references to
those objects.

Or, for a file containing a single state dictionary:

```python
net.load_state_dict(torch.load("ckpt.pth"))
```

### Let FlorDB infer from your `torch.save`

With no recognized resume block, FlorDB reads the mapping off the save itself:

```python
torch.save(net.state_dict(), "ckpt.pth")
# Or:
torch.save({"model": net.state_dict(), "optimizer": opt.state_dict()}, "ckpt.pth")
```

Inference names the object that was saved, which is not always the object you
want restored. If your loop trains `net` but saves a copy —
`torch.save(best.state_dict(), "ckpt.pth")` — inference selects `best`, and
nothing fails, because the shapes fit. Check the mapping FlorDB prints at
replay startup, and use `flor.restore` if it names the wrong destination.

Computed paths and unrecognized helper functions may also need an explicit
`flor.restore`. State spread over multiple files needs consolidating into one
file, or explicit enrollment of the objects: multiple `flor.restore` calls do
not combine files — each replaces the previous mapping.

### Moving between GPU and CPU

GPU-written checkpoints restore on a CPU-only machine, and vice versa. Construct
the live objects on whatever device the current run has; FlorDB reads every
checkpoint through the CPU and lets `load_state_dict()` copy the state onto the
device those objects already sit on. This holds for both enrolled objects and
mirrored `torch.save` files.

Reading through the CPU is also the cheaper route on a GPU box: the state
crosses the bus exactly once either way, and staging it in host memory keeps a
second full copy of it off the GPU.

Your own `torch.load` calls are untouched — FlorDB forwards their arguments as
written, including `map_location`.

## What happens when a checkpoint is missing?

For an explicit request such as `--iter epoch=1`, with a restore mapping or
enrolled objects:

| Situation | Replay behavior |
|---|---|
| Every requested iteration has a checkpoint, and its nested loops are skipped | Restore each requested iteration's end state before entering its body. |
| A requested checkpoint is missing, or a nested loop will execute | Restore the latest checkpoint strictly before the first requested iteration, then recompute forward through the last requested one, running nested loops in full. |
| No checkpoint exists at or before the first requested iteration | Execute from iteration 0. Reproducing the original state requires deterministic initialization and computation. |

During recomputation, FlorDB suppresses logs from the outer iterations you did
not request; nested loops log every iteration they run. Eligible missing
snapshots are shelved for later replays, without overwriting existing ones. This
caching is disabled under overrides that could change the numbers, such as
`device=cpu`.

When replay must retrain from iteration 0, FlorDB skips loading the final
trained weights through recognized resume code. The model and optimizer keep
their initial values so replay can reproduce the training run. See
[Replay](replay.md) for selection options and replay in a fresh clone.

### Worked example: inspecting the state after epoch 1

This is the structure used by the checkpoint replay tests. Epochs are numbered
from zero; training happens inside the nested `step` loop.

```python
flor.set_ckpt_interval(0)
for epoch in flor.loop("epoch", range(3)):
    for step in flor.loop("step", range(2)):
        ...  # compute loss, backpropagate, and update model and optimizer
    flor.log("weight_sum", float(sum(p.sum() for p in model.parameters())))
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, "ckpt.pth")
```

To inspect epoch 1:

```bash
python train.py --replay_flor --apply weight_sum \
    --iter epoch=1 --iter step=none
```

| Checkpoints available | What executes |
|---|---|
| Epoch 1 exists | Load epoch 1's saved state, skip the training steps, evaluate `weight_sum`. |
| Only epoch 0 exists | Load epoch 0's saved state, recompute epoch 1 — both training steps run despite `step=none` — and evaluate `weight_sum`. |
| None exist | Run epochs 0 and 1, including their training steps; emit the requested metric only for epoch 1. |

`step=none` skips the training steps only in the first row, where epoch 1's
checkpoint makes them unnecessary; use `--iter step=all` to rerun them there.

Recomputation ignores nested selections entirely: `step=last` and specific step
indices behave like `step=all`, and every step logs.

Skipping a nested loop skips only that loop, not the rest of the iteration
body. In the example above, replay still reaches `flor.log` and `torch.save`,
but the save hook checks the replay flag and leaves your `ckpt.pth` untouched.
Eligible missing snapshots may be cached directly in FlorDB's object store.