# Comparing the Many Versions of a Model

## Path 1: Pull the model in Jupyter

Start Jupyter inside the experiment's Git repository so FlorDB finds its local
database and object store.

**1. Find the runs.** `flor.dataframe()` holds each run's timestamp and recorded
arguments. `flor.checkpoints(tstamp)` lists what one run saved (`name`, `kind`,
`backend`, `path`); a `run` checkpoint is the latest forward save under that
name.

```python
import flordb as flor

runs = flor.dataframe()
runs = runs[runs.source == "forward"].drop_duplicates("tstamp")
flor.checkpoints(runs.iloc[0].tstamp)
```

**2. Load a model.** Pass the name your script saved under:

```python
row = runs.iloc[0]
checkpoint = flor.load_checkpoint(row.tstamp, "ckpt.pth")
model = make_model(row)  # your model factory, using the recorded architecture arguments
model.load_state_dict(checkpoint["model"])
```

What to pass to `load_state_dict` depends on how the script saved:

| Your script | State dictionary |
|---|---|
| `torch.save({"model": net.state_dict(), ...}, "ckpt.pth")` | `flor.load_checkpoint(tstamp, "ckpt.pth")["model"]` |
| `torch.save(net.state_dict(), "ckpt.pth")` | `flor.load_checkpoint(tstamp, "ckpt.pth")` |

The name can be omitted only when the run has exactly one run checkpoint.

FlorDB loads saved state; it never imports or runs historical model code. You
write `make_model(row)`, in the notebook or in a module that doesn't start
training on import. It must build the architecture that run used, picking the
right definition if the architecture changed between runs. The
[comparison notebook](../notebooks/compare_models.ipynb) is a complete example
for `examples/v4/train.py`.

### Compare a new metric across runs

Gradients aren't saved in a state dictionary, so to compare `grad_norm`, run a
fresh backward pass on the **same evaluation batch and loss** for every model.
With `make_model`, `inputs`, `targets`, and `loss_fn` defined in your notebook:

```python
import pandas as pd
import torch

measurements = []
for row in runs.itertuples(index=False):
    model = make_model(row).cpu()
    model.load_state_dict(flor.load_checkpoint(row.tstamp, "ckpt.pth")["model"])
    model.eval()
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(inputs.cpu()), targets.cpu())
    loss.backward()
    grad_norm = sum(
        p.grad.detach().double().square().sum()
        for p in model.parameters() if p.grad is not None
    ).sqrt().item()
    measurements.append({"tstamp": row.tstamp, "grad_norm": grad_norm})

comparison = runs.merge(pd.DataFrame(measurements), on="tstamp", validate="one_to_one")
comparison.sort_values("grad_norm")
```

The measurements stay in your notebook. Loading a checkpoint doesn't start a
run or write replay logs.

### What gets loaded

- **The last save, which may not be the final model.** Each forward run keeps
  one copy of each file it saves, replaced on every save. If your script saves
  only its best model, or exits early, that is what you load.
- **When copies are taken.** Once a Flor run is active (for example, after
  `flor.arg` or `flor.loop`), every successful `torch.save` to a file path
  replaces the run's copy, inside the loop or after it. Saves to file-like
  objects are not captured. See [Checkpoints](checkpoints.md).
- **Older runs** may have only `iteration` snapshots. FlorDB won't quietly use
  one as the run's final model, so pick it by name from
  `flor.checkpoints(tstamp)`, e.g.
  `flor.load_checkpoint(tstamp, "ckpt_epoch_2.pth")`.
- **Device and trust.** Checkpoints load onto the CPU unless you pass
  `map_location`. `weights_only=True` is the default. Only load checkpoints you
  trust, especially cloudpickle snapshots or torch files that need
  `weights_only=False`.
- **Checkpoints don't travel with Git**, and `flor unpack` rebuilds only the
  database. Without the object store, `flor.checkpoints` returns an empty
  dataframe and `flor.load_checkpoint` raises `FileNotFoundError`. To inspect a
  run on another machine, copy `.flor/obj_store/<tstamp>/` from the one that
  trained it.

## Path 2: Push the code with Hindsight Logging

Forgot to log something? Add the logging statement now and recover the values
from runs that already happened.

```python
flor.log("grad_norm", ...)     # add to the script
```

```bash
python -m flordb replay --apply grad_norm
```

FlorDB first commits all unignored working-tree changes, including untracked
files, before it estimates the cost or asks you to confirm. After you confirm,
it goes through each historical run: it checks out the run's commit, splices
your new statement into that version of the script, and re-executes it from
the start.

Recovered values are tagged `source = 'replay'`, which keeps them separate from
`forward`, the values the original run actually observed. They live only in the
local database: they aren't added to the run's JSONL history, and
`python -m flordb unpack` clears them.

### Narrowing the work

`--iter` selects which loop iterations to log:

```bash
python -m flordb replay --apply grad_norm --iter epoch=0,2 --iter step=all
```

- With any `--iter` flag, loops you don't mention default to `last`.
- With no `--iter` flags, the default depends on where the log statement sits.
  At the first or second level of loop nesting, every outer iteration is
  selected and inner loops stay at `last`.
- FlorDB can't determine `last` for manually managed `flor.iteration` blocks. It
  warns and logs every iteration instead.

`--iter` chooses which iterations log, not which ones run. Replay always starts
at iteration 0 and runs the outermost loop through the last selected iteration,
with logs off for the iterations in between. Nested loops always run in full,
because what comes after them depends on the state they leave. So
`--iter epoch=9` costs ten epochs, and `--iter step=none` still trains.

### Replaying one run directly

To skip the orchestrator, pass the same verbs to the script itself:

```bash
python train.py --replay_flor \
    --apply loss,val_acc \
    --iter epoch=0,2 \
    --iter step=all \
    --override device=cpu
```

This runs the **current** script against the latest run record in `.flor/runs/`
of the current checkout. Nothing is checked out and no code is spliced in.
Unmentioned `flor.loop` loops default to `last`.

`--override` is for environment settings such as `device`. It is checked
against the run-level `flor.arg` keys the historical run recorded:

- Recorded keys are rejected, except `device`.
- Keys the run never recorded are allowed.

This check covers key names only. It doesn't guarantee that an override leaves
the results unchanged.

### Replaying in a fresh clone

First, populate the local database from the committed run records:

```bash
python -m flordb unpack
```

That's all a clone needs, unless a run resumed from an earlier run's
checkpoint. Replaying that run needs the earlier run's copy in
`.flor/obj_store/`; without it, replay stops and names the run and commit to
copy it from. If the script guards its resume load with a working-file existence
check, restore the recorded copy to that path too, or make the load
unconditional. Replay refuses to proceed when the guard skips a recorded
setup load.

Replaying from iteration 0 assumes deterministic seeding. A run that resumed
from an earlier run starts from the same checkpoint on replay. Otherwise, the
script's resume load becomes a no-op, so it can't put end-of-run weights over
the initialization. See
[Checkpoints during replay](checkpoints.md#checkpoints-during-replay).
