# Hindsight Logging and Replay

Forgot to log something? Add the logging statement now and recover the values
from runs that already happened.

```python
flor.log("grad_norm", ...)     # add to the script
```

```bash
python -m flordb replay --apply grad_norm
```

FlorDB estimates the cost, asks for confirmation, then walks the historical
versions: it checks each run's commit out, splices your new statement into that
version of the script, restarts from the nearest checkpoint, and records the
recovered values.

Values recovered this way are tagged `source = 'replay'`, so they stay
distinguishable from `forward` — what the original run actually observed.

## Narrowing the work

Loops you don't mention default to their last iteration:

```bash
python -m flordb replay --apply grad_norm --iter epoch=0,2 --iter step=all
```

## Replaying one run directly

No orchestration, no git checkout — the same verbs are flags on the script
itself:

```bash
python train.py --replay_flor \
    --apply loss,val_acc \
    --iter epoch=0,2 \
    --iter step=all \
    --override device=cpu
```

`--override` exists for environment-shaped settings such as `device` —
replaying on a different machine is fine, but hyperparameters that defined the
original run are rejected, since changing those makes it a new experiment
rather than a replay.

## Replaying in a fresh clone

Checkpoints don't travel with git, so the first replay of a run in a fresh clone
has to recompute from iteration 0. FlorDB does that automatically and shelves
the checkpoints it passes on the way, so only the first replay pays the cost.
Two rules keep that from corrupting history: a recomputed checkpoint never
overwrites one the forward run wrote, and warming is disabled under an
`--override` that could move the numbers (`device=cpu`).

Replaying from iteration 0 assumes the script seeds deterministically. If your
script has a resume block (`torch.load("ckpt.pth")` at module scope), FlorDB
neutralizes it for that first replay so it can't load end-of-run weights over
the fresh initialization — your checkpoint file is left where it is, untouched.
