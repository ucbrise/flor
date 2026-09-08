# Checkpoints

FlorDB mirrors model snapshots into its own object store, addressable by loop
iteration, so replay can restart from an iteration instead of from scratch.

## An existing `torch.save` is enough

If your script already saves checkpoints inside a named loop, you're done — no
`with flor.checkpointing(...)` block is required:

```python
for epoch in flor.loop("epoch", range(epochs)):
    ...
    torch.save({"model": net.state_dict()}, "ckpt.pth")
```

FlorDB writes a second, independent copy into `.flor/obj_store/<tstamp>/`,
named by the iteration it was taken at. Your own `ckpt.pth` is written exactly
as before.

Mirroring is rate-limited: at most one snapshot every `ckpt_interval_s`
(default 60 seconds), so a fast loop produces a mirror every *minute*, not
every epoch. Replay restarts from the nearest one it finds.

## Non-torch objects

Use `flor.checkpointing(...)` to enroll objects `torch.save` won't see —
scikit-learn estimators, plain dicts:

```python
with flor.checkpointing(model=clf, optimizer=opt):
    for epoch in flor.loop("epoch", range(epochs)):
        ...
```

## Bounding disk use

```python
flor.set_ckpt_interval(seconds)   # default: at most one snapshot every 60s
```

Checkpoints are the expensive artifact — roughly 19MB per run, and they don't
dedup — so they stay out of git. See [Storage](storage.md) for what that means
for teammates, and how the first replay in a fresh clone warms the store.
