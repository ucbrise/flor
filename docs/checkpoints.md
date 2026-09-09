# Checkpoints

FlorDB mirrors model snapshots into its own object store, addressable by loop
iteration, so replay can restart from a nearby iteration instead of from scratch.

## An existing `torch.save` is enough

If your script already saves inside a named loop, you're done — no
`with flor.checkpointing(...)` required:

```python
for epoch in flor.loop("epoch", range(epochs)):
    ...
    torch.save({"model": net.state_dict()}, "ckpt.pth")
```

FlorDB writes a second, independent copy into `.flor/obj_store/<tstamp>/`, named
by the iteration it was taken at; your own `ckpt.pth` is written exactly as
before. Mirroring is rate-limited to one snapshot every `ckpt_interval_s`
(default 60 seconds), so a fast loop produces a mirror every *minute*, not every
epoch. Replay restarts from the nearest one it finds.

## Putting a mirror back

Mirroring needs nothing from you — FlorDB copies whatever you saved. Putting a
mirror *back* is the half that needs semantics: which object does this file
belong in? FlorDB reads that from your script's source, preferring the most
direct evidence it can find — `flor.restore` over a resume block, a resume block
over an inferred `torch.save`.

### From your resume block

The best evidence, because it names the object each slice goes into outright.
Module scope, in either of the two shapes people write. Keyed, when the
checkpoint is a dict:

```python
_resume = torch.load("ckpt.pth")          # literal path, plain assignment
net.load_state_dict(_resume["model"])     # subscript with a literal key
optimizer.load_state_dict(_resume["optimizer"])
```

or flat, when the file holds one object outright:

```python
net.load_state_dict(torch.load("ckpt.pth"))    # no key: the file *is* net's state
```

On replay FlorDB re-runs those same calls per iteration with `torch.load`
redirected to that iteration's mirror. Your own resume block is left intact — it
still runs once before the loop.

### No resume block at all

Plenty of scripts checkpoint and never resume. There's still nothing to declare:
your `torch.save` says what the file holds, and FlorDB reads the mapping off it
instead.

```python
torch.save(net.state_dict(), "ckpt.pth")                    # -> the file is net's state
torch.save({"model": net.state_dict(),                      # -> "model" is net's,
            "optimizer": opt.state_dict()}, "ckpt.pth")     #    "optimizer" is opt's
```

Entries that aren't a `state_dict()` — the `"epoch"` and `"loss"` most
checkpoints carry — are skipped; they aren't things to restore into.

Saving and restoring aren't the same statement, though, so this is the one
inference that can be wrong without raising:

```python
best = copy.deepcopy(net)          # the loop trains net
torch.save(best.state_dict(), "ckpt.pth")
```

The mapping FlorDB reads here — *`ckpt.pth` holds `best`'s state* — is true and
useless: `net` is the object your metrics come from, but `best` exists, takes
the state, and the shapes fit, so nothing fails. Replay names the target it
picked on startup, and that line is your only warning. If it named the wrong
one, say so with `flor.restore(...)`.

### What inference can't reach

A path that isn't a literal (FlorDB has to name the mirror file before your
script runs), a resume block inside a function, a `torch.save` inside a helper
(its locals name objects replay has no frame to reach), and state split across
two checkpoint files (replay addresses one file per run, and restoring half of
your state is worse than refusing). Replay says so on startup in each case.
Answer it with `flor.restore(...)`, placed where the resume block would go:

```python
flor.restore(path, net)                        # whole file into net
flor.restore(path, model=net, optimizer=opt)   # keyword = key in the saved dict
```

The path is an ordinary argument here, so a computed one is fine.

On a forward run this *is* your resume block: if the file exists it loads and
applies it, so an interrupted run picks up where it left off. On replay it
applies nothing at module scope — the per-iteration restore owns those objects
then. Because replay re-reads your script from disk, you can add the declaration
to a run that already happened; there's no need to train again.


## Enrolling objects explicitly

The `torch.save` hook is a piggy-back: it only fires if your script already
saves inside a loop. When it doesn't — you save once at the end, or not at all —
enroll the objects yourself, and they're serialized at every checkpoint trigger:

```python
with flor.checkpointing(model=net, optimizer=optimizer):
    for epoch in flor.loop("epoch", range(epochs)):
        ...
```

What it accepts is decided by shape, not by class. Anything carrying
`state_dict()` / `load_state_dict()` is the serializer's first case — your model
and optimizer, but equally the LR scheduler and the `GradScaler` beside them,
which are neither a `Module` nor an `Optimizer` and are just as much part of the
state a run resumes from:

```python
with flor.checkpointing(model=net, optimizer=opt, sched=sched, scaler=scaler):
```

Objects `torch.save` would never see go through the same block — numpy arrays,
pandas DataFrames, dicts, and plain Python objects (via cloudpickle, restored
through their instance dict). An object flor can serialize but could never put
back — no `state_dict`, no mapping, no `__dict__` — is refused by
`flor.checkpointing` itself, on the forward run, rather than shelving snapshots
for a replay that would fail on them.

Enrollment restores across machines: mirrors are read back with
`map_location="cpu"` and copied into whatever device your live objects are on,
so a checkpoint written on a GPU box replays on one without.

Enrolling is a second stream of snapshots, not a replacement for the hook's: if
your script also calls `torch.save`, both are written. They are taken at the
same iterations, though — one trigger decision per iteration, honored by both —
which is what lets replay restore an iteration outright instead of recomputing
its way there.

One thing enrollment does not do is resume a forward run. `flor.checkpointing`
only restores under replay; if you want an interrupted run to pick up where it
left off, that is still your own `torch.save`/`torch.load` (or
[`flor.restore`](#what-inference-cant-reach), which does it for you).

## Bounding disk use

```python
flor.set_ckpt_interval(seconds)   # default: at most one snapshot every 60s
```

Checkpoints are the expensive artifact — roughly 19MB per run, and they don't
dedup — so they stay out of git. See [Storage](storage.md) for what that means
for teammates, and how the first replay in a fresh clone warms the store.
