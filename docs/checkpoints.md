# Checkpoints

FlorDB mirrors model snapshots into its own object store, addressable by loop
iteration, so replay can restart from a nearby iteration instead of from scratch.

## An existing `torch.save` is enough

If your script already saves checkpoints inside a named loop, you're done (no
`with flor.checkpointing(...)` block is required):

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

## Putting a mirror back

Mirroring needs nothing from you — FlorDB copies whatever you saved. Putting a
mirror *back* is the half that needs semantics: which object does this file
belong in? FlorDB works that out from your script's source, preferring the most
direct evidence it can find.

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
torch.save(net.state_dict(), "ckpt.pth")       # somewhere in the loop
...
net.load_state_dict(torch.load("ckpt.pth"))    # no key: the file *is* net's state
```

From that FlorDB learns which object takes which slice of the file, and on
replay re-runs those same calls per iteration with `torch.load` redirected to
that iteration's mirror. Your own resume block is left intact — it still runs
once before the loop.

### No resume block at all

Plenty of scripts checkpoint and never resume. There's still nothing to declare:
your `torch.save` says what the file holds, one step less directly than a load
would, and FlorDB reads the mapping off it instead.

```python
torch.save(net.state_dict(), "ckpt.pth")                    # -> the file is net's state
torch.save({"model": net.state_dict(),                      # -> "model" is net's,
            "optimizer": opt.state_dict()}, "ckpt.pth")     #    "optimizer" is opt's
```

Entries that aren't a `state_dict()` — the `"epoch"` and `"loss"` most
checkpoints carry — are skipped; they aren't things to restore into.

This is the weakest of the three signals, and replay says out loud what it
concluded, because saving and restoring aren't the same statement:

```python
best = copy.deepcopy(net)          # the loop trains net
torch.save(best.state_dict(), "ckpt.pth")
```

Here the mapping FlorDB reads is *`ckpt.pth` holds `best`'s state*, which is
true — and useless, because `net` is the object your metrics come from. Nothing
fails: `best` exists, takes the state, and the shapes fit. So this is the one
inference that can be wrong without raising, and the startup line naming the
target it picked is the only warning you get. If it named the wrong one, say so
with `flor.restore(...)`.

A resume block wins over this whenever you have one, and `flor.restore` wins
over both.

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
then — and takes precedence over anything FlorDB inferred. Because replay
re-reads your script from disk, you can add the declaration to a run that
already happened; there's no need to train again.

## Replay refuses to guess

Restoring the wrong iteration is worse than not replaying at all: the numbers
still come out, and they look historical. So the paths that could substitute
one iteration's state for another's raise instead.

- **A mirror the run needs isn't on the shelf.** After a forward run your own
  `ckpt.pth` holds *end-of-run* weights, so falling back to it would answer
  "what did epoch 3 look like?" with the last epoch's model. `flor.loop` routes
  around a throttled iteration by restarting from the nearest earlier mirror
  and recomputing forward; where it can't — `flor.iteration`, which doesn't own
  its iteration space — replay stops and tells you to narrow to an iteration
  that has one.
- **A target doesn't take its state.** A name that no longer resolves, an
  object with no `load_state_dict`, a key the checkpoint never carried, tensors
  that no longer fit because the model changed shape — each used to be skipped
  silently, leaving that object on some other iteration's state. All of them
  now fail the replay, naming the target and (for an inferred mapping) pointing
  at `flor.restore` as the correction.

An unrelated `torch.load` is untouched: the check is whether the shelf holds
other iterations of *that* file, so cached tensors and datasets load normally.

## Enrolling objects explicitly

The `torch.save` hook is a piggy-back: it only fires if your script already
saves inside a loop. When it doesn't — you save once at the end, or not at all —
enroll the objects yourself with `flor.checkpointing(...)`, and they're
serialized at every checkpoint trigger:

```python
with flor.checkpointing(model=net, optimizer=optimizer):
    for epoch in flor.loop("epoch", range(epochs)):
        ...
```

This accepts anything, torch included — `torch.nn.Module` and
`torch.optim.Optimizer` are the serializer's first case, stored via their
`state_dict()`. It's also the only path for objects `torch.save` would never
see: scikit-learn estimators, numpy arrays, pandas DataFrames, and plain
Python objects (via cloudpickle).

The same interval throttles this path — enrolling objects doesn't add a
second stream of snapshots on top of the hook's.

## Bounding disk use

```python
flor.set_ckpt_interval(seconds)   # default: at most one snapshot every 60s
```

Checkpoints are the expensive artifact — roughly 19MB per run, and they don't
dedup — so they stay out of git. See [Storage](storage.md) for what that means
for teammates, and how the first replay in a fresh clone warms the store.
