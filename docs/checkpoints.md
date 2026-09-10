# Checkpoints

FlorDB keeps one copy of each checkpoint file your script saves, per run. Your
`torch.save` writes its file as usual, and FlorDB copies it into
`.flor/obj_store/<tstamp>/`. The next run overwrites your file, but the copy
keeps this run's version, so every run's model stays available to
[compare in a notebook](replay.md#path-1-pull-the-model-in-jupyter) or to
resume from later.

## What gets saved

There's nothing to set up. Save the way you already do:

```python
for epoch in flor.loop("epoch", range(epochs)):
    ...
    torch.save({"model": net.state_dict(), "optimizer": opt.state_dict()}, "ckpt.pth")
```

- **One copy per saved path, per run.** Every successful `torch.save` to that
  path replaces the run's copy, so it holds the last save. If your script saves
  only its best model, or stops early, that is what's kept.
- **Saves count once a Flor run is active** (for example, after `flor.arg` or
  `flor.loop`), inside the loop or after it.
- **Only saves to a file path are copied.** Saves to file-like objects, such as
  an open file or `io.BytesIO`, are not.
- **Disk use grows by one checkpoint per saved path per run**, however many
  iterations the run has. Copies stay out of Git; see [Storage](storage.md).

## Resuming from an earlier run

A run can pick up where an earlier one left off. FlorDB records which run it
started from, by that run's tstamp and commit, and replay starts from the same
checkpoint. There are two ways to resume.

**Ask FlorDB for the checkpoint.** Choose the run with any query over
`flor.dataframe()`, such as the best `val_acc` among runs with a given width,
and load its copy:

```python
runs = flor.dataframe("hidden", "val_acc")
runs = runs[(runs.source == "forward") & (runs.hidden == 500)]
best = runs.sort_values("val_acc").iloc[-1].tstamp

state = flor.load_checkpoint(best, "ckpt.pth")
net.load_state_dict(state["model"])
opt.load_state_dict(state["optimizer"])
```

This doesn't depend on what's in your working directory. FlorDB records the run
the query returned, so replay loads that checkpoint even if the same query
picks a different run by then. Separate loads of the same checkpoint name are
recorded in call order, so a script can load that name from multiple runs.
Keep that load order when replaying. If a query or guard causes an extra load
that the historical run never made, replay stops instead of loading a new
starting state. This includes a fresh run whose initially empty query now
finds checkpoints.

**Load your own file.** A resume block that loads the checkpoint the last run
saved works as it is:

```python
if os.path.exists("ckpt.pth"):
    state = torch.load("ckpt.pth")
    net.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
```

When the script loads a file before its training loop, FlorDB checks it against
the copies earlier runs kept: each copy records the file's path, size, and
modification time from when its run saved it. A match means the file is
exactly what that run saved. A file edited or replaced since, or one no FlorDB
run wrote, matches nothing, and the run counts as a fresh start.

## Checkpoints during replay

[Replay](replay.md#path-2-push-the-code-with-hindsight-logging) re-executes a
run from iteration 0. It never resumes from a checkpoint partway through a run,
and it leaves your checkpoints as they are:

- **Your file is left alone.** Replay skips the script's `torch.save` calls, so
  `ckpt.pth` keeps whatever the latest forward run wrote. Replay doesn't write
  copies either.
- **A resumed run starts from the same checkpoint.** Replay answers the
  script's load with the copy the run recorded. If that copy isn't in
  `.flor/obj_store/`, as in a fresh clone, replay stops and names the run and
  commit whose copy it needs. The script must execute its original setup load:
  if a missing working file makes an existence guard skip it, replay stops
  before training. The error names the saved copy; restore that file to the
  path checked by the guard, or make the original load unconditional, and retry.
  Replay does not create or overwrite the working file for you.
- **A fresh start stays fresh.** When the run recorded no start, a resume
  block's load becomes a no-op during replay, so it can't put the latest run's
  weights over the initialization. FlorDB recognizes the block above, or a
  single state dictionary loaded directly:

  ```python
  net.load_state_dict(torch.load("ckpt.pth"))
  ```

  The block can sit at module scope or inside one function or method. The path
  must be a literal, and the load and its `load_state_dict` calls must share a
  scope. If your script calls `torch.load` before the loop in any other form,
  FlorDB warns when replay starts, because it can't tell whether that load
  brings in trained weights. If it recognizes a block but can't neutralize it,
  replay stops rather than recompute from the wrong starting point.

Replaying from iteration 0 reproduces the run only if the script seeds
deterministically, for example with `torch.manual_seed(flor.arg("seed", 42))`.

## Older scripts and runs

`flor.checkpointing(...)`, `flor.restore(...)`, and `flor.set_ckpt_interval(...)`
no longer do anything. Scripts that call them still run, so replay can still
execute older versions of your code, but FlorDB prints a notice and you can
delete the calls. If a script relied on `flor.checkpointing(model=net)` to keep
its model, save the model with `torch.save` instead.

Runs recorded before this change may have one checkpoint per iteration instead
of a single copy. `flor.checkpoints(tstamp)` lists them with `kind ==
"iteration"`, and `flor.load_checkpoint(tstamp, name)` loads one by name. They
also recorded no starting point, so a run among them that resumed from an
earlier one replays from initialization.
