# Automatic Log Capture

FlorDB captures what your script already prints. Adding `import flordb as flor`
to a script that uses `print` or `logging` is enough — the terminal looks the
same, but every line is versioned, committed, and queryable.

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

import flordb as flor          # <-- the only new line

for epoch in range(3):
    print(f"epoch {epoch} | loss: {1.0 / (epoch + 2):.4f}")
    logging.info("checkpoint saved")
```

```python
flor.io()
```

```
  projid                     tstamp  filename   source        channel                    line
0  zero2 2026-08-13 12:34:47.951515  train.py  forward     io::stdout  epoch 0 | loss: 0.5000
1  zero2 2026-08-13 12:34:47.951515  train.py  forward  io::log::info        checkpoint saved
2  zero2 2026-08-13 12:34:47.951515  train.py  forward     io::stdout  epoch 1 | loss: 0.3333
3  zero2 2026-08-13 12:34:47.951515  train.py  forward  io::log::info        checkpoint saved
4  zero2 2026-08-13 12:34:47.951515  train.py  forward     io::stdout  epoch 2 | loss: 0.2500
5  zero2 2026-08-13 12:34:47.951515  train.py  forward  io::log::info        checkpoint saved
```

## Channels

Captured lines carry a channel: `io::stdout`, `io::stderr`, and
`io::log::<level>` (`io::log::info`, `io::log::error`, …). Filter by passing one
to `flor.io("io::stdout")`, or treat it as a column with
`flor.dataframe("io::stdout")`.

Capture is not installed in IPython (where explicit `flor.log` is the path), for
`python -c`, or for flor's own CLI commands.

## Name the loop

Naming a loop tells FlorDB what an iteration is, so each captured line is tagged
with the iteration it came from:

```python
for epoch in flor.loop("epoch", range(3)):   # was: for epoch in range(3):
```

```
  projid                     tstamp  filename   source  epoch        channel                    line
0  zero2 2026-08-13 12:34:49.944234  train.py  forward      0     io::stdout  epoch 0 | loss: 0.5000
1  zero2 2026-08-13 12:34:49.944234  train.py  forward      0  io::log::info        checkpoint saved
2  zero2 2026-08-13 12:34:49.944234  train.py  forward      1     io::stdout  epoch 1 | loss: 0.3333
3  zero2 2026-08-13 12:34:49.944234  train.py  forward      1  io::log::info        checkpoint saved
4  zero2 2026-08-13 12:34:49.944234  train.py  forward      2     io::stdout  epoch 2 | loss: 0.2500
5  zero2 2026-08-13 12:34:49.944234  train.py  forward      2  io::log::info        checkpoint saved
```

That `epoch` column is what the rest of FlorDB is built on: checkpoints
addressable by iteration (see [Checkpoints](checkpoints.md)), and replay that
can jump to one (see [Replay](replay.md)).

## Extracting metrics from captured text

Captured text stays out of `flor.dataframe()` — it isn't a metric — but FlorDB
can read metrics out of it on demand. This runs over runs you have already
recorded, so there is nothing to turn on and no reason to train again. See what
it would do to *your* logs first:

```bash
python -m flordb capture --preview
```

```
Would extract 3 value(s) across 1 metric(s) from 6 captured line(s): loss
Indexed by: epoch

  epoch=0  loss  0.5          <- epoch 0 | loss: 0.5000
  epoch=1  loss  0.3333       <- epoch 1 | loss: 0.3333
  epoch=2  loss  0.25         <- epoch 2 | loss: 0.2500

Nothing was written. Run again with --extract to write them.
```

Every line shows the value, and the text it was read out of, so a bad guess is
visible before it becomes a column. When the reading looks right, write it:

```bash
python -m flordb capture --extract
```

```
  projid                     tstamp  filename   source  epoch    loss
0  zero2 2026-08-13 12:34:49.944234  train.py  extract      0     0.5
1  zero2 2026-08-13 12:34:49.944234  train.py  extract      1  0.3333
2  zero2 2026-08-13 12:34:49.944234  train.py  extract      2    0.25
```

`loss` is a real column in `flor.dataframe("loss")` now, and the raw line stays
on record. Nothing is lost by guessing wrong: run `--extract` again and the
previous reading is replaced, not layered onto.

Extraction always reads every channel and every run in your cache — there is no
way to narrow it, because what gets committed should not depend on what someone
happened to ask about. It does rewrite only the runs it read: a run whose
captured text is not in your cache — a teammate's, cloned but not yet unpacked
— keeps the reading it arrived with. Extraction never mistakes "I did not look
at it" for "it has nothing to say".

### Index and measure are not the same column

`epoch 0 | loss: 0.5000` holds two different things. `loss` is a
**measurement** — it fills a cell. `epoch` is an **index** — it says *which
row* that cell belongs to. Extraction keeps them apart, sending `epoch` to the
loop context and `loss` to the value.

That is why the script above did not need `flor.loop`. Drop the naming:

```python
for epoch in range(3):                       # no flor.loop
    print(f"epoch {epoch} | loss: {1.0 / (epoch + 2):.4f}")
```

and `--extract` still recovers the index from the text:

```
  projid                     tstamp  filename   source  epoch    loss
0  zero2 2026-08-13 12:35:02.118307  train.py  extract      0     0.5
1  zero2 2026-08-13 12:35:02.118307  train.py  extract      1  0.3333
2  zero2 2026-08-13 12:35:02.118307  train.py  extract      2    0.25
```

Recording `epoch` as a metric instead would make it a peer of `loss`, and the
two would have nothing to join on — three epochs and three losses would come
back as nine rows. Naming the loop is still better, because it indexes
checkpoints and lets [replay](replay.md) jump to an iteration; extraction only
gets you the table.

A name is read as an index when it is one of `epoch`, `step`, `iteration`,
`iter`, or `batch` and it leads the line. The vocabulary is fixed on purpose:
an open-ended rule would read `Loading 5 files` as a loop named `Loading`. When
a loop of that name is already in context, the recorded one wins.

### Where the extracted rows live

Beside the run they were read from, in `.flor/extracted/<tstamp>.jsonl` —
tracked, like `runs/`, so extraction travels with the run:

```
.flor/
  runs/<tstamp>.jsonl       what the run observed
  extracted/<tstamp>.jsonl  what was read out of it
```

They are deliberately not *in* `runs/<tstamp>.jsonl`. That file is the
immutable record of what the run observed, and a reading of text is not an
observation — it is a guess, and `--extract` has to be able to replace it.
Keeping the two in separate files is what lets the guess be revised without
ever rewriting the observation.

`--extract` commits the result for you, so a teammate needs nothing but the
clone:

```bash
git clone … && git checkout flor.branch && python -m flordb unpack
```

```
   projid                     tstamp  filename   source  epoch    loss
0  origin 2026-08-13 12:35:02.118307  train.py  extract      0     0.5
1  origin 2026-08-13 12:35:02.118307  train.py  extract      1  0.3333
2  origin 2026-08-13 12:35:02.118307  train.py  extract      2    0.25
```

They ran no extraction command. They see the reading you saw — not what their
version of the rule would have guessed — because the values, not the intent,
are what got committed.

Two things follow from that commit. It is made only on a shadow branch, so
running `--extract` from `main` saves the file and tells you it did not commit
rather than putting one on the branch you review. And it stages
`.flor/extracted/` alone — unlike the per-run auto-commit, whose `git add -A`
is deliberate — so an edit you were mid-way through is not swept in.

The rows also land in the sqlite cache, tagged `source='extract'` — the same
column that separates a forward run from a [replay](replay.md), so a derived
guess never passes for a value your script logged. That half is disposable:
`flor unpack` rebuilds it from the tracked files, and deleting the `.db` costs
you nothing.

## Tuning and switching off

```python
flor.set_capture(max_records=50_000)   # per-run ceiling (default 10,000)
flor.set_capture(False)                # or FLOR_CAPTURE=0 in the environment
```
