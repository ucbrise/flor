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
on record. Extraction always reads every channel and every run in your cache. 

### Where the extracted rows live

In the sqlite cache, and nowhere else. Nothing is written beside the run and
nothing is committed:

```
.flor/
  runs/<tstamp>.jsonl  tracked: the captured text, and everything else observed
  <projid>.db          ignored: the reading, alongside the rest of the cache
```

The rows carry `source='extract'` — the same column that separates a forward
run from a [replay](replay.md), so a derived guess never passes for a value
your script logged.

Keeping the reading out of git is deliberate. The text it was read from is
already committed and the rule that reads it is in FlorDB, so a stored copy
would be a second version of something git already carries — one that can
drift from the text it claims to summarize. What you give up is that
extraction does not travel. A teammate clones, unpacks, and has every captured
line but no `loss` column until they run the command themselves:

```bash
git clone … && git checkout flor.branch && python -m flordb unpack
python -m flordb capture --extract
```

```
   projid                     tstamp  filename   source  epoch    loss
0  origin 2026-08-13 12:35:02.118307  train.py  extract      0     0.5
1  origin 2026-08-13 12:35:02.118307  train.py  extract      1  0.3333
2  origin 2026-08-13 12:35:02.118307  train.py  extract      2    0.25
```

Same command, same rule, same io — so the same table. For the same reason,
deleting the `.db` drops the extracted columns until you re-run `--extract`;
the runs themselves come back from `flor unpack` untouched.

## Tuning and switching off

```python
flor.set_capture(max_records=50_000)   # per-run ceiling (default 10,000)
flor.set_capture(False)                # or FLOR_CAPTURE=0 in the environment
```
