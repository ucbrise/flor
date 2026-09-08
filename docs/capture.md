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
can pull metrics out of it if you ask. See what that would do to *your* logs
before turning it on:

```bash
python -m flordb capture --preview
```

```
Would extract 3 value(s) across 1 metric(s) from 6 captured line(s): loss

  loss                 0.5            <- epoch 0 | loss: 0.5000
  loss                 0.3333         <- epoch 1 | loss: 0.3333
  loss                 0.25           <- epoch 2 | loss: 0.2500

Nothing was written. Enable with flor.set_capture(extract=True) in your script.
```

With `flor.set_capture(extract=True)`, `loss` becomes a real column in
`flor.dataframe("loss")` while the raw line stays on record. It is off by
default because a bad guess would invent a column you didn't ask for.

## Tuning and switching off

```python
flor.set_capture(max_records=50_000)   # per-run ceiling (default 10,000)
flor.set_capture(False)                # or FLOR_CAPTURE=0 in the environment
```
