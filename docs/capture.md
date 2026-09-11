# Automatic Log Capture

FlorDB captures what a run prints and logs from the moment `flordb` is imported.
Add `import flordb as flor` to the script you run: your output prints as
before, and when the run ends FlorDB commits the captured lines with the code
that produced them.

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

The `channel` column identifies where a line came from:

| Channel | Output |
|---|---|
| `io::stdout` | Standard output, including `print` |
| `io::stderr` | Standard error |
| `io::log::<level>` | Python logging records, such as `io::log::info` |

Filter captured output by channel:

```python
flor.io("io::stdout")
```

## Extracting metrics from captured text

Captured text is not included in `flor.dataframe()` because it is not a metric.
FlorDB can extract metrics from captured text after a run has finished, so you
do not need to enable anything or train again. Preview the results against your
own logs first. In the example above, extraction reads the `epoch` index
and `loss` value from the printed text; the script uses an ordinary Python loop:

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

Query the extracted metric:

```python
flor.dataframe("loss")
```

```
  projid                     tstamp  filename   source  epoch    loss
0  zero2 2026-08-13 12:34:47.951515  train.py  extract      0     0.5
1  zero2 2026-08-13 12:34:47.951515  train.py  extract      1  0.3333
2  zero2 2026-08-13 12:34:47.951515  train.py  extract      2    0.25
```

`loss` is a real column in `flor.dataframe("loss")` now, and the raw line stays
on record.

## Tuning and switching off

```python
flor.set_capture(max_records=50_000)   # per-run ceiling (default 10,000)
flor.set_capture(False)                # or FLOR_CAPTURE=0 in the environment
```

## Next: declare experiment data

To record named values and iteration context in your code, see
[Experiment tracking with the Flor API](tracking.md). That guide covers
inputs, metrics, named loops, and how loop context also enriches captured output.
