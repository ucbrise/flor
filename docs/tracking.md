# Experiment Tracking with the Flor API

Declare the inputs, metrics, and iterations you want to query across runs.
Each call adds structure to the run's records:

| Call | Records |
|---|---|
| `flor.arg("lr", 1e-3)` | An input with a default that can be overridden from the command line |
| `flor.log("loss", loss)` | A named value at the current point in the run |
| `flor.loop("epoch", range(3))` | Iteration context for records produced inside the loop |

Run your script in a Git repository so FlorDB can version it. See
[Working on a FlorDB branch](branches.md) for where the commits go.

## Declare inputs and record metrics

Save this as `train.py`:

```python
import flordb as flor

lr = flor.arg("lr", 1e-3)
epochs = flor.arg("epochs", 3)

for epoch in flor.loop("epoch", range(epochs)):
    loss = 1.0 / (epoch + 2)  # example metric
    flor.log("loss", loss)
```

`flor.arg` returns the chosen value and records it with the run. Use the defaults
with `python train.py`, or override them from the command line:

```bash
python train.py --kwargs lr=5e-4 epochs=5
```

After the run, query the declared inputs and logged metrics in a notebook or
Python session:

```python
import flordb as flor

flor.dataframe("lr", "epochs", "loss")
```

The result includes an `epoch` column and one loss row per iteration, with the
run's `lr` and `epochs` attached. Values recorded with `flor.log` are available
directly in `flor.dataframe()`.

## Name your loops

Wrap an iterable with `flor.loop` to name its iteration context. It yields the
iterable's values, while FlorDB records iteration positions starting at zero.
Each named loop becomes a column in query results.

```python
for epoch in flor.loop("epoch", range(epochs)):
    for x, y in flor.loop("step", trainloader):
        ...
        flor.log("loss", loss.item())
    flor.log("val_acc", validate(net))
```

Here, `loss` carries both `epoch` and `step`; `val_acc` carries only `epoch`
because it is logged after the inner loop. Run-level inputs recorded before
the loops are available alongside either metric.

Named iterations also let you select which iterations record recovered metrics
during [replay](replay.md#path-2-push-the-code-with-hindsight-logging).
For saved model files, see [Checkpoints](checkpoints.md).

## Combine named loops with captured output

[Automatic log capture](capture.md) works with ordinary Python loops. Once you
add `flor.loop`, captured lines also carry the enclosing iteration context:

```python
for epoch in flor.loop("epoch", range(3)):
    print(f"loss: {1.0 / (epoch + 2):.4f}")
```

`flor.io()` now includes an `epoch` column for these lines, even though their
text contains no epoch number. You can keep existing `print` and `logging`
calls while adding named metrics wherever useful.

Captured text stays queryable through `flor.io()`. To derive metric columns
from it after a run, use [metric extraction](capture.md#extracting-metrics-from-captured-text).
