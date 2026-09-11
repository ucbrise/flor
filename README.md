# FlorDB: Log-Forward Metadata Management for AI/ML Training and Evaluation

[![PyPI](https://img.shields.io/pypi/v/flordb.svg?nocache=1)](https://pypi.org/project/flordb/)

FlorDB starts with the `print` and `logging` output of the scripts you already run as part of model training, and, over time, grows with you into sustained experiment tracking, model evaluation, and some measure of reproducibility. No new schema or service to adopt. 

## 🌻 Why FlorDB?

- **Starting from an Existing Project**  
  Add `import flordb as flor` to a `.py` script you run. FlorDB captures the run's `print` and `logging` output with each run tied to the code that produced it. You can query these values with `flor.io()`.

- **Experiment Tracking with Logging Statements**  
  `flor.log(n, v)` records what a run produces: loss, accuracy, anything you'd print. `flor.arg(n, v)` records what it consumes: learning rate, batch size, random seed, each settable from the command line. You can query these values with `flor.dataframe()`.

- **Evaluation: Pull the Model or Push the Code**  
  Missed a metric? Load a past run's checkpoint in a notebook and measure it, or add the log statement and replay past runs to retrieve it.

- **Reproducibility Without Friction**  
  Every run is versioned via Git, replays reuse the forward run's hyperparameters and seed, and one checkpoint per run is mirrored automatically.

- **Keep Run History With Your Project**  
  Your run history stays local, alongside your code. FlorDB keeps a record of your experiments as you work, with no server to set up or maintain.

Keep using the tools you already work with: Make, Airflow, Slurm, Jupyter, VSCode, or a plain terminal.




## 📦 Installation

```bash
pip install flordb
```

For contributors or bleeding-edge features:

```bash
git clone https://github.com/ucbrise/flor.git
cd flor
pip install -e .
```

## 🪵 Already using `print` and `logging`? Add one import

> *Requires a Git repository for automatic versioning.*

Add one import to the script you run. The rest of your code stays as it is:

```python
import flordb as flor          # <-- the only new line

for epoch in range(3):
    print(f"epoch {epoch} | loss: {1.0 / (epoch + 2):.4f}")
```

Your output prints as before. When the run ends, FlorDB commits it and says so; the captured lines are queryable with `flor.io()`.

→ [Automatic log capture](docs/capture.md): channels and turning captured text into real metric columns.

### FlorDB commits to its own git branch

Run from `main` and FlorDB creates and switches to `flor.branch` (or a numbered
variant), keeping auto-commits off your working branches. You stay on that flor branch after the run: subsequent runs accumulate history.

Prefer to name the branch yourself? Create it with a `flor.` prefix, such as
`flor.experiment`, and FlorDB commits there instead of creating one. Exploring
several leads? Give each its own `flor.` branch.


→ [Working on Flor Branches](docs/branches.md): saving changes, pushing your
branch, and bringing code back for review.

## 🧪 Track Experiments with the Flor API

Use `flor.arg` to declare inputs, `flor.log` to record named values, and
`flor.loop` to attach iteration context. Query these records with `flor.dataframe()`.

### First Log in 30 Seconds

> *Requires a Git repository for automatic versioning.*

```bash
mkdir flor_sandbox
cd flor_sandbox
git init
ipython
```

```python
import flordb as flor
flor.log("message", "Hello ML World!")
```
```
message: Hello ML World!

Run committed successfully.
```

Retrieve logs anytime:

```python
flor.dataframe("message")
```
```
         projid              tstamp filename   source          message
0  flor_sandbox 2025-10-13 18:13:48  ipython  forward  Hello ML World!
```

### Record hyperparameters and metrics by iteration

Adopt as much as you want. Every step buys a specific thing:

```python
import flordb as flor

lr = flor.arg("lr", 1e-3)                     # CLI-settable, recorded with the run
batch_size = flor.arg("batch_size", 32)

for epoch in flor.loop("epoch", range(epochs)):
    for x, y in flor.loop("step", trainloader):
        ...
        flor.log("loss", loss.item())
    flor.log("val_acc", validate(net))

    torch.save({"model": net.state_dict()}, "ckpt.pth")   # flor keeps each run's copy
```

Change hyperparameters from the CLI:

```bash
python train.py --kwargs lr=5e-4 batch_size=64
```

View metrics across runs:

```python
flor.dataframe("lr", "batch_size", "loss")
```

```
        projid                     tstamp  filename   source  epoch  step      lr batch_size    loss
0  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      0     0  0.0005         64     0.5
1  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      0     1  0.0005         64  0.3333
2  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      1     0  0.0005         64  0.3333
3  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      1     1  0.0005         64    0.25
4  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      2     0  0.0005         64    0.25
5  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      2     1  0.0005         64     0.2
```

Each named `flor.loop` becomes its own column, and a row carries the loops that enclosed the
`flor.log` that produced it. `loss` is logged inside `step`, so you get one row
per step, with its epoch and the run's hyperparameters attached—no JOIN needed. 

→ [Experiment tracking](docs/tracking.md): declaring inputs, recording metrics,
and naming your loops with the Flor API.

→ [Checkpoints](docs/checkpoints.md): what gets copied, and how replay treats
your checkpoint file.

To evaluate saved models in Jupyter, discover a run's checkpoints with
`flor.checkpoints(row.tstamp)` and load one with
`flor.load_checkpoint(row.tstamp, "ckpt.pth")`. See
[Pull the model in Jupyter](docs/replay.md#path-1-pull-the-model-in-jupyter) and
the [comparison notebook](notebooks/compare_models.ipynb).

## 🔍 Hindsight Logging, or Logging After the Fact

Forgot to log gradient norms? Add the statement to the script now:

```python
flor.log("grad_norm", ...)
```

```bash
python -m flordb replay --apply grad_norm
```

FlorDB walks the historical versions, splices your new statement into each one,
re-executes it from the start, and records the recovered values.

→ [Replay](docs/replay.md): narrowing the work by iteration, replaying a single
run, and `--override`.

## 📁 What FlorDB Writes

Everything is project-local; nothing lands in your home directory. FlorDB
auto-commits to your current branch if its name starts with `flor.`. Otherwise,
it creates and switches to `flor.branch` (or a numbered variant), keeping
auto-commits off `main` and your other working branches. Run history is versioned
alongside the code that produced it. A teammate runs
`git fetch && git checkout flor.branch && python -m flordb unpack` (substituting
your `flor.*` branch name) and has everyone's metrics.

→ [Storage](docs/storage.md): the `.flor/` layout, the shadow branch, and what
syncs vs. what's rebuilt on demand.

<!-- ## 🏗 Real ML Systems Built on FlorDB

FlorDB powers full AI/ML lifecycle tooling—feature stores, model registries,
document parsing with feedback loops, and continuous training pipelines. See
[Scan Studio](https://github.com/bwerick/scan_studio) and
[Document Parser](https://github.com/rlnsanz/document_parser) for real-world
integrations. -->

## 📚 Publications

FlorDB is based on research from UC Berkeley’s RISE Lab continued at Arizona State University.

- *Flow with FlorDB: Incremental Context Maintenance for the Machine Learning Lifecycle* ([CIDR 2025](https://vldb.org/cidrdb/papers/2025/p33-garcia.pdf))  
- *The Management of Context in the ML Lifecycle* ([UCB Tech Report 2024](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-142.html))  
- *Hindsight Logging for Model Training* ([PVLDB 2021](http://www.vldb.org/pvldb/vol14/p682-garcia.pdf))  

## 🛠 License

[Apache v2 License](https://www.apache.org/licenses/LICENSE-2.0) — free to use, modify, and distribute.

## 💡 Get Involved

FlorDB is actively developed. Contributions, issues, and real-world use cases are welcome!

```bash
make test        # full suite, including real forward runs and replays
make test-fast   # unit tests only (~1s)
```

**Email:** rolando.garcia@asu.edu  
**Tutorial Video:** https://youtu.be/mKENSkk3S4Y
