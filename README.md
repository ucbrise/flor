# FlorDB: Log-Forward Metadata Management for AI/ML Training and Evaluation

[![PyPI](https://img.shields.io/pypi/v/flordb.svg?nocache=1)](https://pypi.org/project/flordb/)

FlorDB starts with the `print` and `logging` output of the scripts you already run as part of model training, and, over time, grows with your help into sustained experiment tracking, model evaluation, and some measure of reproducibility. 

### 🌻 Why FlorDB?

- **Start Tracking with One Line**  
  Add `import flordb as flor` to a Python script you run. FlorDB captures that run's `print` and `logging` output and ties it to the code that produced it. You can query these values with `flor.io()`.

- **Experiment Tracking with Logging Statements**  
  `flor.log(n, v)` records what a run produces: loss, accuracy, anything you'd print. `flor.arg(n, v)` declares and records arguments: learning rate, batch size, seed &mdash; all configurable from the command line. You can query these values with `flor.dataframe()`.

- **Evaluation: Pull the Model or Push the Logging**  
  Missed a metric? Load a past run's checkpoint in a notebook (or anywhere else) and compute it. Or, add a logging statement and replay past runs to record it.

- **Reproducibility, Replay with Recorded Inputs**  
  FlorDB versions runs in Git and replays them with their recorded hyper-parameters and seeds.

- **At Home in Your Workflow**  
  Keep using your existing tools for orchestration, experiment tracking, and interactive work: from Make, Airflow, and Slurm to Jupyter, VS Code, or a terminal. FlorDB works within each run.

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

## 🪵 Already using `print` or `logging`? Add one import

> *Requires a Git repository for automatic versioning.*

Add one import to the script you run. The rest of your code stays as it is:

```python
import flordb as flor          # <-- the only new line

for epoch in range(3):
    print(f"epoch {epoch} | loss: {1.0 / (epoch + 2):.4f}")
```

Your output prints as before. When the run ends, FlorDB commits it and says so; the captured lines are queryable with `flor.io()`.

→ [Automatic log capture](docs/capture.md): channels and turning captured text into real metric columns.

## 🌿 FlorDB auto-commits to its designated git branch

Run from `main` and FlorDB creates and switches to `flor.branch` (or a numbered
variant), keeping auto-commits off your working branches. You stay on that flor branch after the run: subsequent runs accumulate history.

Prefer to name the branch yourself? Create it with a `flor.` prefix, such as `flor.experiment`, and FlorDB commits there instead of creating one. Exploring several leads? Give each its own `flor.` branch.


→ [Working on Flor Branches](docs/branches.md): saving changes, pushing your
branch, and bringing code back for review.

## 🧪 Track Experiments with the Flor API

Your runs already have captured output and a place in Git history. Add named
arguments and metrics when you want to compare experiments with `flor.dataframe()`.
You can keep your existing `print` and `logging` calls alongside them.
<!-- TODO: The overhead of this is something Akshit can evaluate. -->

In a training script, use `flor.arg(n, v)` for hyper-parameters and `flor.log(n, v)` for
metrics. Wrap your loops with `flor.loop` so each metric carries its epoch
or training step:

```python
import flordb as flor

lr = flor.arg("lr", default=1e-3)
batch_size = flor.arg("batch_size", 32)

for epoch in flor.loop("epoch", range(epochs)):
    for x, y in flor.loop("step", trainloader):
        ...
        flor.log("loss", loss.item())
    flor.log("val_acc", validate(net))

    torch.save({"model": net.state_dict()}, "ckpt.pth")
```

Here, `loss` is recorded at each step and `val_acc` after each epoch. 
`flor.arg` records and returns the learning rate and batch size before
the loops begin. They use the supplied defaults unless you override them
on the command line:

```bash
python train.py --kwargs lr=5e-4 batch_size=64
```

After running `train.py`, query the recorded arguments and metrics in Jupyter
or a Python session. For example, inspect the first few loss records:

```python
flor.dataframe("lr", "batch_size", "loss").head(3)
```

```
        projid                     tstamp  filename   source  epoch  step      lr batch_size    loss
0  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      0     0  0.0005         64     0.5
1  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      0     1  0.0005         64  0.3333
2  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      1     0  0.0005         64  0.3333
```

Using `flor` to set hyper-parameters, name loops, and log the metrics inside them, you give each *training* record its context. Use `flor.dataframe()` to query those values together and compare them across runs.

→ [Experiment tracking](docs/tracking.md): setting arguments, recording metrics,
and naming your loops with the Flor API.

For the `torch.save` call above, FlorDB keeps a local, gitignored copy of the
file for each run. Each save updates that run's copy, leaving its last
checkpoint available for evaluation. The filename is your choice (`ckpt.pth` is just an example). You decide whether to track the original file in Git.

→ [Checkpoints](docs/checkpoints.md): what gets copied, and how replay treats
your checkpoint file.

## 🔍 Evaluate Past Runs

New questions come up after training: a metric you forgot to log, or a bias that only surfaced in production. 
Load a past run's checkpoint in Jupyter and
evaluate the model. Or, add the `flor.log` statement to your script and replay past runs to record it.

### Pull the model into Jupyter

Load saved models in a notebook to evaluate new metrics and compare past runs.
<!-- TODO: Maybe we polish this API -->
→ [Jupyter walkthrough](docs/replay.md#path-1-pull-the-model-in-jupyter) and
[comparison notebook](notebooks/compare_models.ipynb).

### Replay with hindsight logging

Forgot to log gradient norms? Add the statement to the script now:

```python
flor.log("grad_norm", ...)
```

```bash
python -m flordb replay --apply grad_norm
```

FlorDB walks the historical versions, splices your new statement into each one,
re-executes it from the start, and records the recovered values.

→ [Replay](docs/replay.md): choosing which runs to log, replaying one run
on a different device, and replaying from a fresh clone.

## 📁 What FlorDB Writes

FlorDB stores run records, checkpoints, and its query cache in a `.flor/`
directory at the root of your Git repository.
Run records are tracked in Git; checkpoints and the cache stay local.
FlorDB never pushes; you choose what and when to share.

→ [Storage](docs/storage.md): the file layout, what syncs, and how to rebuild
the query cache after checkout.

<!-- ## 🏗 Real ML Systems Built on FlorDB

FlorDB powers full AI/ML lifecycle tooling—feature stores, model registries,
document parsing with feedback loops, and continuous training pipelines. See
[Scan Studio](https://github.com/bwerick/scan_studio) and
[Document Parser](https://github.com/rlnsanz/document_parser) for real-world
integrations. -->

## 📚 Publications

FlorDB is based on research from UC Berkeley’s [RISE Lab](https://rise.cs.berkeley.edu) continued at Arizona State University.

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

**Email:** rogarcia@berkeley.edu (or) rolando.garcia@asu.edu  
<!-- **Tutorial Video:** https://youtu.be/mKENSkk3S4Y -->
