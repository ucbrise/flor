# FlorDB: Log-First Context Management for ML Practitioners

[![PyPI](https://img.shields.io/pypi/v/flordb.svg?nocache=1)](https://pypi.org/project/flordb/)

FlorDB brings experiment tracking, provenance, and reproducibility to your ML workflow—using the one thing every engineer already writes: **logs**.

Unlike heavyweight MLOps platforms, FlorDB doesn’t ask you to adopt a new UI, schema, or service. Just import it, log as you normally would, and gain full history, lineage, and replay capabilities across your training runs.

## 🌻 Why FlorDB?

- **Zero Code Changes to Start**  
  Already using `print` or `logging`? Import FlorDB and your existing output is captured, versioned, and queryable—no rewrite required.

- **Log-Driven Experiment Tracking**  
  No dashboards to configure or schemas to design. `flor.log(...)` writes structured metadata; `flor.arg(...)` turns a constant into a CLI-settable hyperparameter that is recorded with the run.

- **Hindsight Logging & Replay**  
  Missed a metric? Add a log *after the fact* and replay past runs to capture it—no rerunning from scratch.

- **Reproducibility Without Friction**  
  Every run is versioned via Git, every hyperparameter is recorded, and PyTorch checkpoints are captured and addressable by loop iteration—automatically.

- **Works With Your Stack**  
  Makefiles, Airflow, Slurm, HuggingFace, PyTorch—you don’t change your workflow. FlorDB fits in.

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

## 📝 First Log in 30 Seconds

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

## 🪵 Already Using `print` and `logging`? Just Import

Add one import to a script you already have. Nothing else changes:

```python
import flordb as flor          # <-- the only new line

for epoch in range(3):
    print(f"epoch {epoch} | loss: {1.0 / (epoch + 2):.4f}")
```

Your terminal looks exactly the same. But the run is now versioned, committed,
and queryable with `flor.io()`.

→ [Automatic log capture](docs/capture.md): channels, naming your loops, and
turning captured text into real metric columns.

## 🧪 Track Experiments with Zero Overhead

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

    torch.save({"model": net.state_dict()}, "ckpt.pth")   # mirrored to flor (rate-limited)
```

Change hyperparameters from the CLI:

```bash
python train.py --kwargs lr=5e-4 batch_size=64
```

View metrics across runs:

```python
flor.dataframe("lr", "batch_size", "val_acc")
```

```
        projid                     tstamp  filename   source  epoch epoch_value      lr batch_size val_acc
0  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      0           0  0.0005         64      90
1  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      1           1  0.0005         64      91
2  ml_tutorial 2026-08-13 11:27:06.417615  train.py  forward      2           2  0.0005         64      92
```

Every `flor.loop` you name becomes a column, so nested metrics land at the right
grain without a join table. Raw SQL is available too, via `flor.query(...)`.

→ [Checkpoints](docs/checkpoints.md): what gets mirrored, enrolling objects
explicitly, and bounding disk use.

## 🔍 Hindsight Logging: Fix It After You See It

Forgot to log gradient norms? Add the statement to the script now:

```python
flor.log("grad_norm", ...)
```

```bash
python -m flordb replay --apply grad_norm
```

FlorDB walks the historical versions, splices your new statement into each one,
restarts from the nearest checkpoint, and records the recovered values.

→ [Replay](docs/replay.md): narrowing the work by iteration, replaying a single
run, and `--override`.

## 📁 What FlorDB Writes

Everything is project-local; nothing lands in your home directory. Run history
is tracked in git alongside the code that produced it, so a teammate runs
`git fetch && python -m flordb unpack` and has everyone's metrics—no server, no
bucket, no bill.

→ [Storage](docs/storage.md): the `.flor/` layout, and what syncs vs. what's
rebuilt on demand.

## 🏗 Real ML Systems Built on FlorDB

FlorDB powers full AI/ML lifecycle tooling—feature stores, model registries,
document parsing with feedback loops, and continuous training pipelines. See
[Scan Studio](https://github.com/bwerick/scan_studio) and
[Document Parser](https://github.com/rlnsanz/document_parser) for real-world
integrations.

## 📚 Publications

FlorDB is based on research from UC Berkeley’s RISE Lab and Arizona State University.

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

**GitHub:** https://github.com/ucbrise/flor  
**Tutorial Video:** https://youtu.be/mKENSkk3S4Y
