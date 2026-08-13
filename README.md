# FlorDB: Log-First Context Management for ML Practitioners

[![PyPI](https://img.shields.io/pypi/v/flordb.svg?nocache=1)](https://pypi.org/project/flordb/)


FlorDB brings experiment tracking, provenance, and reproducibility to your ML workflow—using the one thing every engineer already writes: **logs**.

Unlike heavyweight MLOps platforms, FlorDB doesn’t ask you to adopt a new UI, schema, or service. Just import it, log as you normally would, and gain full history, lineage, and replay capabilities across your training runs.

## 🚀 Why FlorDB?

- **Log-Driven Experiment Tracking**  
  No dashboards to configure or schemas to design. `flor.log(...)` writes structured, queryable metadata; `flor.arg(...)` turns a constant into a CLI-settable hyperparameter that is recorded with the run.

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

---

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

## 🧪 Track Experiments with Zero Overhead

Drop FlorDB into your existing training script:

```python
import flordb as flor

# Hyperparameters
lr = flor.arg("lr", 1e-3)
batch_size = flor.arg("batch_size", 32)

for epoch in flor.loop("epoch", range(epochs)):
    for x, y in flor.loop("step", trainloader):
        ...
        flor.log("loss", loss.item())
    flor.log("val_acc", validate(net))

    # Already in your script? Then you're done: flor mirrors this save into
    # its object store, one snapshot per epoch, and replays from it later.
    torch.save({"model": net.state_dict()}, "ckpt.pth")
```

No `with flor.checkpointing(...):` block is required — an existing `torch.save`
inside a `flor.loop` is enough. Use `flor.checkpointing(model=..., optimizer=...)`
to enroll non-torch objects (scikit-learn estimators, plain dicts), and
`flor.set_ckpt_interval(seconds)` to bound how much disk the snapshots take
(default: at most one every 60s).

**Change hyperparameters from the CLI:**

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
grain without a join table. `source` distinguishes values observed on the
original run (`forward`) from ones recovered later by replay (`replay`).
Raw SQL is available too:

```python
flor.query("SELECT * FROM logs WHERE value_name = 'val_acc' AND source = 'forward'")
```

## 🔍 Hindsight Logging: Fix It After You See It

Forgot to log gradient norms?

```python
flor.log("grad_norm", ...)
```

Add the logging statement to the script and run:

```bash
python -m flordb replay --apply grad_norm
```

FlorDB estimates the cost, asks for confirmation, then walks the historical
versions: it checks each run's commit out, splices your new statement into that
version of the script, restarts from the nearest checkpoint, and records the
recovered values. Narrow the work when you don't need every iteration:

```bash
python -m flordb replay --apply grad_norm --iter epoch=0,2 --iter step=all
```

Loops you don't mention default to their last iteration. To replay one run
directly (no orchestration, no git checkout), the same verbs are flags on the
script itself:

```bash
python train.py --replay_flor \
    --apply loss,val_acc \
    --iter epoch=0,2 \
    --iter step=all \
    --override device=cpu
```

`--override` exists for environment-shaped settings such as `device` — replaying
on a different machine is fine, but hyperparameters that defined the original
run are rejected, since changing those makes it a new experiment rather than a
replay.

## 📁 What FlorDB Writes

Everything is project-local; nothing lands in your home directory.

```
.flor/
  runs/<tstamp>.jsonl     one immutable record per forward run
  obj_store/<tstamp>/     checkpoints, addressable by loop iteration
  <projid>.db             sqlite query cache, rebuildable at any time
.flor.cmd                 tracked: the run's tstamp and command line
```

`.flor/` is added to `.gitignore` on first run, and each run makes one
`FLOR::Auto-commit::<tstamp>` commit on a shadow branch whose message body
carries the run's hyperparameters — so a run stays reproducible even if the
logs are gone. Lost or moved the cache? Rebuild it from the JSONL:

```bash
python -m flordb unpack
```

## 🏗 Real ML Systems Built on FlorDB

FlorDB powers full AI/ML lifecycle tooling:

- **Feature Stores & Model Registries**
- **Document Parsing & Feedback Loops**
- **Continuous Training Pipelines**

See our [Scan Studio](https://github.com/bwerick/scan_studio) and [Document Parser](https://github.com/rlnsanz/document_parser) examples for real-world integration.


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
