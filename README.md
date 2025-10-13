# FlorDB: Log-First Context Management for ML Practitioners

FlorDB brings experiment tracking, provenance, and reproducibility to your ML workflow—using the one thing every engineer already writes: **logs**.

Unlike heavyweight MLOps platforms, FlorDB doesn’t ask you to adopt a new UI, schema, or service. Just import it, log as you normally would, and gain full history, lineage, and replay capabilities across your training runs.

---

## 🚀 Why FlorDB?

- **Log-Driven Experiment Tracking**  
  No dashboards to configure or schemas to design. FlorDB turns your existing `print()` or `log()` calls into structured, queryable metadata.

- **Hindsight Logging & Replay**  
  Missed a metric? Add a log *after the fact* and replay past runs to capture it—no rerunning from scratch.

- **Reproducibility Without Friction**  
  Every run is versioned via Git, every hyperparameter is recorded, and every model checkpoint is stored—automatically.

- **Works With Your Stack**  
  Makefiles, Airflow, Slurm, HuggingFace, PyTorch—you don’t change your workflow. FlorDB fits in.

---

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

Retrieve logs anytime:

```python
flor.dataframe("message")
```

---

## 🧪 Track Experiments with Zero Overhead

Drop FlorDB into your existing training script:

```python
import flordb as flor

# Hyperparameters
lr = flor.arg("lr", 1e-3)
batch_size = flor.arg("batch_size", 32)

with flor.checkpointing(model=net, optimizer=optimizer):
    for epoch in flor.loop("epoch", range(epochs)):
        for x, y in flor.loop("step", trainloader):
            ...
            flor.log("loss", loss.item())
```

**Change hyperparameters from the CLI:**

```bash
python train.py --kwargs lr=5e-4 batch_size=64
```

View metrics across runs:

```python
flor.dataframe("lr", "batch_size", "loss")
```

---

## 🔍 Hindsight Logging: Fix It After You See It

Forgot to log gradient norms?

```python
flor.log("grad_norm", ...)
```

Replay past runs—no retraining required:

```bash
python -m flordb replay grad_norm
```

FlorDB replays only what’s needed, injecting the new log and committing results.

---

## 🏗 Real ML Systems Built on FlorDB

FlorDB powers full AI/ML lifecycle tooling:

- **Feature Stores & Model Registries**
- **Document Parsing & Feedback Loops**
- **Continuous Training Pipelines**

See our Document Parser example for real-world integration.

---

## 📚 Publications

FlorDB is based on research from UC Berkeley’s RISE Lab and Arizona State University.

- *Flow with FlorDB: Incremental Context Maintenance for the Machine Learning Lifecycle* (CIDR 2025)  
- *Hindsight Logging for Model Training* (VLDBJ 2021)  
- *The Management of Context in the ML Lifecycle* (UCB Tech Report 2024)  

Full reference list in the repository.

---

## 🛠 License

Apache 2.0 — free to use, modify, and distribute.

---

## 💡 Get Involved

FlorDB is actively developed. Contributions, issues, and real-world use cases are welcome!

**GitHub:** https://github.com/ucbrise/flor  
**Tutorial Video:** https://youtu.be/mKENSkk3S4Y
