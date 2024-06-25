FlorDB
================================
[![PyPI](https://img.shields.io/pypi/v/flordb.svg?nocache=1)](https://pypi.org/project/flordb/)



Flor (for "fast low-overhead recovery") is a record-replay system for deep learning, and other forms of machine learning that train models on GPUs. Flor was developed to speed-up hindsight logging: a cyclic-debugging practice that involves adding logging statements *after* encountering a surprise, and efficiently re-training with more logging. Flor takes low-overhead checkpoints during training, or the record phase, and uses those checkpoints for replay speedups based on memoization and parallelism.

FlorDB integrates Flor, `git` and `sqlite3` to manage model developer's logs, execution data, versions of code, and training checkpoints. In addition to serving as an experiment management solution for ML Engineers, FlorDB extends hindsight logging across model trainging versions for the retroactive evaluation of iterative ML. FlorDB has been extended to support Dataflow operations.

Flor and its evolutions are software developed at UC Berkeley's [RISE](https://rise.cs.berkeley.edu/) Lab.

[![FlorDB Demo](https://img.youtube.com/vi/x4ObDb5B2Us/0.jpg)](https://youtu.be/x4ObDb5B2Us)

You can follow along yourself by starting a Jupyter server from this directory and opening [`tutorial.ipynb`](tutorial.ipynb).

# Installation

```bash
pip install flordb
```

# Getting Started

We start by selecting (or creating) a `git` repository to save our model training code as we iterate and experiment. Flor automatically commits your changes on every run, so no change is lost. Below we provide a sample repository you can use to follow along:

```bash
$ git clone git@github.com:ucbepic/ml_tutorial
$ cd ml_tutorial/
```

Run the `train.py` script to train a small linear model, 
and test your `florflow` installation.

```bash
$ python train.py
```

Flor will manage checkpoints, logs, command-line arguments, code changes, and other experiment metadata on each run (More details [below](#storage--data-layout)). All of this data is then expesed to the user via SQL or Pandas queries.

# View your experiment history
From the same directory you ran the examples above, open an iPython terminal, then load and pivot the log records.

```bash
$ python -m flor dataframe

        projid               tstamp  filename device seed hidden epochs batch_size     lr print_every accuracy correct
0  ml_tutorial  2023-08-28T15:04:07  train.py    cpu   78    500      5         32  0.001         500    97.71    9771
1  ml_tutorial  2023-08-28T15:04:35  train.py    cpu    8    500      5         32  0.001         500    98.01    9801
```

# Run some more experiments

The `train.py` script has been prepared in advance to define and manage four different hyper-parameters:

```bash
$ cat train.py | grep flor.arg
hidden_size = flor.arg("hidden", default=500)
num_epochs = flor.arg("epochs", 5)
batch_size = flor.arg("batch_size", 32)
learning_rate = flor.arg("lr", 1e-3)
```

You can control any of the hyper-parameters (e.g. `hidden`) using Flor's command-line interface:
```bash 
$ python train.py --kwargs hidden=75
```


# Application Programming Interface (API)

Flor is shipped with utilities for serializing and checkpointing PyTorch state,
and utilities for resuming, auto-parallelizing, and memoizing executions from checkpoint.

The model developer passes objects for checkpointing to `flor.checkpointing(**kwargs)`,
and gives it control over loop iterators by 
calling `flor.loop(name, iterator)` as follows:

```python
import flor
import torch

hidden_size = flor.arg("hidden", default=500)
num_epochs = flor.arg("epochs", 5)
batch_size = flor.arg("batch_size", 32)
learning_rate = flor.arg("lr", 1e-3)

trainloader: torch.utils.data.DataLoader
testloader:  torch.utils.data.DataLoader
optimizer:   torch.optim.Optimizer
net:         torch.nn.Module
criterion:   torch.nn._Loss

with flor.checkpointing(model=net, optimizer=optimizer):
    for epoch in flor.loop("epoch", range(num_epochs)):
        for data in flor.loop("step", trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            flor.log("loss", loss.item())
            optimizer.step()
        eval(net, testloader)
```
As shown, 
we wrap both the nested training loop and main loop with `flor.loop` so Flor can manage their state. Flor will use loop iteration boundaries to store selected checkpoints adaptively, and on replay time use those same checkpoints to resume training from the appropriate epoch.  

### Logging API

You call `flor.log(name, value)` and `flor.arg(name, default=None)` to log metrics and register tune-able hyper-parameters, respectively. 

```bash
$ cat train.py | grep flor.arg
hidden_size = flor.arg("hidden", default=500)
num_epochs = flor.arg("epochs", 5)
batch_size = flor.arg("batch_size", 32)
learning_rate = flor.arg("lr", 1e-3)

$ cat train.py | grep flor.log
        flor.log("loss", loss.item()),
```

The `name`(s) you use for the variables you intercept with `flor.log` and `flor.arg` will become a column (measure) in the full pivoted view (see [Viewing your exp history](#view-your-experiment-history)).


## Publications

To cite this work, please refer to the [Multiversion Hindsight Logging](https://arxiv.org/abs/2310.07898) paper (pre-print '23).

FlorDB is open source software developed at UC Berkeley. 
[Joe Hellerstein](https://dsf.berkeley.edu/jmh/) (databases), [Joey Gonzalez](http://people.eecs.berkeley.edu/~jegonzal/) (machine learning), and [Koushik Sen](https://people.eecs.berkeley.edu/~ksen) (programming languages) 
are the primary faculty members leading this work.

This work is released as part of [Rolando Garcia](https://rlnsanz.github.io/)'s doctoral dissertation at UC Berkeley,
and has been the subject of study by Eric Liu and Anusha Dandamudi, 
both of whom completed their master's theses on FLOR.
Our list of publications are reproduced below.
Finally, we thank [Vikram Sreekanti](https://www.vikrams.io/), [Dan Crankshaw](https://dancrankshaw.com/), and [Neeraja Yadwadkar](https://cs.stanford.edu/~neeraja/) for guidance, comments, and advice.
[Bobby Yan](https://bobbyy.org/) was instrumental in the development of FLOR and its corresponding experimental evaluation.

* [Multiversion Hindsight Logging for Continuous Training](https://arxiv.org/abs/2310.07898). _R Garcia, A Dandamudi, G Matute, L Wan, JE Gonzalez, JM Hellerstein, K Sen_. pre-print on ArXiv, 2023.
* [Hindsight Logging for Model Training](http://www.vldb.org/pvldb/vol14/p682-garcia.pdf). _R Garcia, E Liu, V Sreekanti, B Yan, A Dandamudi, JE Gonzalez, JM Hellerstein, K Sen_. The VLDB Journal, 2021.
* [Fast Low-Overhead Logging Extending Time](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-117.html). _A Dandamudi_. EECS Department, UC Berkeley Technical Report, 2021.
* [Low Overhead Materialization with FLOR](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2020/EECS-2020-79.html). _E Liu_. EECS Department, UC Berkeley Technical Report, 2020. 


## License
FlorDB is licensed under the [Apache v2 License](https://www.apache.org/licenses/LICENSE-2.0).
