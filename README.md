FlorDB: Nimble Experiment Management for Iterative ML
================================
[![PyPI](https://img.shields.io/pypi/v/flordb)](https://pypi.org/project/flordb/)


Flor (for "fast low-overhead recovery") is a record-replay system for deep learning, and other forms of machine learning that train models on GPUs. Flor was developed to speed-up hindsight logging: a cyclic-debugging practice that involves adding logging statements *after* encountering a surprise, and efficiently re-training with more logging. Flor takes low-overhead checkpoints during training, or the record phase, and uses those checkpoints for replay speedups based on memoization and parallelism.

FlorDB integrates Flor, `git` and `sqlite3` to manage model developer's logs, execution data, versions of code, and training checkpoints. In addition to serving as an experiment management solution for ML Engineers, FlorDB extends hindsight logging across model trainging versions for the retroactive evaluation of iterative ML.

Flor and FlorDB are software developed at UC Berkeley's [RISE](https://rise.cs.berkeley.edu/) Lab.

<!-- [![Napa Retreat Demo](https://i.ytimg.com/vi/TNSt5-i7kR4/sddefault.jpg)](https://youtu.be/TNSt5-i7kR4) -->

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
and test your `flordb` installation.

```bash
$ python train.py --flor myFirstRun
```

Flor will manage checkpoints, logs, command-line arguments, code changes, and other experiment metadata on each run (More details [below](#storage--data-layout)). All of this data is then expesed to the user via SQL or Pandas queries.

# View your experiment history
From the same directory you ran the examples above, open an iPython terminal, then load and pivot the log records.

```bash
$ pwd
/Users/rogarcia/git/ml_tutorial

$ ipython
```

```ipython
In [1]: from flor import full_pivot, log_records
In [2]: full_pivot(log_records())
Out[2]: 
                            projid       runid               tstamp        vid  epoch  step      loss hidden batch_size epochs     lr
0   ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      1   100  0.246695    500         32      5  0.001
1   ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      1   200  0.279637    500         32      5  0.001
2   ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      1   300  0.247390    500         32      5  0.001
3   ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      1   400  0.536536    500         32      5  0.001
4   ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      1   500  0.198422    500         32      5  0.001
..                             ...         ...                  ...        ...    ...   ...       ...    ...        ...    ...    ...
85  ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      5  1400  0.003081    500         32      5  0.001
86  ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      5  1500  0.002184    500         32      5  0.001
87  ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      5  1600  0.042605    500         32      5  0.001
88  ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      5  1700  0.007986    500         32      5  0.001
89  ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      5  1800  0.006866    500         32      5  0.001

[90 rows x 11 columns]
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
$ python train.py --flor mySecondRun --hidden 75
```

### Advanced (Optional): Batch Processing
Alternatively, we can call `flor.batch()` from an interactive environment
inside our model training repository, to dispatch a group of jobs that can be long-runnning:
```bash
$ pwd
/Users/rogarcia/git/ml_tutorial

$ ipython
```

```ipython
In [1]: import flor

In [2]: flor.batch(flor.cross_prod(
    hidden=[i*100 for i in range(1,6)],
    lr=(1e-4, 1e-3)
    ))
Out[2]:
--hidden 100 --lr 0.0001 
--hidden 100 --lr 0.001 
--hidden 200 --lr 0.0001 
--hidden 200 --lr 0.001 
--hidden 300 --lr 0.0001 
--hidden 300 --lr 0.001 
--hidden 400 --lr 0.0001 
--hidden 400 --lr 0.001 
--hidden 500 --lr 0.0001 
--hidden 500 --lr 0.001 
```

Then, we start a `flordb` server to process the batch jobs:

```bash
$ python -m flor serve
```

or, if we want to allocate a GPU to the flor server:
```bash
$ python -m flor serve 0 
```
(where 0 is replaced by the GPU id).

You can check the progress of your jobs with the following query:

```bash
$ watch "sqlite3 ~/.flor/main.db -header 'select done, path, count(*) from jobs group by done, path;'"

done|path|count(*)
0|/Users/rogarcia/git/ml_tutorial|5
1|/Users/rogarcia/git/ml_tutorial|5
```

When finished, you can view the updated pivot view with all your experiment data:

```bash
$ pwd
/Users/rogarcia/git/ml_tutorial

$ ipython
```
```ipython
In [1]: from flor import full_pivot, log_records
In [2]: full_pivot(log_records())
Out[2]: 
                              projid       runid               tstamp        vid  epoch  step      loss batch_size hidden     lr epochs
0     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      1   100  0.246695         32    500  0.001      5
1     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      1   200  0.279637         32    500  0.001      5
2     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      1   300  0.247390         32    500  0.001      5
3     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      1   400  0.536536         32    500  0.001      5
4     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  c0418c...      1   500  0.198422         32    500  0.001      5
...                              ...         ...                  ...        ...    ...   ...       ...        ...    ...    ...    ...
1075  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T10:11:48  7b4dfc...      5  1400  0.012752         32    500  0.001      5
1076  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T10:11:48  7b4dfc...      5  1500  0.005932         32    500  0.001      5
1077  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T10:11:48  7b4dfc...      5  1600  0.058090         32    500  0.001      5
1078  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T10:11:48  7b4dfc...      5  1700  0.000570         32    500  0.001      5
1079  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T10:11:48  7b4dfc...      5  1800  0.043115         32    500  0.001      5

[1080 rows x 11 columns]
```



# Model Traing Kit (MTK)

The Model Training Kit (MTK) includes utilities for serializing and checkpointing PyTorch state,
and utilities for resuming, auto-parallelizing, and memoizing executions from checkpoint.

The model developer passes objects for checkpointing to `MTK.checkpoints(*args)`,
and gives it control over loop iterators by 
calling `MTK.loop(iterator)` as follows:

```python
import flor
from flor import MTK

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

MTK.checkpoints(net, optimizer)
for epoch in MTK.loop(range(num_epochs)):
    for data in MTK.loop(trainloader):
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
we wrap both the nested training loop and main loop with `MTK.loop` so Flor can manage their state. Flor will use loop iteration boundaries to store selected checkpoints adaptively, and on replay time use those same checkpoints to resume training from the appropriate epoch.  

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

# Storage & Data Layout
On each run, Flor will:
1. Save model checkpoints in `~/.flor/`
1. Commit code changes, command-line args, and log records to `git`, inside a dedicated `flor.shadow` branch.


```bash
$ ls ~/.flor 
ml_tutorial_flor.shadow.readme

$ pwd
/home/rogarcia/git/ml_tutorial

$ git branch   
* flor.shadow.readme

$ ls -la ./.flor   
drwxr-xr-x  5 rogarcia   160 Jul 19 09:02 .
drwxr-xr-x  9 rogarcia   288 Jul 19 09:01 ..
-rw-r--r--  1 rogarcia   225 Jul 19 09:02 .replay.json
-rw-r--r--  1 rogarcia  2895 Jul 19 09:02 log_records.csv
-rw-r--r--  1 rogarcia   228 Jul 19 09:02 seconds.json
```

Flor will access and interpret contents of `.flor` automatically. The data and log records will be exposed to the user via SQL or Pandas queries.

# Hindsight Logging

Suppose you wanted to start logging the `device`
identifier where the model is run, as well as the
final `accuracy` after training.
You would add the corresponding logging statements
to `train.py`, for example:
```bash
$ cat train.py | grep -C 5 flor.log
```

```python
...
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

flor.log('device', str(device))    # <-- New logging stmt
Flor.checkpoints(model, optimizer)

# Train the model
total_step = len(train_loader)
for epoch in Flor.loop(range(num_epochs)):
    for i, (images, labels) in Flor.loop(enumerate(train_loader)):
        ...
        if (i + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    total_step,
                    flor.log("loss", loss.item()),
                )
            )

...
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        ...
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the 10000 test images: {} %".format(
            flor.log("accuracy", 100 * correct / total) # <-- New logging stmt
        )
    )
```

```bash
$ pwd
/home/rogarcia/git/ml_tutorial

$ git branch
* flor.shadow.readme

$ git commit -am "hindsight logging stmts added."
[flor.shadow.readme 3c23919] hindsight logging stmts added.
 1 file changed, 2 insertions(+), 2 deletions(-)
```

Typically, when you add a logging statement, logging 
begins "from now on", and you have no visibility into the past.
With hindsight logging, the aim is to allow model developers to send
new logging statements back in time, and replay the past 
efficiently from checkpoint.

In order to do that, we open up an interactive environent from within the `ml_tutorial` directory, and call `flor.replay()`, asking flor to apply the logging statements with the names `device` and `accuracy` to all previous versions (leave `where_clause` null in `flor.replay()`):
```bash
$ pwd
/home/rogarcia/git/ml_tutorial

$ ipython
```
```ipython
In [1]: import flor

In [2]: flor.full_pivot(flor.log_records())
Out[2]: 
                              projid       runid               tstamp  ... batch_size     lr  epochs
0     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  ...         32  0.001       5
1     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  ...         32  0.001       5
2     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  ...         32  0.001       5
3     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  ...         32  0.001       5
4     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T09:01:51  ...         32  0.001       5
...                              ...         ...                  ...  ...        ...    ...     ...
1075  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T10:11:48  ...         32  0.001       5
1076  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T10:11:48  ...         32  0.001       5
1077  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T10:11:48  ...         32  0.001       5
1078  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T10:11:48  ...         32  0.001       5
1079  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T10:11:48  ...         32  0.001       5

[1080 rows x 11 columns]

In [3]: flor.replay(['device', 'accuracy'])
What is the log level of logging statement `device`? Leave blank to infer `DATA_PREP`: 
What is the log level of logging statement `accuracy`? Leave blank to infer `DATA_PREP`: 
...
Continue replaying 12 versions at DATA_PREP level for 39.19 seconds?[Y/n]? Y
Flordb registered 12 replay jobs.
```

Finally, spin up a `flordb` server with the GPU identifier (leave blank for CPU) you wish to allocate to the replay worker:
```bash
python -m flor serve
```
or
```bash
python -m flor serve 0
```

```bash
$ watch "sqlite3 ~/.flor/main.db -header 'select done, path, appvars, count(*) from replay group by done, path, appvars;'"

done|path|appvars|count(*)
0|/Users/rogarcia/git/ml_tutorial|device, accuracy|5
1|/Users/rogarcia/git/ml_tutorial|device, accuracy|5
```

When the process is finished, you will be able to view the values for `device` and `accuracy` for historical executions, and they will continue to be logged in subsequent iterations:

```bash
$ pwd
/Users/rogarcia/git/ml_tutorial

$ ipython
```
```ipython
In [1]: from flor import full_pivot, log_records
In [2]: full_pivot(log_records())
Out[2]: 
                              projid       runid               tstamp        vid  epoch  step      loss batch_size device epochs     lr hidden accuracy
0     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T20:45:36  4054c7...      1   100  0.208682         32   cuda      5  0.001    500    97.58
1     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T20:45:36  4054c7...      1   200  0.213655         32   cuda      5  0.001    500    97.58
2     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T20:45:36  4054c7...      1   300  0.491789         32   cuda      5  0.001    500    97.58
3     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T20:45:36  4054c7...      1   400  0.344357         32   cuda      5  0.001    500    97.58
4     ml_tutorial_flor.shadow.readme  myFirstRun  2023-07-19T20:45:36  4054c7...      1   500  0.269210         32   cuda      5  0.001    500    97.58
...                              ...         ...                  ...        ...    ...   ...       ...        ...    ...    ...    ...    ...      ...
1075  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T20:54:52  25d4c3...      5  1400  0.005633         32   cuda      5  0.001    500    98.03
1076  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T20:54:52  25d4c3...      5  1500  0.021457         32   cuda      5  0.001    500    98.03
1077  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T20:54:52  25d4c3...      5  1600  0.049711         32   cuda      5  0.001    500    98.03
1078  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T20:54:52  25d4c3...      5  1700  0.004853         32   cuda      5  0.001    500    98.03
1079  ml_tutorial_flor.shadow.readme       BATCH  2023-07-19T20:54:52  25d4c3...      5  1800  0.009038         32   cuda      5  0.001    500    98.03

[1080 rows x 13 columns]
```
Note the new columns `device` and `accuracy` that are backfilled.

## Publications

To cite this work, please refer to the [Hindsight Logging](http://www.vldb.org/pvldb/vol14/p682-garcia.pdf) paper (VLDB '21).

FLOR is open source software developed at UC Berkeley. 
[Joe Hellerstein](https://dsf.berkeley.edu/jmh/) (databases), [Joey Gonzalez](http://people.eecs.berkeley.edu/~jegonzal/) (machine learning), and [Koushik Sen](https://people.eecs.berkeley.edu/~ksen) (programming languages) 
are the primary faculty members leading this work.

This work is released as part of [Rolando Garcia](https://rlnsanz.github.io/)'s doctoral dissertation at UC Berkeley,
and has been the subject of study by Eric Liu and Anusha Dandamudi, 
both of whom completed their master's theses on FLOR.
Our list of publications are reproduced below.
Finally, we thank [Vikram Sreekanti](https://www.vikrams.io/), [Dan Crankshaw](https://dancrankshaw.com/), and [Neeraja Yadwadkar](https://cs.stanford.edu/~neeraja/) for guidance, comments, and advice.
[Bobby Yan](https://bobbyy.org/) was instrumental in the development of FLOR and its corresponding experimental evaluation.

* [Hindsight Logging for Model Training](http://www.vldb.org/pvldb/vol14/p682-garcia.pdf). _R Garcia, E Liu, V Sreekanti, B Yan, A Dandamudi, JE Gonzalez, JM Hellerstein, K Sen_. The VLDB Journal, 2021.
* [Fast Low-Overhead Logging Extending Time](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-117.html). _A Dandamudi_. EECS Department, UC Berkeley Technical Report, 2021.
* [Low Overhead Materialization with FLOR](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2020/EECS-2020-79.html). _E Liu_. EECS Department, UC Berkeley Technical Report, 2020. 


## License
FLOR is licensed under the [Apache v2 License](https://www.apache.org/licenses/LICENSE-2.0).
