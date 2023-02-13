<!-- ![Travis](https://travis-ci.com/ucbrise/flor.svg?branch=master)
![Python37](https://img.shields.io/badge/python-3.7-blue.svg)
[![](https://badge.fury.io/py/pyflor.svg)](https://pypi.org/project/pyflor/)
[![codecov](https://codecov.io/gh/ucbrise/flor/branch/master/graph/badge.svg)](https://codecov.io/gh/ucbrise/flor)
 -->

FLOR: Experiment Management for ML Engineers
================================

You can use FLOR to take checkpoints during model training.
These checkpoints allow you to restore arbitrary training data post-hoc and efficiently,
thanks to memoization and parallelism speedups on replay.

FLOR is a suite of machine learning tools for hindsight logging.
Hindsight logging is an optimistic logging practice favored by agile model developers. 
Model developers log training metrics such as the loss and accuracy by default, 
and selectively restore additional training data --- like tensor histograms, images, and overlays --- post-hoc, 
if and when there is evidence of a problem. 

FLOR is software developed at UC Berkeley's [RISE](https://rise.cs.berkeley.edu/) Lab, 
and is being released as part of an accompanying [VLDB publication](http://www.vldb.org/pvldb/vol14/p682-garcia.pdf).

# Installation

```bash
pip install pyflor
```
FLOR expects a recent version of Python (3.7+) and PyTorch (1.0+).

```bash
git branch flor.shadow
git checkout flor.shadow
python3 examples/linear.py --flor linear
```
Run the ``linear.py`` script to test your installation. 
This script will train a small linear model on MNIST.
Think of it as a ''hello world'' of deep learning.
We will cover FLOR shadow branches later.

```bash
ls ~/.flor/linear
```
Confirm that FLOR saved checkpoints of the ``linear.py`` execution on your home directory.
FLOR will access and interpret contents of ``~/.flor`` automatically. 
Do watch out for storage footprint though. 
If you see disk space running out, check ``~/.flor``.
FLOR includes utilities for spooling its checkpoints to [S3](https://aws.amazon.com/s3).

# Preparing your Training Script

```python
from flor import MTK as Flor
for epoch in Flor.loop(range(...)):
    ...
```

First, wrap the iterator of the main loop with FLOR's generator: ``Flor.loop``. 
The generator enables FLOR to parallelize replay of the main loop,
and to jump to an arbitrary epoch for data recovery.

```python
from flor import MTK as Flor

import torch

trainloader: torch.utils.data.DataLoader
testloader:  torch.utils.data.DataLoader
optimizer:   torch.optim.Optimizer
net:         torch.nn.Module
criterion:   torch.nn._Loss

Flor.checkpoints(net, optimizer)
for epoch in Flor.loop(range(...)):
    for data in Flor.loop(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}")
    eval(net, testloader)
```
That's it, your training code is now ready for record-replay.

# Training your model

```bash
python3 training_script.py --flor NAME [your_script_flags]
```

Before we train your model, 
**make sure that your model training code is part of a git repository**.
Model training is exploratory and it's common to iterate dozens of times
before finding the right fit.
We'd hate for you to be manually responsible for managing all those versions.
Instead, we ask you to create a FLOR shadow branch
that we can automatically commit changes to.
Think of it as a sandbox: you get the benefits of autosaving,
without worrying about us poluting your main branch with frequent & automatic commits.
Later, you can merge the changes you like.

In FLOR, all experiments need a name. 
As your training scripts and configurations evolve,
keep the same experiment name so FLOR 
associates the checkpoints as versions of the same experiment.
If you want to re-use the name from the previous run, 
you may leave the field blank.

# Hindsight Logging

```python
from flor import MTK as Flor
import torch

trainloader: torch.utils.data.DataLoader
testloader:  torch.utils.data.DataLoader
optimizer:   torch.optim.Optimizer
net:         torch.nn.Module
criterion:   torch.nn._Loss

for epoch in Flor.loop(range(...)):
    for batch in Flor.loop(trainloader):
        ...
    eval(net, testloader)
    log_confusion_matrix(net, testloader)
```

Suppose you want to view a confusion matrix as it changes
throughout training.
Add the code to generate the confusion matrix, as sugared above.

```bash
python3 training_script.py --replay_flor
```

You first switch to the FLOR shadow branch,
and select the version you wish to replay
from the `git log` list. 
In our example, we won't checkout version,
because we want to replay the latest version,
which is selected by default.

You will tell FLOR to replay by setting the flag ``--replay_flor``. 
FLOR is performing fast replay, so you may generalize this
example to recover ad-hoc training data.
In our example, FLOR will compute your confusion matrix 
and automatically skip the nested training loop 
by loading its checkpoints.

```python
from flor import MTK as Flor
import torch

trainloader: torch.utils.data.DataLoader
testloader:  torch.utils.data.DataLoader
optimizer:   torch.optim.Optimizer
net:         torch.nn.Module
criterion:   torch.nn._Loss

for epoch in Flor.loop(range(...)):
    for batch in Flor.loop(trainloader, probed=True):
        ...
    eval(net, testloader)
    log_confusion_matrix(net, testloader)
```

Now, suppose you also want [TensorBoard](https://www.tensorflow.org/tensorboard)
to plot the tensor histograms.
In this case, it is not possible to skip the nested training loop
because we are probing intermediate data.
We tell FLOR to step into the nested training loop by setting ``probed=True``.

Although we can't skip the nested training loop, we can parallelize replay or
re-execute just a fraction of the epochs (e.g. near the epoch where we see a loss anomaly).

```bash
python3 training_script.py --replay_flor PID/NGPUS [your_flags]
```

As before, you tell FLOR to run in replay mode by setting ``--replay_flor``.
You'll also tell FLOR how many GPUs from the pool to use for parallelism,
and you'll dispatch this script simultaneously, varying the ``pid:<int>``
to span all the GPUs. To run segment 3 out of 5 segments, you would write: ``--replay_flor 3/5``.

If instead of replaying all of training you wish to re-execute only a fraction of the epochs
you can do this by setting the value of ``ngpus`` and ``pid`` respectively.
Suppose you want to run the tenth epoch of a training job that ran for 200 epochs. You would set
``pid:9``and ``ngpus:200``.

We provide additional examples in the ``examples`` directory. A good starting point is ``linear.py``. 

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
