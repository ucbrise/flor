<!-- ![Travis](https://travis-ci.com/ucbrise/flor.svg?branch=master)
![Python37](https://img.shields.io/badge/python-3.7-blue.svg)
[![](https://badge.fury.io/py/pyflor.svg)](https://pypi.org/project/pyflor/)
[![codecov](https://codecov.io/gh/ucbrise/flor/branch/master/graph/badge.svg)](https://codecov.io/gh/ucbrise/flor)
 -->

FLOR: Fast Low-Overhead Recovery
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
FLOR expects a recent version of Python (3.6+) and PyTorch (1.0+).

```bash
python3 examples/linear.py --flor linear
```
Run the ``linear.py`` script to test your installation. 
This script will train a small linear model on MNIST.
Think of it as a ''hello world'' of deep learning.

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
import flor
for epoch in flor.it(range(...)):
    ...
```

First, wrap the iterator of the main loop with FLOR's generator: ``flor.it``. 
The generator enables FLOR to parallelize replay of the main loop,
and to jump to an arbitrary epoch for data recovery.
FLOR also relies on this generator for initialization and clean-up, so don't skip this step.

```python
import flor
import torch

trainloader: torch.utils.data.DataLoader
testloader:  torch.utils.data.DataLoader
optimizer:   torch.optim.Optimizer
net:         torch.nn.Module
criterion:   torch.nn._Loss

for epoch in flor.it(range(...)):
    if flor.SkipBlock.step_into('training_loop'):
        for data in trainloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"loss: {loss.item()}")
    flor.SkipBlock.end(net, optimizer)
    eval(net, testloader)
```

Then, wrap the nested training loop inside a ``flor.SkipBlock`` as shown above.
Add the stateful ``torch`` objects to ``flor.SkipBlock.end`` so FLOR checkpoints them
periodically.  

You can use SkipBlocks to memoize long-running code. 
Just make sure you give each SkipBlock a unique name (e.g. `training_loop`).

**That's it!** Your code is now ready for record-replay.

# Training your model

```bash
python3 training_script.py --flor NAME [your_script_flags]
```

In FLOR, all experiments need a name. 
As your training scripts and configurations evolve,
keep the same experiment name so FLOR 
associates the checkpoints as versions of the same experiment.

# Hindsight Logging

```python
import flor
import torch

trainloader: torch.utils.data.DataLoader
testloader:  torch.utils.data.DataLoader
optimizer:   torch.optim.Optimizer
net:         torch.nn.Module
criterion:   torch.nn._Loss

for epoch in flor.it(range(...)):
    if flor.SkipBlock.step_into('training_loop'):
        ...
    flor.SkipBlock.end(net, optimizer)
    eval(net, testloader)
    log_confusion_matrix(net, testloader)
```

Suppose you want to view a confusion matrix as it changes
throughout training.
Add the code to generate the confusion matrix, as sugared above.

```bash
python3 training_script.py --flor NAME --replay_flor [your_script_flags]
```

And tell FLOR to replay by setting the flag ``--replay_flor``. 
FLOR is performing fast replay, so you may generalize this
example to recover ad-hoc training data.
In our example, FLOR will compute your confusion matrix 
and automatically skip the nested training loop 
by loading its checkpoints.

```python
import flor
import torch

trainloader: torch.utils.data.DataLoader
testloader:  torch.utils.data.DataLoader
optimizer:   torch.optim.Optimizer
net:         torch.nn.Module
criterion:   torch.nn._Loss

for epoch in flor.it(range(...)):
    if flor.SkipBlock.step_into('training_loop', probed=True):
        ...
        log_tensor_histograms(net.parameters())
    flor.SkipBlock.end(net, optimizer)
    eval(net, testloader)
    log_confusion_matrix(net, testloader)
```

Now, suppose you also want [TensorBoard](https://www.tensorflow.org/tensorboard)
to plot the tensor histograms.
In this case, it is not possible to skip the nested training loop
because we are probing intermediate data.
We tell FLOR to step into the nested training loop by setting ``probed=True`` 
(an argument to the training loop's SkipBlock).

Although we can't skip the nested training loop, we can parallelize replay or
re-execute just a fraction of the epochs (e.g. near the epoch where we see a loss anomaly).

```bash
python3 training_script.py --flor NAME --replay_flor PID/NGPUS [your_flags]
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

## License
FLOR is licensed under the [Apache v2 License](https://www.apache.org/licenses/LICENSE-2.0).
