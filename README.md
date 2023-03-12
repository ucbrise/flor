FlorDB: Nimble Experiment Management for Iterative ML
================================

Flor (for "fast low-overhead recovery") is a record-replay system for deep learning, and other forms of machine learning that train models on GPUs. Flor was developed to speed-up hindsight logging: a cyclic-debugging practice that involves adding logging statements *after* encountering a surprise, and efficiently re-training with more logging. Flor takes low-overhead checkpoints during training, or the record phase, and uses those checkpoints for replay speedups based on memoization and parallelism.

FlorDB integrates Flor, `git` and `sqlite3` to manage model developer's logs, execution data, versions of code, and training checkpoints. In addition to serving as an experiment management solution for ML Engineers, FlorDB extends hindsight logging across model trainging versions for the retroactive evaluation of iterative ML.

Flor and FlorDB are software developed at UC Berkeley's [RISE](https://rise.cs.berkeley.edu/) Lab.
# Installation

```bash
pip install flordb
```

# First run

Run the ``examples/rnn.py`` script to test your installation. 
This script will train a small linear model on MNIST.
FLOR shadow branches permit us to commit your work
automatically on every run, without interfering with your 
other commits. You can later review and merge the flor shadow branch as you would any other git branch.

```bash
git checkout -b flor.shadow
python examples/rnn.py --flor readme
```

When finished, you will have committed to the shadow branch and written execution metadata into a `.flor` directory in your current directory. Additionally, flor created a directory tree in your HOME to organize your experiments. You can find our running experiment as follows:

```bash
ls ~/.flor/
```
Confirm that Flor saved checkpoints of the ``examples/rnn.py`` execution on your home directory.
Flor will access and interpret contents of ``~/.flor`` automatically. 
You should routinely clear this stash or spool it to the cloud to clear up disk space.

# View your experiment history
From the same directory you ran the examples above, open an iPython terminal, then load and pivot the log records.

```ipython
In [1]: from flor import full_pivot, log_records
In [2]: full_pivot(log_records())
Out[2]: 
     runid              tstamp  epoch  step device learning_rate                 loss
0   readme 2023-03-12 12:23:53      1   100    cpu          0.01   0.5304957032203674
1   readme 2023-03-12 12:23:53      1   200    cpu          0.01  0.21829535067081451
2   readme 2023-03-12 12:23:53      1   300    cpu          0.01  0.15856705605983734
3   readme 2023-03-12 12:23:53      1   400    cpu          0.01  0.11441942304372787
4   readme 2023-03-12 12:23:53      1   500    cpu          0.01  0.06835074722766876
5   readme 2023-03-12 12:23:53      1   600    cpu          0.01  0.13750575482845306
6   readme 2023-03-12 12:23:53      2   100    cpu          0.01  0.11708579957485199
7   readme 2023-03-12 12:23:53      2   200    cpu          0.01  0.08852845430374146
8   readme 2023-03-12 12:23:53      2   300    cpu          0.01  0.16527307033538818
9   readme 2023-03-12 12:23:53      2   400    cpu          0.01  0.11036019027233124
10  readme 2023-03-12 12:23:53      2   500    cpu          0.01  0.05740281194448471
11  readme 2023-03-12 12:23:53      2   600    cpu          0.01  0.07785198092460632

```

# Model Training Kit (MTK)
The MTK includes utilities for serializing and checkpointing PyTorch state,
and utilities for resuming, auto-parallelizing, and memoizing executions from checkpoint.
The model developer passes objects for checkpointing to flor,
and gives it control over loop iterators by calling `MTK.checkpoints`
and `MTK.loop` as follows:

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
    eval(net, testloader)
```
As shown, we pass the neural network and optimizer to Flor 
for checkpointing with `Flor.checkpoints(net, optimizer)`.
We wrap both the nested training loop and main loop with 
`Flor.loop`. This lets Flor jump to an arbitrary epoch
using checkpointed state, 
and skip the nested training loop when intermediate
state isn't probed.


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
python3 mytrain.py --replay_flor PID/NGPUS [your_flags]
```

As before, you tell FLOR to run in replay mode by setting ``--replay_flor``.
You'll also tell FLOR how many GPUs from the pool to use for parallelism,
and you'll dispatch this script simultaneously, varying the ``pid:<int>``
to span all the GPUs. To run segment 3 out of 5 segments, you would write: ``--replay_flor 3/5``.

If instead of replaying all of training you wish to re-execute only a fraction of the epochs
you can do this by setting the value of ``ngpus`` and ``pid`` respectively.
Suppose you want to run the tenth epoch of a training job that ran for 200 epochs. You would set
``pid:9``and ``ngpus:200``.

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
