FlorDB
================================
[![PyPI](https://img.shields.io/pypi/v/flordb.svg?nocache=1)](https://pypi.org/project/flordb/)


FlorDB is a hindsight logging database for the AI/ML lifecycle. It works in tandem with any workflow management solution for Python, such as `Make`, `Airflow`, `MLFlow`, `Docker`, `Slurm`, and `Jupyter`, to manage model developers' logs, execution data, versions of code (via `git`), and `torch` checkpoints. In addition to serving as a nimble experiment management solution for ML Engineers, FlorDB subsumes functionality from bespoke ML systems, operating as a **model registry**, **feature store**, **labeling solution**, and others, as needed.

FlorDB contains a record-replay sub-system to enable hindsight logging: a post-hoc analysis practice that involves adding logging statements *after* encountering a surprise, and efficiently re-training with more logging as needed. When model weights are updated during training, Flor takes low-overhead checkpoints, and uses those checkpoints for replay speedups based on memoization, program slicing, and parallelism. As we will soon discuss, most FlorDB use-cases (e.g. data prep, featurization) do not involve `torch` checkpointing and can use the Flor data model independently of the record-replay sub-system.

FlorDB is software developed at UC Berkeley's [RISE](https://rise.cs.berkeley.edu/) Lab (2017 - 2024). It is actively maintained by [Rolando Garcia](https://rlnsanz.github.io) (rolando.garcia@asu.edu) at ASU's School of Computing & Augmented Intelligence (SCAI).

## Installation
To install the latest stable version of FlorDB, run:

```bash
pip install flordb
```

### Development Installation

For developers who want to contribute, are co-authors on a FlorDB manuscript and plan to run experiments, or need the latest features, install directly from the source:

```bash
git clone https://github.com/ucbrise/flor.git
cd flor
pip install -e .
```

To keep your local copy up-to-date with the latest changes, remember to regularly pull updates from the repository (from within the `flor` directory):

```bash
git pull origin
```

## Just start logging

FlorDB is designed to be easy to use. 
You don't need to define a schema, or set up a database.
Just start logging your runs with a single line of code:

```python
import flor
flor.log("msg", "Hello world!")
```
```
msg: Hello, World!
Changes committed successfully
```

You can read your logs with a Flor Dataframe:

```python
import flor
flor.dataframe("msg")
```
![msg dataframe](img/just_start.png)

## Logging your experiments
FlorDB has a low floor, but a high ceiling. 
You can start logging with a single line of code, but you can also log complex experiments with many hyper-parameters and metrics.

Here's how you can modify your existing PyTorch training script to incorporate FlorDB logging:


```python
import flor
import torch

# Define and log hyper-parameters
hidden_size = flor.arg("hidden", default=500)
batch_size = flor.arg("batch_size", 32)
learning_rate = flor.arg("lr", 1e-3)
...

# Initialize your data loaders, model, optimizer, and loss function
trainloader: torch.utils.data.DataLoader
testloader:  torch.utils.data.DataLoader
optimizer:   torch.optim.Optimizer
net:         torch.nn.Module
criterion:   torch.nn._Loss

# Use FlorDB's checkpointing to manage model states
with flor.checkpointing(model=net, optimizer=optimizer):
    for epoch in flor.loop("epoch", range(num_epochs)):
        for data in flor.loop("step", trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Log the loss value for each step
            flor.log("loss", loss.item())

        # Evaluate the model on the test set
        eval(net, testloader)
```

To view the hyper-parameters and metrics logged during training, you can use the `flor.dataframe` function:

```python
import flor
flor.dataframe("hidden", "batch_size", "lr", "loss")
```
![loss dataframe](img/loss_df.png)

### Logging hyper-parameters
As shown above, you can log hyper-parameters with `flor.arg`:

```python
# Define and log hyper-parameters

hidden_size = flor.arg("hidden", default=500)
batch_size = flor.arg("batch_size", 32)
learning_rate = flor.arg("lr", 1e-3)
...
seed = flor.arg("seed", default=randint(1, 10000))

# Set the random seed for reproducibility
torch.manual_seed(seed)
```

When the experiment is run, the hyper-parameters are logged, and their values are stored in FlorDB.

During replay, `flor.arg` reads the values from the database, so you can easily reproduce the experiment.

### Setting hyper-parameters from the command line
You can set the value of any `flor.arg` from the command line:
```bash 
python train.py --kwargs hidden=250 lr=5e-4
```


## Hindsight Logging for when you miss something
Hindsight logging is a post-hoc analysis practice that involves adding logging statements *after* encountering a surprise, and efficiently re-training with more logging as needed. FlorDB supports hindsight logging across multiple versions with its record-replay sub-system.

### Clone a sample repository
To demonstrate hindsight logging, we will use a sample repository that contains a simple PyTorch training script. Let's clone the repository and install the requirements:

```bash
git clone https://github.com/rlnsanz/ml_tutorial.git
cd ml_tutorial
make install
```

### Record the first two runs
Once you have the repository cloned, and the dependencies installed, you can record the first run with FlorDB:

```bash
python train.py
```
```bash
Created and switched to new branch: flor.shadow
device: cuda
seed: 4179
hidden: 500
epochs: 5
batch_size: 32
lr: 0.001
print_every: 500
epoch: 0, step: 500, loss: 1.2232707738876343
epoch: 0, step: 1000, loss: 0.9084039926528931
...
epoch: 4, step: 1500, loss: 0.4354817569255829
epoch: 4, val_acc: 91.3   
5it [00:23,  4.69s/it]    
accuracy: 91.26
correct: 9126
Changes committed successfully
```
Notice that the `train.py` script logs the loss and accuracy during training. The loss is logged for each step, and the accuracy is logged at the end of each epoch.

Next, you'll want to run training with different hyper-parameters. You can do this by setting the hyper-parameters from the command line:

```bash
python train.py --kwargs epochs=3 batch_size=64 lr=0.0005
```
```bash
device: cuda
seed: 6589
hidden: 500
epochs: 3
batch_size: 64
lr: 0.0005
print_every: 500
epoch: 0, step: 500, loss: 0.2713785469532013
epoch: 0, val_acc: 92.35 
epoch: 1, step: 500, loss: 0.2224295288324356
epoch: 1, val_acc: 92.05 
epoch: 2, step: 500, loss: 0.24274785816669464
epoch: 2, val_acc: 91.75 
3it [00:11,  4.00s/it]   
accuracy: 92.41
correct: 9241
Changes committed successfully
```

Now, you have two runs recorded in FlorDB. You can view the hyper-parameters and metrics logged during training with the `flor.dataframe` function:

```python
import flor
flor.dataframe("device", "seed", "epochs", "batch_size", "lr", "accuracy")
```
![dataframe of two runs](img/two_runs.png)

### Replay the previous runs

Although we logged the learning rate at the beginning of training, we forgot to log the learning rate during training. We can replay the previous runs and log the learning rate for each epoch:

...

## Publications

To cite this work, please refer to the [Multiversion Hindsight Logging](https://arxiv.org/abs/2310.07898) paper (pre-print '23).

FlorDB is open source software developed at UC Berkeley. 
[Joe Hellerstein](https://dsf.berkeley.edu/jmh/) (databases), [Joey Gonzalez](http://people.eecs.berkeley.edu/~jegonzal/) (machine learning), and [Koushik Sen](https://people.eecs.berkeley.edu/~ksen) (programming languages) 
are the primary faculty members leading this work.

This work is released as part of [Rolando Garcia](https://rlnsanz.github.io/)'s [doctoral dissertation](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-142.html) at UC Berkeley,
and has been the subject of study by Eric Liu and Anusha Dandamudi, 
both of whom completed their master's theses on FLOR.
Our list of publications are reproduced below.
Finally, we thank [Vikram Sreekanti](https://www.vikrams.io/), [Dan Crankshaw](https://dancrankshaw.com/), and [Neeraja Yadwadkar](https://cs.stanford.edu/~neeraja/) for guidance, comments, and advice.
[Bobby Yan](https://bobbyy.org/) was instrumental in the development of FLOR and its corresponding experimental evaluation.

* [The Management of Context in the Machine Learning Lifecycle](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-142.html). _R Garcia_. EECS Department, University of California, Berkeley, 2024. UCB/EECS-2024-142.
* [Multiversion Hindsight Logging for Continuous Training](https://arxiv.org/abs/2310.07898). _R Garcia, A Dandamudi, G Matute, L Wan, JE Gonzalez, JM Hellerstein, K Sen_. pre-print on ArXiv, 2023.
* [Hindsight Logging for Model Training](http://www.vldb.org/pvldb/vol14/p682-garcia.pdf). _R Garcia, E Liu, V Sreekanti, B Yan, A Dandamudi, JE Gonzalez, JM Hellerstein, K Sen_. The VLDB Journal, 2021.
* [Fast Low-Overhead Logging Extending Time](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-117.html). _A Dandamudi_. EECS Department, UC Berkeley Technical Report, 2021.
* [Low Overhead Materialization with FLOR](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2020/EECS-2020-79.html). _E Liu_. EECS Department, UC Berkeley Technical Report, 2020. 
* [Context: The Missing Piece in the Machine Learning Lifecycle](https://rlnsanz.github.io/dat/Flor_CMI_18_CameraReady.pdf). _R Garcia, V Sreekanti, N Yadwadkar, D Crankshaw, JE Gonzalez, JM Hellerstein. CMI, 2018.


## License
FlorDB is licensed under the [Apache v2 License](https://www.apache.org/licenses/LICENSE-2.0).
