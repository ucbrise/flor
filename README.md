Jarvis
=====

Build, configure, and track workflows with Jarvis.

## What is Jarvis?
Jarvis is a system with a declarative DSL embedded in python for managing the workflow development phase of the machine learning lifecycle. Jarvis enables data scientists to describe ML workflows as directed acyclic graphs (DAGs) of *Actions* and *Artifacts*, and to experiment with different configurations by automatically running the workflow many times, varying the configuration. To date, Jarvis serves as a build system for producing some desired artifact, and serves as a versioning system that enables tracking the evolution of artifacts across multiple runs in support of reproducibility.

## How do I run it?

Clone or download this repository.

You'll need Anaconda, preferable version 4.4+

Please read [this guide](https://conda.io/docs/user-guide/tasks/manage-environments.html) to set up a Python 3.6 environment inside Anaconda. **Whenever you work with Jarvis, make sure the Python 3.6 environment is active**.

Once the Python 3.6 environment in Anaconda is active, please run the following command (use the requirements.txt file in this repo):
```
pip install -r requirements.txt
```

Next, we will install RAY, a Jarvis dependency:

```
brew update
brew install cmake pkg-config automake autoconf libtool boost wget

pip install numpy funcsigs click colorama psutil redis flatbuffers cython --ignore-installed six
conda install libgcc

pip install git+https://github.com/ray-project/ray.git#subdirectory=python
```

Next, **Add the directory containing this jarvis package (repo) to your `PYTHONPATH`.**

For examples on how to write your own jarvis workflow, please have a look at:
```
examples/twitter.py -- classic example
examples/plate.py -- multi-trial example
examples/lifted_twitter.py -- multi-trial + aggregation example
```

Make sure you:
1. Import `jarvis`
2. Initialize a `jarvis.Experiment`
2. set the experiment's `groundClient` to 'git'.

Once you build the workflow, call `parallelPull()` on the artifact you want to produce. You can find it in `~/jarvis.d/`.

If you pass in a non-empty `dict` to `parallelPull` (see `lifted_twitter.py`), the call will return a pandas dataframe with literals and requested artifacts for the columns, and different trials for the rows.

## Example program
Contents of the `examples/plate.py` file:
```python
import jarvis

ex = jarvis.Experiment('plate_demo')

ex.groundClient('git')

ones = ex.literal([1, 2, 3], "ones")
ones.forEach()

tens = ex.literal([10, 100], "tens")
tens.forEach()

@jarvis.func
def multiply(x, y):
    z = x*y
    print(z)
    return z

doMultiply = ex.action(multiply, [ones, tens])
product = ex.artifact('product.txt', doMultiply)

product.pull()
product.plot()
```
On run produces:
```shell
10
20
30
100
200
300
```

## Motivation
Jarvis should facilitate the development of auditable, reproducible, justifiable, and reusable data science workflows. Is the data scientist building the right thing? We want to encourage discipline and best practices in ML workflow development by making dependencies explicit, while improving the productivity of adopters by automating multiple runs of the workflow under different configurations. 

## Features
* **Simple and Expressive Object Model**:  The Jarvis object model consists only of *Actions*, *Artifacts*, and *Literals*. These are connected to form dataflow graphs.
* **Data-Centric Workflows**: Machine learning applications have data dependencies that obscure traditional abstraction boundaries. So, the data "gets everywhere": in the models, and the applications that consume them. It makes sense to think about the data carefully and specifically. In Jarvis, data is a first-class citizen.
* **Artifact Versioning**: Jarvis uses git to automatically version every Artifact (data, code, etc.) and Literal that is in a Jarvis workflow. 
* **Artifact Contextualization**: Jarvis uses [Ground](http://www.ground-context.org/) to store data about the context of *Artifacts*: their relationships, their lineage. Ground and git are complementary services used by Jarvis. Together, they enable experiment reproduction and replication. 
* **Parallel Multi-Trial Experiments**: Jarvis should enable data scientists to try more ideas quickly. For this, we need to enhance speed of execution. We leverage parallel execution systems such as [Ray](https://github.com/ray-project/ray) to execute multiple trials in parallel.
* **Visualization and Exploratory Data Analysis**: To establish the fitness of data for some particular purpose, or gain valuable insights about properties of the data, Jarvis will leverage visualization techniques in an interactive environment such as Jupyter Notebook. We use visualization for its ability to give immediate feedback and guide the creative process.


## License
Jarvis is licensed under the [Apache v2 License](https://www.apache.org/licenses/LICENSE-2.0).
