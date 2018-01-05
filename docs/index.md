# Jarvis

![MLLifecycle](MLLifecycle.png "Machine Learning Lifecycle")

Jarvis is a system with a declarative DSL embedded in python for managing the workflow development phase of the machine learning lifecycle. Jarvis enables data scientists to describe ML workflows as directed acyclic graphs (DAGs) of *Actions* and *Artifacts*, and to experiment with different configurations by automatically running the workflow many times, varying the configuration. To date, Jarvis serves as a build system for producing some desired artifact, and serves as a versioning system that enables tracking the evolution of artifacts across multiple runs in support of reproducibility.

## Features
* **Simple and Expressive Object Model**:  The Jarvis object model consists only of *Actions*, *Artifacts*, and *Literals*. These are connected to form dataflow graphs.
* **Data-Centric Workflows**: Machine learning applications have data dependencies that obscure traditional abstraction boundaries. So, the data "gets everywhere": in the models, and the applications that consume them. It makes sense to think about the data carefully and specifically. In Jarvis, data is a first-class citizen.
* **Artifact Versioning**: Jarvis uses git to automatically version every Artifact (data, code, etc.) and Literal that is in a Jarvis workflow. 
* **Artifact Contextualization**: Jarvis uses [Ground](http://www.ground-context.org/) to store data about the context of *Artifacts*: their relationships, their lineage. Ground and git are complementary services used by Jarvis. Together, they enable experiment reproduction and replication. 
* **Parallel Multi-Trial Experiments**: Jarvis should enable data scientists to try more ideas quickly. For this, we need to enhance speed of execution. We leverage parallel execution systems such as [Ray](https://github.com/ray-project/ray) to execute multiple trials in parallel.
* **Visualization and Exploratory Data Analysis**: To establish the fitness of data for some particular purpose, or gain valuable insights about properties of the data, Jarvis will leverage visualization techniques in an interactive environment such as Jupyter Notebook. We use visualization for its ability to give immediate feedback and guide the creative process.

## Here for the demo?
1. Please read and complete the HOW TO section of the [README](https://github.com/ucbrise/jarvis/#how-do-i-run-it)
2. Please activate your Python 3.6 environment in Anaconda: `source activate [name_of_env]`
3. `cd` into `jarvis\examples\` and run: `python twitter_demo.py` (Your Python 3.6 environment in Anaconda should still be active).
4. Open `train_model.py` and comment line 50 and uncomment line 51.
5. Run: `python twitter_demo.py`
6. From `jarvis\examples\`, and with your Python 3.6 environment in Anaconda still active, type `jupyter notebook`
7. Walk-through `JarvisDemo.ipynb` from within Jupyter Notebook.
8. To exit out of the Country-of-origin prediction app, simply enter `exit` into the text input box.
