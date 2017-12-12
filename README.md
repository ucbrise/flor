Jarvis
=====

Build, configure, and track workflows with Jarvis.

## What is Jarvis?
Jarvis is a system with a declarative DSL embedded in python for managing the workflow development phase of the machine learning lifecycle. Jarvis enables data scientists to describe ML workflows as directed acyclic graphs (DAGs) of *Actions* and *Artifacts*, and to experiment with different configurations by automatically running the workflow many times, varying the configuration. To date, Jarvis serves as a build system for producing some desired artifact, and serves as a versioning system that enables tracking the evolution of artifacts across multiple runs in support of reproducibility.

## Example program
Contents of the `plate.py` file:
```python
import jarvis

jarvis.groundClient('git')
jarvis.jarvisFile('plate.py')

ones = jarvis.Literal([1, 2, 3], "ones")
ones.forEach()

tens = jarvis.Literal([10, 100], "tens")
tens.forEach()

@jarvis.func
def multiply(x, y):
z = x*y
print(z)
return z

doMultiply = jarvis.Action(multiply,
[ones, tens])
product = jarvis.Artifact('product.txt',
doMultiply)

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
* **Simple and Expressive Object Model**: The Jarvis object model consists exclusively of *Actions*, *Artifacts*, and *Literals*, that are connected to form dataflow graphs.
* **Data-Centric Workflows**: Machine learning applications have data dependencies that obscure traiditional abstraction boundaries, and complicate the use of standard software engineering practices and tools. In Jarvis, data is a first-class citizen.
* **Artifact Versioning**: Jarvis uses git to automatically version every (data, code, etc.) Artifact in a Jarvis workflow.
* **Artifact Contextualization**: Jarvis uses [Ground](http://www.ground-context.org/) to store data about the context of *Artifacts*. Ground and git are complementary services used by Jarvis.
* **Parallel Multi-Trial Experiments**: We hope that Jarvis will enable data scientists to try more ideas quickly. For this, we need to enhance speed of execution, and therefore levarage parallel execution systems such as [Ray](https://github.com/ray-project/ray) to execute multiple trials in parallel.
* **Visualization and Exploratory Data Analysis**: In order to establish the fitness of data for some particular purpose, or gain valuable insights about charactersitics of the data, it will be useful to levarage visualization techniques in an interactive environment such as Jupyter Notebook. We use visualization for its ability to give immediate feedback and guide the creative process. 

## License
Jarvis is licensed under the [Apache v2 License](https://www.apache.org/licenses/LICENSE-2.0).