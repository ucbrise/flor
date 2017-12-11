Jarvis
=====

Build, configure, and track workflows with Jarvis.

## What is Jarvis?
Jarvis is a system with a declarative DSL embedded in python for managing the workflow development phase of the machine learning lifecycle. Jarvis enables data scientists to describe ML workflows as directed acyclic graphs (DAGs) of *Actions* and *Artifacts*, and to experiment with different configurations by automatically running the workflow many times, varying the configuration. To date, Jarvis serves as a build system for producing some desired artifact, and serves as a versioning system that enables tracking the evolution of artifacts across multiple runs in support of reproducibility.

## Installation
Instructions pending.

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
Jarvis should facilitate the development of auditable, reproducible, justifiable, and reusable data science workflows. Is the data scientist building the right thing? We want to encourage discipline and best practices in ML workflow development by making dependencies explicit, while improving the productivity of adopters by automating multiple runs of the workflow under different configurations. The motivation for Jarvis has been explained as porting software engineering practices to machine learning. Data scientists should not be training models on default hyper-parameters merely because of convenience: what are the assumptions that justify the choice of hyper-parameters? Is the choice of ML model justified given the distribution of the data? Jarvis should also provide lineage information so developers can understand why a model behaved the way that it did, in cases of failure or exceptional performance: what were they hyper-parameters and scripts that led to a high classification accuracy in production?

## Challenges
* **Running the workflow many times, varying the configuration**:  The Jarvis workflow specification language is declarative, and we wanted to maintain this declarativity even when describing parallel and iterative computation. It's likely that a data scientist will want to run some workflow for each value of a hyper-parameter, say gamma, in some range. Jarvis supports this by wrapping that range in a *Literal*, and calling the `forEach()` method on that *Literal*. Then, when that literal is a dependency to some *Action*, Jarvis will run the workflow many times by iterating in parallel over the values in the list wrapped by that literal. When there are many such literals, the number of computations is the cross product of those literals. Keeping the execution environments of these workflows, with filesystem dependencies, isolated, when running in parallel, was a challenge. Reducing the amount of computation by re-using identical artifacts across parallel workflow executions, respecting isolation, is another foreseeable challenge.

* **Coherent types**: Before the introduction of Jarvis *Literals*, the types of objects in a Jarvis workflow were straightforward. There were only Actions and Artifacts. After the introduction of *Literals*, with the added feature for specifying parallel and iterative computation declaratively, the typing was complicated: is a Jarvis Action that depends on an iterable array of configurations a single action or many actions? What about the downstream artifacts this action produces: are they one artifact or many? We are still in the process of working through this added complication. Some alternatives to explore are python type annotations, or a formal specification of the types in the documentation.'

* **Adding features without bloating the object model**: Our goal is for the Jarvis workflow specification language to be simple and expressive, but often simplicity and expressivity are at odds with each other. Throughout the development of Jarvis, we added classes to `Fork`, `Sample`, `Split`, and had other classes in mind. These classes were in some ways hybrids of Jarvis *Actions* and *Artifacts*. The motivation behind them was to natively support common Machine Learning patterns, but we noticed that the introduction of these classes soon made the Jarvis types unintelligible. Thus, we removed these classes, and found that the introduction of a *Literal* was sufficient to express parallel and iterative computation, without unduly complicating the object types. This simplicity comes at the cost of the programmer having to write their own algorithms for common Machine Learning patterns. We hope to reduce this cost by writing libraries on top of Jarvis: if we can explain the higher level functionality in terms of Jarvis *Actions*, *Artifacts*, and *Literals*, it may be possible to boost productivity without compromising simplicity.

* [Pending] **Live monitoring and interruption during validation**: The workflow validation stage is the stage during which Jarvis will run many, possibly time consuming, trials to establish the reliability or trustworthiness of the workflow, and enable the data scientist to know which workflow configurations they should use. Because the workflow validation stage is expected to have a long duration, it is important to offer the data scientist tools to monitor the live behavior of the experiments, and control or interrupt the execution to save time and computing resources, or guide the search in better directions. TensorBoard is a promising system for these purposes, and our first attempt will be to integrate it with Jarvis and extend it the best we can before trying other systems or implementing our own.

* [Pending] **Incremental Computation**:  Different trials for the same experiment differ only in their configuration, and the same experiment as it evolves in time has pieces we can reuse. At a minimum, we will support *make* style incremental computation: we will use the timestamps of downstream artifacts to determine whether to run that action. However, there are non-functional alterations to code that change the timestamp but don't justify re-execution of a subgraph in the workflow: for example, adding a comment in the code. Being able to determine *semantic change* in an efficient manner is a foreseeable challenge. Additionally, we will need to think of change in terms of data as well as code. Determining whether data artifacts have changed, and to what extent, is another foreseeable challenge.

* [Pending] **Interactive environment support**: Data scientists in the wild use interactive environments such as Jupyter Notebooks. As a workflow development system, Jarvis should support the creative design and discovery of data science workflows. This process will necessarily involve real-time feedback and support for user interaction. Thus, we have numerous reasons to support interactive environments. However, interactive environments pose challenges for reproducibility, since the cells can be executed non-linearly. Supporting interactivity without compromising versioning and reproducibility is another foreseeable challenge. 

## Approach
### Simple and expressive object model
### Data-centric workflows

## Results
This project is still work in progress. Please check back at a later time.

