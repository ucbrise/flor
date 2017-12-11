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
