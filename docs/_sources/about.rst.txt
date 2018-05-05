About
=====

Flor should facilitate the development of auditable, reproducible, justifiable, and reusable data science workflows. Is the data scientist building the right thing? We want to encourage discipline and best practices in ML workflow development by making dependencies explicit, while improving the productivity of adopters by automating multiple runs of the workflow under different configurations. 

Features
--------

* **Simple and Expressive Object Model**:  The Flor object model consists only of *Actions*, *Artifacts*, and *Literals*. These are connected to form dataflow graphs.
* **Data-Centric Workflows**: Machine learning applications have data dependencies that obscure traditional abstraction boundaries. So, the data "gets everywhere": in the models, and the applications that consume them. It makes sense to think about the data carefully and specifically. In Flor, data is a first-class citizen.
* **Artifact Versioning**: Flor uses git to automatically version every Artifact (data, code, etc.) and Literal that is in a Flor workflow. 
* **Artifact Contextualization**: Flor uses Ground_ to store data about the context of *Artifacts*: their relationships, their lineage. Ground and git are complementary services used by Flor. Together, they enable experiment reproduction and replication. 
* **Parallel Multi-Trial Experiments**: Flor should enable data scientists to try more ideas quickly. For this, we need to enhance speed of execution. We leverage parallel execution systems such as Ray_ to execute multiple trials in parallel.

Contributors
------------

Flor is developed and maintained by the RISE_ Lab:

    * Rolando Garcia
    * Vikram Sreekanti
    * Daniel Crankshaw
    * Neeraja Yadwadkar
    * Sona Jeswani
    * Eric Liu
    * Malhar Patel
    * Joseph Gonzalez
    * Joseph Hellerstein

License
-------

Flor is Licensed under the `Apache V2 License`__.

__ https://www.apache.org/licenses/LICENSE-2.0

.. _Ground: http://www.ground-context.org/
.. _Ray: https://github.com/ray-project/ray
.. _RISE: https://rise.cs.berkeley.edu/
