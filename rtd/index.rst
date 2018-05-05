UCB Flor
========

Flor_ (formerly known as Jarvis_) is a system with a declarative DSL embedded in python for managing the workflow development phase of the machine learning lifecycle. Flor enables data scientists to describe ML workflows as directed acyclic graphs (DAGs) of *Actions* and *Artifacts*, and to experiment with different configurations by automatically running the workflow many times, varying the configuration. To date, Flor serves as a build system for producing some desired artifact, and serves as a versioning system that enables tracking the evolution of artifacts across multiple runs in support of reproducibility.



Install Flor
------------

Clone or download the Flor_ repository.

You'll need Anaconda, preferably version 4.4+

Please read `this guide`__ to set up a Python 3.6 environment inside Anaconda. **Whenever you work with Flor, make sure the Python 3.6 environment is active**.

__ https://conda.io/docs/user-guide/tasks/manage-environments.html

Once the Python 3.6 environment in Anaconda is active, please run the following command (use the requirements.txt file in this__ repo):

.. code-block:: bash

	pip install -r requirements.txt

__ https://github.com/ucbrise/flor

Next, we will install RAY, a Flor dependency:

.. code-block:: bash

	brew update
	brew install cmake pkg-config automake autoconf libtool boost wget

	pip install numpy funcsigs click colorama psutil redis flatbuffers cython --ignore-installed six
	conda install libgcc

	pip install git+https://github.com/ray-project/ray.git#subdirectory=python

Next, **Add the directory containing this flor package (repo) to your :bash:`PYTHONPATH`.**

.. _Jarvis: https://github.com/ucbrise/jarvis
.. _Flor: https://github.com/ucbrise/flor
.. role:: bash
	:class: code
	:language: bash










Table of Contents
^^^^^^^^^^^^^^^^^

.. toctree::
   
   about
   api_documentation
   support
