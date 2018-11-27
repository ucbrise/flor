Flor: The Context Management System that feels like a logger
=============================================================

Flor_ is made for data scientists who write ML code to train models, it helps you understand what alternatives you've tried, and with what results.
It alleviates the burden of tracking the data, code, and parameters used to train a model,
and associating such metadata with the model's metrics, for each execution.
Flor_ automatically tracks this context on every execution so your changes are **reversible** and **redo-able** --
you can focus on exploration and composition.


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

Next, **Add the directory containing this flor package (repo) to your PYTHONPATH.**

Quickstart
----------

Create a Python file named `plate.py`:

.. code-block:: python

    import flor

    with flor.Experiment('plate_demo') as ex:

        ex.groundClient('git')

        ones = ex.literalForEach([1, 2, 3], "ones")
        tens = ex.literalForEach([10, 100], "tens")
        
        @flor.func
        def multiply(x, y):
            print(x*y)
            return x*y

        doMultiply = ex.action(multiply, [ones, tens])
        product = ex.artifact('product.txt', doMultiply)

    product.plot()
    product.pull()

To run the file:

.. code-block:: bash

    # Within a Python3.6 Anaconda environment
    $ python plate.py

The expected output is as follows:

.. code-block:: bash

    10
    20
    30
    100
    200
    300




.. _Flor: https://github.com/ucbrise/flor

Table of Contents
^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 1
    :caption: Installation

   about
   api_documentation
   support
