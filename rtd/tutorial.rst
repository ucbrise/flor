Tutorial
========

Flor is a *context-centric* logger and automatic version controller to help you create model training pipelines.
Context is enriched metadata: it is also the **version history** and **lineage** of data (or other artifacts or entities
in the Machine Learning lifecycle). Flor leverages git for version control and relies on program analysis to establish
the lineage of metrics or output artifacts from data, code, and parameters.
You may, for example, want to know whether a new parameterization of the model improved accuracy;
alternatively, you may ask which earlier version produced the best results, and want to restore that previous
state for further exploration.

Flor uses the Ground context meta-model (`Figure 2 <http://cidrdb.org/cidr2017/papers/p111-hellerstein-cidr17.pdf>`_).
You can read more about the motivation for Flor in our workshop `paper <https://rlnsanz.github.io/dat/Flor_CMI_18_CameraReady.pdf>`_.


Introduction
------------

Below is a toy script we'll use to introduce several key Flor concepts:

.. code-block:: python

    import numpy as np
    from sklearn.metrics import mean_squared_error

    # mock: get the test data
    y_actual = np.array([
        0.37454012, 0.95071431,
        0.73199394, 0.59865848,
        0.15601864])

    for seed in range(50):
        # mock: fit the model
        np.random.seed(seed)

        # mock: make a prediction
        y_pred = np.random.rand(len(y_actual))

        # mock: measure the loss
        mse = mean_squared_error(y_actual, y_pred)

        print("seed: {}, mse: {}".format(seed, mse))

We can think of the numpy random number generator as a model parameterized by the seed --
for simplicity of exposition, we defer the discussion of how to use data.
We start by "fitting" the random number generator to the seed-parameter, and then make a "prediction"
with the same dimensions as the ``y_actual`` vector. Finally, we compare ``y_pred`` and ``y_actual`` to get a measure of loss.
If we were doing optimization and tuning, we would care about which parameterization minimized the mean squared error.

First, we'll do the least amount of work needed for Flor tracking:

.. code-block:: python

    import numpy as np
    from sklearn.metrics import mean_squared_error

    ##### IMPORT ####
    import flor
    log = flor.log
    #################

    y_actual = np.array([
        0.37454012, 0.95071431,
        0.73199394, 0.59865848,
        0.15601864])

    ### Put the code in a decorated function ###
    @flor.track
    def fit_and_score_model():
        for seed in range(50):
            np.random.seed(seed)
            y_pred = np.random.rand(len(y_actual))
            mse = mean_squared_error(y_actual, y_pred)
            print("seed: {}, mse: {}".format(seed, mse))

    ### Invoke the function from within a Flor Context ###
    with flor.Context('introduction'):
        fit_and_score_model()

We've imported Flor and moved the code we want to track into a flor-decorated function.
The code we want to be tracked has to be in a function body because Python is an interpreted language,
but Flor does program analysis by walking a parse-tree (which is compiled rather than interpreted).
You can ignore this detail, but you should remember that **any code you want Flor to track should be in
a Flor decorated function**.

Another detail you will notice as unusual is that instead of simply invoking ``fit_and_score_model()``,
the function was invoked from within a (named) Flor context. This is because the *identity* of experiments
or executions is important. Your experiment may have the same identity even if you change the filename of the files
that contain it, run the experiment in a different computer or repository, or make even more drastic changes.
We rely on you to tell us what experiment you're running, rather than trying to infer this ourselves.
This gives you the flexibility to make any change you want, so long as you keep the name of the experiment the same,
and to have multiple experiments in the same directory
or repository --- more on this later. The important takeaway is that **every Flor experiment must have a Flor context
and the name of the context must be unique in your scope (for now, your personal computer).**
Although the code that runs inside Flor decorated functions and contexts is tracked, it has no additional restrictions:
you run vanilla Python.

When you run the code above, you see the same results as before, but also, Flor automatically
versions your code in a git repository and writes a JSON file with the name
``introduction_log.json`` to the current directory. Flor will co-mingle (mix) your commits with its own commits.
If the code you are running is already in a repository, it will use the same repository. If the code is in a directory
that is not in a repository (or is a subdirectory of a repository), Flor will initialize a git repository and show
you a prompt. We'll explain versioning in more detail later.

The JSON file looks something like this:

.. code-block:: json

    {
        "block_type": "function_body :: fit_and_score_model",
        "log_sequence": [
            {
                "block_type": "loop_body"
            },
            ...
        ]
    }

We can learn little from this particular JSON file other than the name of the function that was invoked,
and the fact that the function contains a loop that ran 50 times. Let's change that, and do some real tracking:

.. code-block:: python

    import numpy as np
    from sklearn.metrics import mean_squared_error

    import flor
    log = flor.log

    y_actual = np.array([
        0.37454012, 0.95071431,
        0.73199394, 0.59865848,
        0.15601864])

    @flor.track
    def fit_and_score_model():
        for seed in range(50):
            # We log the seed parameter
            np.random.seed(log.param(seed))
            y_pred = np.random.rand(len(y_actual))
            # We log the mse metric
            mse = log.metric(mean_squared_error(y_actual, y_pred))

    with flor.Context('introduction'):
        fit_and_score_model()

We've added two log statements. One to log the seed parameter and the other to log the mse metric.
This change produces the following log:

.. code-block:: json

    {
        "block_type": "function_body :: fit_and_score_model",
        "log_sequence": [
            {
                "block_type": "loop_body",
                "log_sequence": [
                    {
                        "assignee": null,
                        "caller": "np.random.seed",
                        "from_arg": false,
                        "in_execution": "fit_and_score_model",
                        "in_file": "/Users/rogarcia/Desktop/sandbox/randy.py",
                        "instruction_no": 16,
                        "keyword_name": null,
                        "pos": 0,
                        "runtime_value": 0,
                        "typ": "param",
                        "value": "seed"
                    },
                    {
                        "assignee": "mse",
                        "caller": null,
                        "from_arg": false,
                        "in_execution": "fit_and_score_model",
                        "in_file": "/Users/rogarcia/Desktop/sandbox/randy.py",
                        "instruction_no": 19,
                        "keyword_name": null,
                        "pos": null,
                        "runtime_value": 0.03541292928458963,
                        "typ": "metric",
                        "value": "mean_squared_error(y_actual, y_pred)"
                    }
                ]
            },
            ...
        ]
    }

From the log we learn more information than from the print statement, we're working on making these logs
queryable. For help interpreting the logs, please read its `documentation <https://flor.readthedocs.io/en/latest/log_cfg.html>`_.

**RECAP:**

1. Any code you want Flor to track should be in a Flor decorated function

2. Every Flor experiment must have a Flor context, the name of the context must be unique in your scope, and the top-level flor decorated function must be invoked from a Flor context

3. Wrap any value you want to track in a ``log.param()`` or ``log.metric()``

4. Flor automatically versions the code and results of your execution

5. Flor produces rich JSON logs, and writes them to the same directory

Exercise
--------

Using what you've learned, try to wrap the following code in Flor. Track the relevant parameters and metrics:

.. code-block:: python

    from sklearn import datasets
    from sklearn import svm
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target,
                                              test_size=0.15,
                                              random_state=430)

    clf = svm.SVC(gamma=0.001, C=100.0)
    clf.fit(X_tr, y_tr)

    score = clf.score(X_te, y_te)

More Examples
-------------

1. `**BASIC** <https://github.com/ucbrise/flor/tree/master/examples/logger>`_: See ``basic.py``. This example shows you how to track the data you read
and the models you serialize using ``log.read()`` and ``log.write()``. Additionally, it separates a model-training pipeline into multiple functions,
and demonstrates the extent to which Flor can infer dataflow and lineage in the `logs <https://github.com/ucbrise/flor/blob/master/examples/logger/basic_log.json>`_.

2. `**PYTORCH** <https://github.com/ucbrise/flor/tree/master/examples/pytorch>`_: These examples show you how PyTorch code can be wrapped in Flor.