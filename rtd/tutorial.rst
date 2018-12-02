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
with the same dimensions as the `y_actual` vector. Finally, we compare `y_pred` and `y_actual` to get a measure of loss.
If we were doing optimization and tuning, we would care about which parameterization minimized the mean squared error.

