The Flor API
=================

Starting or Continuing an Experiment
------------------------------------

A Flor Experiment is an object from which we can declare *Action*, *Artifact*, and *Literal*
members, and thus define the experiment’s dependency graph.

.. code-block:: python

	with flor.Experiment('NAME_AS_STRING') as ex:
		# Your experiment is defined in this context

.. autoclass:: flor.experiment.Experiment