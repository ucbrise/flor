<!-- ![Travis](https://travis-ci.com/ucbrise/flor.svg?branch=master)
![Python37](https://img.shields.io/badge/python-3.7-blue.svg)
[![](https://badge.fury.io/py/pyflor.svg)](https://pypi.org/project/pyflor/)
[![codecov](https://codecov.io/gh/ucbrise/flor/branch/master/graph/badge.svg)](https://codecov.io/gh/ucbrise/flor)
 -->

FLOR: Fast Low-Overhead Recovery
================================

FLOR is a suite of machine learning tools for hindsight logging.

# What is hindsight logging?

Hindsight logging is an optimistic logging practice favored by agile model developers. Model developers log training metrics such as the loss and accuracy by default, and selectively restore additional training data --- like tensor histograms, images, and overlays --- post-hoc, if and when there is evidence of a problem.

# What tools does FLOR bundle?

1. A low-overhead background materializer. By our microbenchmarks, the background materializer cuts logging overheads by 75% on average. This tool lets you use your logger of choice in the backgroud: e.g. TensorBoard, WandB, MLFLow, and others. When your logging practice is optimistic, logging overheads are light---but if you're logging more conservatively, or hindisght logging (i.e. restoring) heavy volumes of data post-hoc, you should use this toold.

2. A periodic checkpointing library. 

3. A SkipBlock API.

4. An instrumentation library.



## License
Flor is licensed under the [Apache v2 License](https://www.apache.org/licenses/LICENSE-2.0).
