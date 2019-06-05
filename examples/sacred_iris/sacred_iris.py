from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

import torch
import numpy as np

from tensorboardX import SummaryWriter
from collections import defaultdict
writer = SummaryWriter()

from sacred import Experiment
ex = Experiment()

@ex.capture
def fit_and_score_model(gamma, C, test_size, random_state, iter, _run):

    iris = datasets.load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target,
                                                  test_size=test_size,
                                                  random_state=random_state)

    clf = svm.SVC(gamma=gamma, C=C)

    clf.fit(X_tr, y_tr)

    score = clf.score(X_te, y_te)
    _run.log_scalar('score', score, iter)

    print('Gamma: ' , gamma)
    print('Score: ' , score)

    return score

@ex.automain
def main():
    gammas = [0.01, 0.05, 0.1]
    dct = defaultdict(list)
    for i in range(5):
        for gamma in gammas:

            dct[str(gamma)] = float(fit_and_score_model(gamma=gamma, C=100.0, test_size=0.15, random_state=100, iter=i))

        writer.add_scalars('score', dct, i)