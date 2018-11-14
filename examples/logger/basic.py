import flor
log = flor.log

# Import standard libraries
import pandas as pd

import cloudpickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from .empty.nothing import foo

@flor.track_execution
def train_model(n_estimators, X_tr, y_tr):
    clf = RandomForestClassifier(n_estimators=log.parameter(n_estimators)).fit(X_tr, y_tr)
    with open(log.write('clf.pkl'), 'wb') as classifier:
        cloudpickle.dump(clf, classifier)
    return clf

@flor.track_execution
def test_model(clf, X_te, y_te):
    score = log.metric(clf.score(X_te, y_te))

@flor.track_execution
def main(x, y, z):
    # Load the Data
    movie_reviews = pd.read_json(log.read('data.json'))

    movie_reviews['rating'] = movie_reviews['rating'].map(lambda x: 0 if x < z else 1)

    # Do train/test split-
    # TODO: With parser, insert code here to get the name of positional args of train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(movie_reviews['text'], movie_reviews['rating'],
                                              test_size=log.parameter(x),
                                              random_state=log.parameter(y))

    # Vectorize the English sentences
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_tr)
    X_tr = vectorizer.transform(X_tr)
    X_te = vectorizer.transform(X_te)

    # Fit the model
    for i in [1, 5]:
        clf = train_model(i, X_tr, y_tr)
        test_model(clf, X_te, y_te)

with flor.Context('basic'):
    main(0.2, 92, 5)
