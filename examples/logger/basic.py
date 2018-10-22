
import flor
from flor import log

# Import standard libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

@flor.track_execution
def main(x, y=None):
    # Load the Data
    movie_reviews = pd.read_json(log.read('data.json'))

    # Do light preprocessing (see if you can extract the lambda transform)
    # movie_reviews appears on the LHS... so it's being modified.
    # Record the modifying transformation
    # What if we just record the line itself?
    movie_reviews['rating'] = movie_reviews['rating'].map(lambda x: 0 if x < 5 else 1)

    # Do train/test split-
    X_tr, X_te, y_tr, y_te = train_test_split(movie_reviews['text'], movie_reviews['rating'],
                                              test_size=log.parameter(0.20),
                                              random_state=log.parameter(92))

    # Vectorize the English sentences
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_tr)
    X_tr = vectorizer.transform(X_tr)
    X_te = vectorizer.transform(X_te)

    # Fit the model
    for i in range(5):
        #############################
        clf = RandomForestClassifier(n_estimators=log.parameter(i)).fit(X_tr, y_tr)
        #############################

        score = log.metric(clf.score(X_te, y_te))

    print(score)

print(main(1, 20))
