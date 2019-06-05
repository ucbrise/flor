from flor import Flog
if Flog.flagged():
    flog = Flog(False)
Flog.flagged() and flog.write({'file_path':
    '/Users/rogarcia/sandbox/examples/basic.py', 'lsn': 0})
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model(n_estimators, X_tr, y_tr):
    from flor import Flog
    if Flog.flagged():
        flog = Flog()
    Flog.flagged() and flog.write({'file_path':
        '/Users/rogarcia/sandbox/examples/basic.py', 'lsn': 0})
    Flog.flagged() and flog.write({'start_function': 'train_model', 'lsn': 1})
    Flog.flagged() and flog.write({'lsn': 2, 'params': [{
        '0.raw.n_estimators': flog.serialize(n_estimators)}, {'1.raw.X_tr':
        flog.serialize(X_tr)}, {'2.raw.y_tr': flog.serialize(y_tr)}]})
    clf = RandomForestClassifier(n_estimators=n_estimators).fit(X_tr, y_tr)
    Flog.flagged() and flog.write({'locals': [{'clf': flog.serialize(clf)}],
        'lineage':
        'clf = RandomForestClassifier(n_estimators=n_estimators).fit(X_tr, y_tr)'
        , 'lsn': 4})
    __return__ = clf
    Flog.flagged() and flog.write({'locals': [{'__return__': flog.serialize
        (__return__)}], 'lineage': '__return__ = clf', 'lsn': 6})
    Flog.flagged() and flog.write({'end_function': 'train_model', 'lsn': 5})
    return __return__
    Flog.flagged() and flog.write({'end_function': 'train_model', 'lsn': 3})


def test_model(clf, X_te, y_te):
    from flor import Flog
    if Flog.flagged():
        flog = Flog()
    Flog.flagged() and flog.write({'file_path':
        '/Users/rogarcia/sandbox/examples/basic.py', 'lsn': 0})
    Flog.flagged() and flog.write({'start_function': 'test_model', 'lsn': 1})
    Flog.flagged() and flog.write({'lsn': 2, 'params': [{'0.raw.clf': flog.
        serialize(clf)}, {'1.raw.X_te': flog.serialize(X_te)}, {
        '2.raw.y_te': flog.serialize(y_te)}]})
    score = log.metric(clf.score(X_te, y_te))
    Flog.flagged() and flog.write({'locals': [{'score': flog.serialize(
        score)}], 'lineage': 'score = log.metric(clf.score(X_te, y_te))',
        'lsn': 4})
    Flog.flagged() and flog.write({'end_function': 'test_model', 'lsn': 3})


movie_reviews = pd.read_json('data.json')
Flog.flagged() and flog.write({'locals': [{'movie_reviews': flog.serialize(
    movie_reviews)}], 'lineage':
    'movie_reviews = pd.read_json("data.json")', 'lsn': 1})
movie_reviews['rating'] = movie_reviews['rating'].map(lambda x: 0 if x < 5 else
    1)
Flog.flagged() and flog.write({'locals': [{'movie_reviews["rating"]': flog.
    serialize(movie_reviews['rating'])}], 'lineage':
    'movie_reviews["rating"] = movie_reviews["rating"].map(lambda x: 0 if x < 5 else    1)'
    , 'lsn': 2})
X_tr, X_te, y_tr, y_te = train_test_split(movie_reviews['text'],
    movie_reviews['rating'], test_size=0.2, random_state=92)
Flog.flagged() and flog.write({'locals': [{'X_tr': flog.serialize(X_tr)}, {
    'X_te': flog.serialize(X_te)}, {'y_tr': flog.serialize(y_tr)}, {'y_te':
    flog.serialize(y_te)}], 'lineage':
    'X_tr, X_te, y_tr, y_te = train_test_split(movie_reviews["text"],    movie_reviews["rating"], test_size=0.2, random_state=92)'
    , 'lsn': 3})
vectorizer = TfidfVectorizer()
Flog.flagged() and flog.write({'locals': [{'vectorizer': flog.serialize(
    vectorizer)}], 'lineage': 'vectorizer = TfidfVectorizer()', 'lsn': 4})
vectorizer.fit(X_tr)
X_tr = vectorizer.transform(X_tr)
Flog.flagged() and flog.write({'locals': [{'X_tr': flog.serialize(X_tr)}],
    'lineage': 'X_tr = vectorizer.transform(X_tr)', 'lsn': 5})
X_te = vectorizer.transform(X_te)
Flog.flagged() and flog.write({'locals': [{'X_te': flog.serialize(X_te)}],
    'lineage': 'X_te = vectorizer.transform(X_te)', 'lsn': 6})
for i in [1, 5]:
    Flog.flagged() and flog.write({'start_loop': 31, 'lsn': 7})
    clf = train_model(i, X_tr, y_tr)
    Flog.flagged() and flog.write({'locals': [{'clf': flog.serialize(clf)}],
        'lineage': 'clf = train_model(i, X_tr, y_tr)', 'lsn': 9})
    test_model(clf, X_te, y_te)
    Flog.flagged() and flog.write({'end_loop': 31, 'lsn': 8})
