from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from flor import OpenLog

import time
start_time = time.time()


with OpenLog('iris_raw', depth_limit=2):
    from flor import Flog
    if Flog.flagged():
        flog = Flog()
    Flog.flagged() and flog.write({'file_path':
        '/Users/rogarcia/git/flor/examples/iris/iris_raw.py'})

    iris = datasets.load_iris()
    Flog.flagged() and flog.write({'locals': [{'iris': flog.serialize(iris)
        }], 'lineage': 'iris = datasets.load_iris()'})
    X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target,
        test_size=0.15, random_state=430)
    Flog.flagged() and flog.write({'locals': [{'X_tr': flog.serialize(X_tr)
        }, {'X_te': flog.serialize(X_te)}, {'y_tr': flog.serialize(y_tr)},
        {'y_te': flog.serialize(y_te)}], 'lineage':
        'X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.15, random_state=430)'
        })
    clf = svm.SVC(gamma=0.001, C=100.0)
    Flog.flagged() and flog.write({'locals': [{'clf': flog.serialize(clf)}],
        'lineage': 'clf = svm.SVC(gamma=0.001, C=100.0)'})
    clf.fit(X_tr, y_tr)
    score = clf.score(X_te, y_te)
    Flog.flagged() and flog.write({'locals': [{'score': flog.serialize(score)}], 'lineage': 'score = clf.score(X_te, y_te)'})
    print(score)


print('--- %s seconds ---' % (time.time() - start_time))
