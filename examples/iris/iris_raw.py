from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

from flor import OpenLog

import time
start_time = time.time()

# with OpenLog('iris_raw'):
iris = datasets.load_iris()
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.15, random_state=430)

with OpenLog('iris_raw'):
    clf = svm.SVC(gamma=0.001, C=100.0)
    clf.fit(X_tr, y_tr)

    print(clf.score(X_te, y_te))

print("--- %s seconds ---" % (time.time() - start_time))
