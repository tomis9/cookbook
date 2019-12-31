from sklearn.svm import SVC  # support vector classification
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target

svc = SVC()
svc.fit(X, y)

accuracy_score(svc.predict(X), y)
# sum(svc.predict(X) == y) / len(y)  # the same as accuracy_score

# 0.986667  # pretty good
