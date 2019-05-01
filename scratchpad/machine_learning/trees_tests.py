from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target

dtc = DecisionTreeClassifier()
dtc.fit(X, y)

accuracy_score(y, dtc.predict(X))  # clearly overfitted
