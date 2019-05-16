from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target

knn = KNeighborsClassifier()
knn.fit(X, y)
accuracy_score(y, knn.predict(X))
