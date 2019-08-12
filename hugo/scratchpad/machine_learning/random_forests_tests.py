from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

data = datasets.load_iris()
X, y = data.data, data.target
rfc = RandomForestClassifier()
rfc.fit(X, y)

accuracy_score(rfc.predict(X), y)
