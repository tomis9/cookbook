from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

rfc = RandomForestClassifier()
data = load_iris()

X, y = data.data, data.target

rfc.fit(X, y)

sc = accuracy_score(rfc.predict(X), y)
print(sc)


