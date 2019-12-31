from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

from sklearn.svm import SVC  # support vector classification

svc = SVC()
svc.fit(X_train, y_train)

print(accuracy_score(svc.predict(X_test), y_test)) # pretty good

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

print(accuracy_score(y_test, dtc.predict(X_test)))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

accuracy_score(rfc.predict(X_test), y_test)

