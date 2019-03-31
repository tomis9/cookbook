# https://www.datacamp.com/community/tutorials/xgboost-in-python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import lime
import lime.lime_tabular


boston = load_boston()
print(boston.keys())
print(boston.DESCR)


data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data.info()
data.describe().transpose()
data.head()

boston.target

data.hist()
plt.show()

data['PRICE'] = boston.target
data.PRICE.hist()
plt.show()

# https://learning.oreilly.com/library/view/feature-engineering-for/9781491953235/

X, y = data.iloc[:, :-1], data.iloc[:, -1]
data_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=123)

lm = LinearRegression()
lm.fit(X_train, y_train)
lm_rmse = np.sqrt(mean_squared_error(lm.predict(X_test), y_test))

xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3,
                          learning_rate=0.1, max_depth=5, alpha=10,
                          n_estimators=10)

xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))


explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=boston.feature_names,
    class_names=['price'],
    verbose=True, mode='regression')
# https://pythondata.com/local-interpretable-model-agnostic-explanations-lime-python/
