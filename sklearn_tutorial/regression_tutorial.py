import sklearn.linear_model
import pandas as pd
import io
import requests


url = "https://raw.githubusercontent.com/gagolews/\
Analiza_danych_w_jezyku_Python/master/zbiory_danych/winequality-all.csv"
s = requests.get(url).content
wine = pd.read_csv(io.StringIO(s.decode('utf-8')), comment="#")

white_wine = wine[wine.color == "white"]
white_wine = white_wine.iloc[:, 0:11]
y = white_wine.iloc[:, -1]
X = white_wine.iloc[:, :-1]


ls = sklearn.linear_model.LinearRegression()
ls.fit(X, y)
ls.intercept_
ls.coef_
