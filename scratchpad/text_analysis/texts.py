import numpy as np
import re
import pandas as pd


text = """
The algorithm exists in two flavors CBOW and Skip-Gram. Given a set of
sentences the model loops on the words of each sentence and either tries
to use the current word of to predict its
neighbors, in which case the method is called Skip-Gram,
or it uses each of these contexts to predict the current word, in which case
the method is called Continuous Bag Of Words. The limit on the
number of words in each context is determined by a parameter
called window size.
"""

text = text.replace("\n", " ")
text = re.sub("[,.]", "", text)
words = text.split(" ")
words = [word for word in words if len(word) > 0]

ws = np.array(words)
us = np.unique(ws, return_counts=True)
df = pd.DataFrame(dict(word=us[0], freq=us[1]))
df.sort('freq', ascending=False)

text = "natural language processing and machine learning is fun and exciting"
text_l = text.split(" ")

l = len(text_l)
m = 2
a = [np.array(range(-m, m+1)) + i for i in range(10)]
r = [x[(x >= 0) & (x <= l-1)] for x in a]
k = dict(zip(range(l), r))
p = {x: k[x][k[x] != x] for x in k}


def f(x):
    a = np.zeros(10)
    a[x] = 1
    return a

p = {tuple(f(x)): p[x] for x in p}
