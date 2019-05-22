---
title: "Useful_processing"
date: 2019-05-17T15:44:38+02:00
draft: true
categories: []
tags: []
---

# What is useful processing?

* many machine learning algorithms require the same kinds of data preprocessing in order for them to work properly. In other words, this sort of processing is useful.

# Examples

* one-hot encoding

```{r}
dt_iris <- data.table::as.data.table(iris)
mltools::one_hot(dt_iris)

# dplyr is not that clever
library(dplyr)
iris %>%
  mutate("Species_setosa" = ifelse(Species == "setosa", 1, 0)) %>%
  mutate("Species_virgninica" = ifelse(Species == "virgninica", 1, 0)) %>%
  mutate("Species_versicolor" = ifelse(Species == "versicolor", 1, 0)) %>%
  head()
```

```{python}
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets

data = datasets.load_iris()
y = [data.target_names[i] for i in data.target]

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
print(integer_encoded[:5])
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded[:5])
```

* scaling

* dividing your dataset test, train and validation subsets
