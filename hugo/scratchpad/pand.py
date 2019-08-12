import pandas as pd
import re
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


def prepare_iris_as_in_r():
    iris_raw = load_iris()
    colnames = [re.sub(' ', '_', re.sub(' \(cm\)', '', x))
                for x in iris_raw.feature_names]
    iris = pd.DataFrame(iris_raw.data, columns=colnames)
    species = pd.DataFrame({'species_index': range(3),
                            'species': iris_raw.target_names})
    iris['species_index'] = iris_raw.target
    iris = pd.merge(iris, species, on='species_index')
    iris.drop('species_index', axis=1, inplace=True)
    return iris


iris = prepare_iris_as_in_r()

iris.groupby('species').size()

iris.plot.line()
plt.show()

11

# 0 reading data from a file

pd.read_csv()

useful parameters:

* `sep`

* `delimiter`

* `header`

# 1 filtering, removing na values

iris[iris.species == "setosa"]
iris[~iris.sepal_length.isna()]

# 2 selecting

iris[['species', 'sepal_width']]

# proper ways of selection + filtering

# `loc` - you may use ranges as well as names
iris.loc[:10, ['species', 'sepal_length']]

# `iloc` - you can use unly ranges
iris.iloc[:10, :3]

11

# 3 aggregating

iris.groupby('species').agg({'sepal_length': sum, 'petal_length': np.median})

# 4 joining


# plotting

iris.groupby('species').size().plot.bar()
plt.show()

plt.scatter(iris.sepal_length, iris.sepal_width)

# ordering

# pivot table


# indexes

# You may have noticed that pandas uses indexes extensively, which may be not
# very intuitive if you come with R or SQL background (especially that index
# in pandas is means something different than in SQL, which *is* misleading).
# In general the easiest way to cope with their problematic nature is trying
# to avoid them.

