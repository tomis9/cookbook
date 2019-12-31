import featuretools
import pandas as pd

from sklearn.datasets import load_boston

boston_raw = load_boston()

boston = pd.DataFrame(boston_raw.data, columns=boston_raw.feature_names)
