import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, X, y=None):
        X_ = pd.DataFrame(X)
        q1 = X_.quantile(0.25)
        q3 = X_.quantile(0.75)
        self.lower_bound = q3 - 1.5 * q1
        self.upper_bound = q3 + 1.5 * q3
        return self

    def transform(self, X, y=None):
        X_ = pd.DataFrame(X)
        return np.array(X_.clip(lower=self.lower_bound, upper=self.upper_bound, axis=1))
