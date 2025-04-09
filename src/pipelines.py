from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from src.winsorizer import Winsorizer


def get_iris_pipeline():
    return Pipeline([
        ('normalizer', Normalizer()),
    ])

def get_wine_pipeline():
    return Pipeline([
        ('normalizer', Normalizer()),
        ('pca', PCA(n_components=.9, svd_solver='full')),
    ])

def get_bankruptcy_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('winsorizer', Winsorizer()),
        ('normalizer', Normalizer()),
        ('pca', PCA(n_components=.9, svd_solver='full')),
    ])
