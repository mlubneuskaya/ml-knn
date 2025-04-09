from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline

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
    return ImbalancedPipeline([
        ('sampler', RandomUnderSampler()),
        ('imputer', SimpleImputer(strategy='median')),
        ('winsorizer', Winsorizer()),
        ('normalizer', Normalizer()),
        ('pca', PCA(n_components=.9, svd_solver='full')),
    ])
