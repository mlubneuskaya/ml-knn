from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_iris_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
    ])


def get_wine_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
    ])
