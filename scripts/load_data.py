import pandas as pd
from ucimlrepo import fetch_ucirepo


datasets = {
    'iris': 53,
    'wine': 109,
    'bankruptcy': 365
}

for title, dataset_id in datasets.items():
    dataset = fetch_ucirepo(id=dataset_id)
    pd.concat(
        [dataset.data.features, dataset.data.targets],
        axis=1
    ).to_pickle(f"../data/raw/{title}.pkl")
