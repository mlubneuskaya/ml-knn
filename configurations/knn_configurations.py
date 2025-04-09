default_configuration = {
    'clf__n_neighbors': 5,
    'clf__weights': 'uniform',
    'clf__metric': 'minkowski',
}

parameter_grid = {
    'clf__n_neighbors': list(range(1, 16)),
    'clf__weights': ['uniform', 'distance'],
    'clf__metric': ['euclidean', 'manhattan', 'minkowski'],
}