import numpy as np


def log_distance(distances):
    return 1 / np.log(distances + 1e-10)


def robust_weights(distances):
    return np.exp(-distances ** 2 + 1e-10)
