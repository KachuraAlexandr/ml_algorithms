import numpy as np


def euclidean_distance(X, Y):
    X_norm_sqr = np.sum(X ** 2, axis=1)
    Y_norm_sqr = np.sum(Y ** 2, axis=1)
    return np.sqrt(X_norm_sqr[:, np.newaxis] - \
      2 * np.dot(X, Y.T) + Y_norm_sqr)

def cosine_distance(X, Y):
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    return 1 - np.dot(X, Y.T) / (X_norm[:, np.newaxis] * Y_norm)
