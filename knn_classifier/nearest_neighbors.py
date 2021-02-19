import numpy as np
from sklearn.neighbors import NearestNeighbors
from distances import euclidean_distance, cosine_distance


class KNNClassifier:
    epsilon = 1e-5
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        if strategy in ['brute', 'kd_tree', 'ball_tree']:
            self.nn_clf = \
              NearestNeighbors(n_neighbors=k, \
              algorithm=strategy, metric=metric)
            
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if self.strategy != 'my_own':
            self.nn_clf.fit(X, y)
            
    def find_kneighbors(self, X, return_distance):
        if self.strategy == 'my_own':
            if self.metric == 'euclidean':
                distance_matrix = euclidean_distance(X, self.X_train)
            else:
                distance_matrix = cosine_distance(X, self.X_train)
            sort_indexes = np.argsort(distance_matrix, axis=1)
            distance_matrix = np.sort(distance_matrix, axis=1)
            indexes_matrix = \
              np.tile(np.arange(self.X_train.shape[0]), (X.shape[0], 1))
            for i in range(distance_matrix.shape[0]):
                indexes_matrix[i, :] = \
                  indexes_matrix[i, :][sort_indexes[i, :]]
        else:
            if return_distance:
                distance_matrix, indexes_matrix = \
                  self.nn_clf.kneighbors(X=X, return_distance=True)
            else:
                indexes_matrix = self.nn_clf.kneighbors(X=X, return_distance=False)
        if return_distance:
            return distance_matrix[:, :self.k].copy(), \
              indexes_matrix[:, :self.k].copy()
        return indexes_matrix[:, :self.k].copy()
        
    def predict(self, X):
        test_size = X.shape[0]
        test_block_size = min(test_size, self.test_block_size)
        y_test = np.zeros(X.shape[0])
        part_start = 0
        part_stop = test_block_size
        self.indexes_matrix = \
          np.zeros(X.shape[0] * self.k).reshape(X.shape[0], self.k).astype(int)
        if self.weights:
            self.distance_matrix = \
              np.zeros(X.shape[0] * self.k).reshape(X.shape[0], self.k)
            for i in range(test_size // test_block_size):
                X_part = X[part_start:part_stop, :]
                self.distance_matrix[part_start:part_stop, :], \
                  self.indexes_matrix[part_start:part_stop, :] = \
                  self.find_kneighbors(X=X_part, return_distance=True)
                part_start += test_block_size
                part_stop += test_block_size
            if test_size % test_block_size:
                part_start = test_size - test_size % test_block_size
                part_stop = test_size
                X_part = X[part_start:part_stop, :]
                self.distance_matrix[part_start:part_stop, :], \
                  self.indexes_matrix[part_start:part_stop, :] = \
                  self.find_kneighbors(X=X_part, return_distance=True)
            for j in range(test_size):
                nn_answers = self.y_train[self.indexes_matrix[j, :]]
                discriminative_function = \
                  lambda c: np.sum(np.where(nn_answers == c, \
                  np.ones(self.k) / (self.distance_matrix[j, :] + \
                  np.full(self.k, KNNClassifier.epsilon)), np.zeros(self.k)))
                y_test[j] = np.unique(nn_answers)\
                  [np.argmax(np.array([discriminative_function(c) \
                  for c in np.unique(nn_answers)]))]
        else:
            for i in range(test_size // test_block_size):
                X_part = X[part_start:part_stop, :]
                self.indexes_matrix[part_start:part_stop, :] = \
                  self.find_kneighbors(X=X_part, return_distance=False)
                part_start += test_block_size
                part_stop += test_block_size
            if test_size % test_block_size:
                part_start = test_size - test_size % test_block_size
                part_stop = test_size
                X_part = X[part_start:part_stop, :]
                self.indexes_matrix[part_start:part_stop, :] = \
                  self.find_kneighbors(X=X_part, return_distance=False)
            for j in range(test_size):
                nn_answers = self.y_train[self.indexes_matrix[j, :]]
                y_test[j] = np.unique(nn_answers)\
                  [np.argmax(np.array([np.sum(nn_answers == c) \
                  for c in np.unique(nn_answers)]))]
        return y_test.copy()
