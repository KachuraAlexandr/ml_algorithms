import numpy as np
from sklearn.neighbors import NearestNeighbors
from distances import euclidean_distance, cosine_distance

def kfold(n, n_folds):
    folds = []
    indexes = list(range(n))
    validation_start = 0
    first_size_folds_num = n % n_folds
    first_fold_size = n // n_folds + 1
    second_fold_size = n // n_folds
    for i in range(n_folds):
        validation_indexes = list(range(validation_start, \
          validation_start + first_fold_size)) \
          if i < first_size_folds_num \
          else list(range(validation_start, \
          validation_start + second_fold_size))
        train_indexes = [index for index in indexes \
          if index not in validation_indexes]
        folds.append((np.array(train_indexes), \
          np.array(validation_indexes)))
    return folds.copy()

def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):
    if cv is None:
        cv = kfold(X.shape[0], 3)
    if score == 'accuracy':
        cv_dict = {}
        clf_args = kwargs
        clf_args['k'] = k_list[-1]
        clf = KNNClassifier(**clf_args)
        weights = clf.weights
        score_list = []
        indexes_matrixes_list = []
        if weights:
            distance_matrixes_list = []
        for partition in cv:
            validation_size = X[partition[1]].shape[0]
            clf.fit(X[partition[0]], y[partition[0]])
            score_list.append\
              (np.sum(clf.predict(X[partition[1]]) == \
              y[partition[1]]) / validation_size)
            indexes_matrixes_list.append(clf.indexes_matrix)
            if weights:
                distance_matrixes_list.append(clf.distance_matrix)
        cv_dict.update({k_list[-1]: np.array(score_list)})
        y_train = clf.y_train
        epsilon = KNNClassifier.epsilon
        for kn in k_list[-2::-1]:
            clf_args['k'] = kn
            clf = KNNClassifier(**clf_args)
            partition_index = 0
            score_list = []
            for partition in cv:
                validation_size = X[partition[1]].shape[0]
                clf.fit(X[partition[0]], y[partition[0]])
                indexes_matrix = \
                  indexes_matrixes_list[partition_index][:, :kn]
                if weights:
                    distance_matrix = \
                      distance_matrixes_list[partition_index][:, :kn]
                y_validation = np.zeros(validation_size).astype(int)
                if weights:
                    for j in range(validation_size):
                        nn_answers = y_train[indexes_matrix[j, :]]
                        discriminative_function = \
                          lambda c: np.sum(np.where(nn_answers == c, \
                          np.ones(kn) / (distance_matrix[j, :] + \
                          np.full(kn, epsilon)), \
                          np.zeros(kn)))
                        y_validation[j] = np.unique(nn_answers)\
                          [np.argmax(np.array([discriminative_function(c) \
                          for c in np.unique(nn_answers)]))]
                else:
                    for j in range(validation_size):
                        nn_answers = clf.y_train[indexes_matrix[j, :]]
                        y_validation[j] = np.unique(nn_answers)\
                          [np.argmax(np.array([np.sum(nn_answers == c) \
                          for c in np.unique(nn_answers)]))]
                score_list.append\
                  (np.sum(y_validation == y[partition[1]]) / validation_size)
                partition_index += 1
            cv_dict.update({kn: np.array(score_list)})
        return cv_dict.copy()