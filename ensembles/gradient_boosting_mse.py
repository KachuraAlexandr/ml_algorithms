import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """ 
        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.feature_subsample_size = feature_subsample_size
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        self.features_num = X.shape[1]
        if self.feature_subsample_size is None:
            self.feature_subsample_size = int(self.features_num / 3)
        if X_val is None or y_val is None:
            train_size = int(0.7 * X.shape[0])
            X_train = X[:train_size, :]
            y_train = y[:train_size]
            X_val = X[train_size:, :]
            y_val = y[train_size:]
        else:
            X_train = X
            y_train = y
        self.X_train = X_train
        self.y_train = y_train
        self.trees = []
        self.gamma_values = []
        loss_on_step = []
 
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, \
              max_features=self.feature_subsample_size)
            tree.fit(X_train, self.predict(X_train) - y_train)
            distribution_by_leaf = tree.apply(X_train) - 1
            gamma = np.array(list(map(lambda leaf_index: \
              minimize_scalar(lambda gamma_optim: \
              0.5 * np.sum((self.predict(X_train[distribution_by_leaf == leaf_index, :]) - \
              gamma_optim - y_train[distribution_by_leaf == leaf_index]) ** 2)).x, \
              np.unique(distribution_by_leaf))))
            self.trees.append(tree)
            self.gamma_values.append(gamma)
            loss_on_step.append(0.5 * np.sum((self.predict(X_val) - y_val) ** 2))
 
        return loss_on_step.copy()
 
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        X_train = self.X_train
        answers = np.zeros(X.shape[0])
        for tree, gamma in zip(self.trees, self.gamma_values):
            leaves_indexes = np.unique(tree.apply(X_train))
            answers -= np.apply_along_axis(lambda x: \
              gamma[leaves_indexes == tree.apply(x.reshape((1, -1)))[0]], \
              axis=1, arr=X).ravel()
        answers *= self.learning_rate
        answers -= np.mean(self.y_train)
        return answers
