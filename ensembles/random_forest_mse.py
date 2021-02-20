import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
 
 
class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
            
        y_val : numpy ndarray
            Array of size n_val_objects           
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = 1 / 3
        if X_val is None or y_val is None:
            train_size = int(0.7 * X.shape[0])
            X_train = X[:train_size, :]
            y_train = y[:train_size]
            X_val = X[train_size:, :]
            y_val = y[train_size:]
        else:
            X_train = X
            y_train = y
        train_sample_size = X_train.shape[0]
        samples_indices = np.random.randint(train_sample_size, \
          size=(train_sample_size, self.n_estimators)).astype('int')
        self.base_models = []
        base_models = self.base_models
        max_depth = self.max_depth
        feature_subsample_size = self.feature_subsample_size
        loss_on_step = []
 
        for j in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=max_depth, \
              max_features=feature_subsample_size)
            tree.fit(X_train[samples_indices[j, :], :], \
              y_train[samples_indices[j, :]])
            base_models.append(tree)
            loss_on_step.append(0.5 * np.sum((self.predict(X_val) - y_val) \
              ** 2))    
    
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
        y = np.zeros(X.shape[0])
        for tree in self.base_models:
            y += tree.predict(X)
        y /= len(self.base_models)
        return y
