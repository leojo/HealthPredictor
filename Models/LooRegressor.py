from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator

class LooRegressor(BaseEstimator, ClassifierMixin):
    """
    LooRegressor for scikit-learn estimators.

    Parameters
    ----------

    reg : `iterable`
      A list of scikit-learn regressor objects.
    weights : `list` (default: `None`)
      If `None`, an unweighted average will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the weighted averaged of
        the predicted class labels using the given weights will be used.

    """
    def __init__(self, reg):
        self.reg = reg
        self.regs = []

    def fit(self, X, y):
        """
        Fit the model.

        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels

        """
        npX = np.array(X)
        npY = np.array(y)
        self.regs = [self.reg]*npX.shape[0]
        for i in range(len(self.regs)):
            reg = self.regs[i]
            X_i = np.concatenate((npX[:i],npX[i+1:]))
            y_i = np.concatenate((npY[:i],npY[i+1:]))
            reg.fit(X_i, y_i)

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        y : list or numpy array, shape = [n_samples]
            Average prediction of the leave-one-out model.

        """

        self.predictions_ = np.asarray([reg.predict(X) for reg in self.regs])
        y = (sum(self.predictions_)*1.)/len(self.predictions_)

        return y