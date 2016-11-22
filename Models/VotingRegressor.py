from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator

class VotingRegressor(BaseEstimator, ClassifierMixin):
    """
    VotingRegressor for scikit-learn estimators.

    Parameters
    ----------

    reg : `iterable`
      A list of scikit-learn regressor objects.
    weights : `list` (default: `None`)
      If `None`, an unweighted average will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the weighted averaged of
        the predicted class labels using the given weights will be used.

    """
    def __init__(self, regs, weights=None):
        self.regs = regs
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the scikit-learn estimators.

        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels

        """
        for reg in self.regs:
            reg.fit(X, y)

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        y : list or numpy array, shape = [n_samples]
            Weighted averages of the predictions of the estimators

        """

        self.predictions_ = np.asarray([reg.predict(X) for reg in self.regs])
        if self.weights:
            y = sum([p*w for p,w in zip(self.predictions_,self.weights)])/sum(self.weights)
        else:
            y = (sum(self.predictions_)*1.)/len(self.predictions_)

        return y