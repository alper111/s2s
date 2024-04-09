import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KernelDensity

from data import S2SDataset


__author__ = 'Steve James and George Konidaris'
# Modified by Alper Ahmetoglu. Original source:
# https://github.com/sd-james/skills-to-symbols/tree/master


class SupportVectorClassifier:
    """
    An implementation of a probabilistic classifier that
    uses support vector machines with Platt scaling.
    """

    def __init__(self, mask: list[int], probabilistic=True):
        """
        Create a new SVM classifier for preconditions

        Parameters
        ----------
        mask : list[int]
            The state variables that should be kept for classification
        probabilistic : bool, optional
            Whether the classifier is probabilistic
        """
        self._mask = mask
        self._probabilistic = probabilistic
        self._classifier: SVC | None = None

    @property
    def mask(self) -> list[int]:
        """
        Get the precondition mask
        """
        return self._mask

    def fit(self, X, y, **kwargs):
        """
        Fit the data to the classifier using a grid search
        for the hyperparameters with cross-validation

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.

        **kwargs : dict, optional
            Additional keyword arguments.
        """
        c_range = kwargs.get('precondition_c_range', np.arange(1, 16, 2))
        gamma_range = kwargs.get('precondition_gamma_range', np.arange(4, 22, 2))

        param_grid = {'gamma': gamma_range, 'C': c_range}
        grid = GridSearchCV(SVC(class_weight='balanced'), param_grid=param_grid, cv=3, n_jobs=-1)  # 3 fold CV
        data = X[:, self.mask]
        grid.fit(data, y)

        if not self._probabilistic:
            self._classifier = grid.best_estimator_  # we're done
        else:
            # we've found the best hyperparams. Now do it again with Platt scaling turned on
            params = grid.best_params_
            print("Found best SVM hyperparams: C = {}, gamma = {}".format(params['C'], params['gamma']))
            # Now do Platt scaling with the optimal parameters
            self._classifier = SVC(probability=True, class_weight='balanced', C=params['C'], gamma=params['gamma'])
            self._classifier.fit(data, y)
            print("Classifier score: {}".format(self._classifier.score(data, y)))

    def probability(self, states: np.ndarray) -> float:
        """
        Compute the probability of the state given the learned classifier.

        Parameters
        ----------
        states : np.ndarray
            The states.

        Returns
        -------
        float
            The probability of the state according to the classifier.
        """
        assert isinstance(self._classifier, SVC), "Classifier not trained yet"
        if states.ndim == 1:
            states = states.reshape(1, -1)

        masked_states = states[:, self.mask]
        if self._probabilistic:
            return np.mean(self._classifier.predict_proba(masked_states)[0][1])
        else:
            return self._classifier.predict(masked_states)[0]


class KernelDensityEstimator:
    """
    A density estimator that models a distribution over low-level states.
    """

    def __init__(self, mask: list[int]):
        """
        Initialize a new estimator.

        Parameters
        ----------
        mask : list[int]
            The state variables we care about.
        """
        self._mask = mask
        self._kde: KernelDensity | None = None

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """
        Fit the data to the effect estimator using a grid search for the hyperparameters with cross-validation.

        Parameters:
        ----------
        X : np.ndarray
            The data.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        -------
        None
        """
        if kwargs.get('masked', False):
            data = X  # already been masked
        else:
            data = X[:, self.mask]
        bandwidth_range = kwargs.get('effect_bandwidth_range', np.arange(0.001, 0.1, 0.001))
        params = {'bandwidth': bandwidth_range}
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=3)
        grid.fit(data)
        print("Best bandwidth hyperparameter: {}".format(grid.best_params_['bandwidth']))
        self._kde = grid.best_estimator_

    @property
    def mask(self) -> list[int]:
        """
        Get the effect mask.
        """
        return self._mask

    def sample(self, n_samples=100) -> np.ndarray:
        """
        Sample data from the density estimator.

        Parameters:
        ----------
        n_samples : int, optional
            The number of samples to generate. Default is 100.

        Returns:
        -------
        data : np.ndarray
            An array of size [n_samples, len(mask)] containing the sampled data.
        """
        assert isinstance(self._kde, KernelDensity), "Estimator not trained yet"
        data = self._kde.sample(n_samples)
        if data.ndim == 1:  # ensure always shape of (N X D)
            return np.reshape(data, (data.shape[0], 1))
        return data

    def integrate_out(self, variable_list: list[int], **kwargs) -> 'KernelDensityEstimator':
        """
        Integrate out the given variables from the distribution.

        Parameters:
        ----------
        variable_list : list[int]
            A list of variables to be marginalized out from the distribution.

        Returns:
        -------
        KernelDensityEstimator
            A new distribution equal to the original distribution with the specified variables marginalized out.
        """
        variable_list = sorted(variable_list)  # make sure it's always sorted to prevent bugs!

        new_vars = []
        new_indices = []

        # find all the other variables in the mask except what's given
        for pos, val in enumerate(self.mask):  # TODO probably a better way of doing this in numpy
            if val not in variable_list:
                new_vars.append(val)
                new_indices.append(pos)
        n_samples = kwargs.get('estimator_samples', 100)
        new_samples = self.sample(n_samples)[:, new_indices]
        kde = KernelDensityEstimator(mask=new_vars)
        kwargs['masked'] = True  # the data has already been masked
        kde.fit(new_samples, **kwargs)
        return kde


def learn_preconditions(subgoals: dict[tuple[int, int], S2SDataset]) -> dict[tuple[int, int], SupportVectorClassifier]:
    raise NotImplementedError
