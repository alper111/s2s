import logging
import pickle
from typing import Callable, Optional, Union
from collections import defaultdict
import copy
import os
import heapq

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
import torch
from torch.nn.utils.rnn import pad_sequence

from s2s.helpers import dict_to_transition

__author__ = 'Steve James and George Konidaris'
# Modified by Alper Ahmetoglu. Original source:
# https://github.com/sd-james/skills-to-symbols/tree/master
logger = logging.getLogger(__name__)


class Factor:
    """
    A factor is a minimal set of state variables that can be changed by an option.
    """
    def __init__(self, indices: list[int]):
        """
        Create a new factor.

        Parameters
        ----------
        indices : list[int]
            The indices of the state variables that make up the factor.

        Returns
        -------
        None
        """
        self._indices = indices

    @property
    def variables(self) -> list[int]:
        return self._indices

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Factor({self._indices})"

    def __hash__(self):
        return hash(tuple(sorted(self.variables)))


class S2SDataset:
    """
    A dataset of state-to-state transitions. Each sample consists of a state, an option, a reward,
    the next state, and a mask that indicates which variables have changed between the state and the
    next state. The dataset can be object-factored or not.
    """
    def __init__(self, state: np.ndarray, option: np.ndarray, reward: np.ndarray,
                 next_state: np.ndarray, mask: np.ndarray,
                 factors: Optional[list[Factor]] = None):
        """
        Create a new dataset.

        Parameters
        ----------
        state : np.ndarray
            The state of the environment.
        option : np.ndarray
            The option that was executed.
        reward : np.ndarray
            The reward received.
        next_state : np.ndarray
            The next state of the environment.
        mask : np.ndarray
            The mask that indicates which variables have changed.

        Returns
        -------
        None
        """
        self.state = state
        self.option = option
        self.reward = reward
        self.next_state = next_state
        self.mask = mask
        self._factors = factors

    @property
    def factors(self):
        return self._factors

    @factors.setter
    def factors(self, factors):
        self._factors = factors

    @property
    def is_object_factored(self) -> bool:
        return self.state.ndim == 3

    @property
    def n_objects(self) -> int:
        return self.state.shape[1] if self.is_object_factored else 0

    def __len__(self):
        return len(self.state)

    def __getitem__(self, item):
        return self.state[item], self.option[item], self.reward[item], self.next_state[item], self.mask[item]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self.state.ndim == 2:
            n_factor = len(self.factors) if self.factors is not None else "unset"
            return f"S2SDataset(n_sample={len(self)}, n_feature={self.state.shape[1]}, n_factor={n_factor})"
        n_factor = sum([len(obj) for obj in self.factors]) if self.factors is not None else "unset"
        return f"S2SDataset(n_sample={len(self)}, n_object={self.state.shape[1]}, " + \
               f"n_feature={self.state.shape[2]}, n_factor={n_factor})"


class UnorderedDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder: str, transform_action: bool = True, privileged: bool = False):
        self._root_folder = root_folder
        self._privileged = privileged
        if privileged:
            self._state = np.load(os.path.join(root_folder, "priv_state.npy"), allow_pickle=True)
            self._next_state = np.load(os.path.join(root_folder, "priv_next_state.npy"), allow_pickle=True)
        else:
            self._state = np.load(os.path.join(root_folder, "state.npy"), allow_pickle=True)
            self._next_state = np.load(os.path.join(root_folder, "next_state.npy"), allow_pickle=True)
        self._action = pickle.load(open(os.path.join(root_folder, "action.pkl"), "rb"))
        self._transform_action = transform_action

    def __len__(self):
        return len(self._state)

    def __getitem__(self, idx):
        x, x_, key_order = dict_to_transition(self._state[idx], self._next_state[idx])
        if self._transform_action:
            a = self._actions_to_label(self._action[idx], key_order)
        else:
            a = self._action[idx]
        return x, a, x_

    def sample(self, n_samples):
        idx = np.random.choice(len(self), n_samples, replace=True)
        batch = [self[i] for i in idx]
        return self.collate_fn(batch)

    @staticmethod
    def _actions_to_label(action, key_order):
        return NotImplementedError

    @staticmethod
    def collate_fn(batch):
        x, a, x_ = batch[0]
        keys = list(x.keys())
        s = {k: [] for k in keys}
        s_ = {k: [] for k in keys}
        s["masks"] = {k: [] for k in keys}
        s_["masks"] = {k: [] for k in keys}
        a = []
        for x, a_, x_ in batch:
            for k in keys:
                s[k].append(x[k])
                s_[k].append(x_[k])
                s["masks"][k].append(torch.ones(x[k].shape[0], dtype=torch.bool))
                s_["masks"][k].append(torch.ones(x_[k].shape[0], dtype=torch.bool))
            a.append(a_)
        for k in keys:
            s[k] = pad_sequence(s[k], batch_first=True)
            s_[k] = pad_sequence(s_[k], batch_first=True)
            s["masks"][k] = pad_sequence(s["masks"][k], batch_first=True)
            s_["masks"][k] = pad_sequence(s_["masks"][k], batch_first=True)
        if isinstance(a[0], torch.Tensor):
            if a[0].ndim == 0:
                a = torch.tensor(a)
            else:
                a = pad_sequence(a, batch_first=True)
        return s, a, s_


class SupportVectorClassifier:
    """
    An implementation of a probabilistic classifier that
    uses support vector machines with Platt scaling.
    """

    def __init__(self, factors: list[Factor], probabilistic=True):
        """
        Create a new SVM classifier for preconditions

        Parameters
        ----------
        factor : list[Factor]
            The factors that the classifier models
        probabilistic : bool, optional
            Whether the classifier is probabilistic
        """
        self._factors = factors
        self._probabilistic = probabilistic
        self._classifier: SVC | None = None

    @property
    def factors(self) -> list[Factor]:
        """
        Get the precondition mask
        """
        return self._factors

    @property
    def variables(self) -> list[int]:
        variables = []
        for f in self.factors:
            variables.extend(f.variables)
        return variables

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
        c_range = kwargs.get('precondition_c_range', np.logspace(-2, 2, 10))
        gamma_range = kwargs.get('precondition_gamma_range', np.logspace(-2, 2, 10))

        param_grid = {'gamma': gamma_range, 'C': c_range}
        grid = GridSearchCV(SVC(class_weight='balanced'), param_grid=param_grid, cv=3, n_jobs=-1)  # 3 fold CV
        if kwargs.get('masked', False):
            data = X
        else:
            data = X[:, self.variables]
        grid.fit(data, y)

        if not self._probabilistic:
            self._classifier = grid.best_estimator_  # we're done
        else:
            # we've found the best hyperparams. Now do it again with Platt scaling turned on
            params = grid.best_params_
            # Now do Platt scaling with the optimal parameters
            self._classifier = SVC(probability=True, class_weight='balanced', C=params['C'], gamma=params['gamma'])
            self._classifier.fit(data, y)

    def probability(self, states: np.ndarray, masked=False) -> float:
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

        if masked:
            masked_states = states
        else:
            masked_states = states[:, self.variables]
        if self._probabilistic:
            return self._classifier.predict_proba(masked_states)[:, 1]
        else:
            return self._classifier.predict(masked_states)


class DensityEstimator:
    """
    Density estimator abstract class that models a distribution over
    one or more factors (a set of low-level states).
    """

    def __init__(self, factors: list[Factor]):
        self._factors = factors

    def fit(self, X: np.ndarray, **kwargs) -> None:
        pass

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        pass

    @property
    def factors(self) -> list[Factor]:
        return self._factors

    @property
    def variables(self) -> list[int]:
        variables = []
        for f in self.factors:
            variables.extend(f.variables)
        return variables

    @property
    def factor_indices(self) -> dict[Factor, int]:
        indices = {}
        it = 0
        for f in self.factors:
            n_vars = len(f.variables)
            rng = list(range(it, n_vars))
            indices[f] = rng
            it += len(f)
        return indices

    def sample(self, n_samples=100) -> np.ndarray:
        pass


class KernelDensityEstimator(DensityEstimator):
    """
    Kernel density estimator that models a distribution over one or more
    factors (a set of low-level states).
    """

    def __init__(self, factors: list[Factor]):
        """
        Initialize a new estimator.

        Parameters
        ----------
        factors : list[Factor]
            The factors that the estimator models.

        Returns
        -------
        None
        """
        super().__init__(factors)
        self._kde: Optional[KernelDensity] = None

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """
        Fit the data to the effect estimator using a grid search for the hyperparameters with cross-validation.

        Parameters
        ----------
        X : np.ndarray
            The data.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        None
        """
        if kwargs.get('masked', False):
            data = X  # already been masked
        else:
            data = X[:, self.variables]
        bandwidth_range = kwargs.get('effect_bandwidth_range', np.logspace(-3, 1, 20))
        params = {'bandwidth': bandwidth_range}
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=3)
        grid.fit(data)
        # logger.debug("Best bandwidth hyperparameter: {}".format(grid.best_params_['bandwidth']))
        self._kde = grid.best_estimator_

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the log-likelihood of the samples.

        Parameters
        ----------
        X : np.ndarray
            The samples.

        Returns
        -------
        log_prob : np.ndarray
            The log-likelihood of the samples.
        """
        return self._kde.score_samples(X)

    def sample(self, n_samples=100) -> np.ndarray:
        """
        Sample data from the density estimator.

        Parameters
        ----------
        n_samples : int, optional
            The number of samples to generate. Default is 100.

        Returns
        -------
        data : np.ndarray
            An array of size [n_samples, len(mask)] containing the sampled data.
        """
        assert isinstance(self._kde, KernelDensity), "Estimator not trained yet"
        data = self._kde.sample(n_samples)
        if data.ndim == 1:  # ensure always shape of (N X D)
            data = np.reshape(data, (data.shape[0], 1))
        return data


class KNNDensityEstimator(DensityEstimator):
    """
    A density estimator that models a distribution over one or more
    factors (a set of low-level states) with a k-nearest neighbors approach.
    """

    def __init__(self, factors: list[Factor], k: int = 5):
        """
        Initialize a new estimator.

        Parameters
        ----------
        factors : list[Factor]
            The factors that the estimator models.
        k : int, optional
            The number of nearest neighbors to consider. Default is 5.

        Returns
        -------
        None
        """
        super().__init__(factors)
        self._samples: Optional[np.ndarray] = None
        self._k = k

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """
        Fit the data to the effect estimator.

        Parameters
        ----------
        X : np.ndarray
            The data.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        None
        """
        if kwargs.get("masked", False):
            self._samples = X
        else:
            self._samples = X[:, self.variables]

    def score_samples(self, X: np.ndarray, max_samples_used: int = 100) -> np.ndarray:
        """
        Compute the log-likelihood of the samples.

        Parameters
        ----------
        X : np.ndarray
            The samples.

        Returns
        -------
        log_prob : np.ndarray
            The log-likelihood of the samples.
        """
        if X.ndim == 1:
            X = np.reshape(X, (1, -1))
        max_samples_used = min(max_samples_used, self._samples.shape[0])
        idx = np.random.choice(self._samples.shape[0], max_samples_used, replace=False)
        dists = cdist(X, self._samples[idx])
        knn_dists = np.sort(dists, axis=1)[:, :self._k]
        log_prob = -np.mean(knn_dists, axis=1)
        return log_prob

    def sample(self, n_samples=100) -> np.ndarray:
        """
        Sample data from the density estimator.

        Parameters
        ----------
        n_samples : int, optional
            The number of samples to generate. Default is 100.

        Returns
        -------
        data : np.ndarray
            An array of size [n_samples, len(mask)] containing the sampled data.
        """
        assert isinstance(self._samples, np.ndarray), "Estimator not trained yet"
        idx = np.random.choice(self._samples.shape[0], n_samples, replace=True)
        data = self._samples[idx]
        if data.ndim == 1:  # ensure always shape of (N X D)
            data = np.reshape(data, (data.shape[0], 1))
        return data


class Proposition:
    """
    A predicate over one or more factors.
    """

    def __init__(self, idx: int, name: str, estimator: Optional[DensityEstimator],
                 parameters: Optional[list[tuple[str, str]]] = None):
        """
        Create a new predicate.

        Parameters
        ----------
        idx : int
            The index of the predicate.
        name : str
            The name of the predicate.
        estimator : DensityEstimator
            The density estimator that models the predicate.
        parameters : list[(str, str)], optional
            A list of (name, type) tuples that represent the
            parameters of the predicate. The type is None if the
            argument does not have a type.

        Returns
        -------
        None
        """
        self._idx = idx
        self._name = name
        self._estimator = estimator
        self._parameters = parameters
        self.sign = 1  # whether true or the negation of the predicate

    @property
    def estimator(self) -> Optional[DensityEstimator]:
        return self._estimator

    @property
    def name(self) -> str:
        return self._name

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def parameters(self) -> list[(str, str)]:
        return self._parameters

    @property
    def factors(self) -> list[Factor]:
        assert isinstance(self.estimator, DensityEstimator)
        return self.estimator.factors

    @property
    def variables(self) -> list[int]:
        assert isinstance(self.estimator, DensityEstimator)
        return self.estimator.variables

    def sample(self, n_samples) -> np.ndarray:
        assert isinstance(self.estimator, DensityEstimator)
        return self.estimator.sample(n_samples)

    def is_grounded(self) -> bool:
        return self.parameters is None

    def is_independent(self) -> bool:
        return len(self.factors) == 1

    def negate(self) -> 'Proposition':
        """"
        Creates a negated copy of the predicate.
        """
        clone = copy.copy(self)
        clone.sign *= -1
        return clone

    def substitute(self, parameters: list[tuple[str, str]]) -> 'Proposition':
        """
        Substitute the parameters of the predicate with the given parameters.

        Parameters
        ----------
        parameters : list[tuple[str, str]]
            A list of (name, type) tuples that represent the
            parameters of the predicate. The type is None if the
            argument does not have a type.

        Returns
        -------
        new_prop : Proposition
            The new proposition with the substituted parameters.
        """
        assert self.parameters is not None, "Predicate has no parameters."
        clone = copy.copy(self)
        clone._parameters = parameters
        return clone

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        if self.parameters is not None:
            prop_str = f"{self.name}"
            for p in self.parameters:
                prop_str += f" ?{p[0]}"
                if p[1] is not None:
                    prop_str += f" - {p[1]}"
        else:
            prop_str = self.name

        if self.sign < 0:
            prop_str = f"not ({prop_str})"
        return prop_str

    def __hash__(self):
        idx = (self._idx,)
        if self._parameters is not None:
            idx += tuple(self._parameters)
        return hash(idx)

    def __eq__(self, other):
        return hash(self) == hash(other)

    @staticmethod
    def not_failed():
        return Proposition(-2, "notfailed", None)

    @staticmethod
    def empty():
        return Proposition(-1, "empty", None)


class UniquePredicateList:
    """
    A class that wraps a list. The user add density estimators to the list,
    and they are wrapped in PDDL predicates.  The list automatically deals
    with duplicates.
    """

    def __init__(self, comparator: Optional[Callable[[DensityEstimator, DensityEstimator], bool]] = None,
                 density_type="knn"):
        """
        Create a list data structure that ensures no duplicates are added to the list.

        Parameters
        ----------
        comparator : callable, optional
            A function that accepts two objects and returns whether they are equal.
        """
        self._comparator = comparator if comparator is not None else lambda x, y: x is y
        self._density_type = density_type
        self._list = []
        self._projections: list[dict[int, Optional[int]]] = []
        self.mutex_groups = None
        self.factors = None
        self.__idx = 0

    @property
    def density_type(self):
        return self._density_type

    def append(self, data: np.ndarray, factors: list[Factor], parameters: list[tuple[str, str]] = None,
               masked: bool = False, forced: bool = False) -> Proposition:
        """
        Adds a predicate to the predicate list. If the predicate covers multiple factors,
        all possible combinations of projections are added to the vocabulary as well.

        Parameters
        ----------
        data : np.ndarray
            The data on which predicates are fit.
        factors : list[Factor]
            Factors to be modeled.
        parameters : list[tuple[str, str]], optional
            A list of (name, type) tuples that represent the
            parameters of the predicate. The type is None if the
            argument
        masked : bool, optional
            Whether the data is already masked or not.
        forced : bool, optional
            Whether to force the addition of the predicate to
            the vocabulary without checking for duplicates.

        Returns
        -------
        base_predicate : Proposition
            The newly created predicate for the given set of factors.
        """
        if self.density_type == "kde":
            item = KernelDensityEstimator(factors)
        elif self.density_type == "knn":
            item = KNNDensityEstimator(factors)
        item.fit(data, masked=masked)

        # add all possible projections here.
        # (new_estimator, projected_factor, parent_idx)
        add_queue = [(item, None, None)]
        base_predicate = None
        size_before = len(self)
        while len(add_queue) > 0:
            estimator, p_factor, parent_idx = add_queue.pop(0)
            # TODO: KD-Tree-like hash function on the estimator?
            idx = self._get_or_none(estimator, parameters)
            if (idx != -1) and (not forced):
                predicate = self._list[idx]
                if self.density_type == "knn":
                    # add new samples to the existing estimator
                    new_set = np.concatenate((predicate.estimator._samples, estimator._samples), axis=0)
                    predicate.estimator.fit(new_set, masked=True)
                if parameters is not None:
                    predicate = predicate.substitute(parameters)
            else:
                # create a new predicate for this estimator and add it to the vocabulary
                idx = len(self._list)
                predicate = Proposition(idx, f'symbol_{idx}', estimator, parameters)
                self._list.append(predicate)
                self._projections.append({})
                # if there are remaining factors to be projected, add them to the queue
                if not predicate.is_independent():
                    for f in predicate.factors:
                        rem_factors = [fac for fac in predicate.factors if fac != f]
                        if self._density_type == "kde":
                            child_item = KernelDensityEstimator(rem_factors)
                        elif self._density_type == "knn":
                            child_item = KNNDensityEstimator(rem_factors)
                        child_item.fit(data)
                        add_queue.append((child_item, f, idx))

            # if this estimator is from a previous projection, set it.
            if p_factor is not None:
                self._projections[parent_idx][p_factor] = idx

            if base_predicate is None:
                base_predicate = predicate
        size_after = len(self)
        logger.debug(f"Vocabulary size={len(self)}; {size_after-size_before} new predicates.")

        return base_predicate

    def add_projection(self, symbol: Proposition, factor: Factor, projection: Proposition) -> None:
        """
        Add a projection to the vocabulary.

        Parameters
        ----------
        symbol : Proposition
            The original symbol.
        factor : Factor
            The factor that is projected out.
        projection : Proposition
            The projected symbol.

        Returns
        -------
        None
        """
        self._projections[symbol.idx][factor] = projection.idx

    def project(self, symbol: Proposition, factors: list[Factor]) -> Proposition:
        """
        Project a symbol to a subset of factors.

        Parameters
        ----------
        symbol : Proposition
            The symbol to be projected.
        factors : list[Factor]
            The factors to project out.

        Returns
        -------
        Proposition
            The projected symbol.
        """
        if len(factors) == 0:
            return symbol
        elif len(symbol.factors) == 1 and len(factors) == 1:
            if factors[0] in symbol.factors:
                return Proposition.empty()
            else:
                return symbol
        else:
            if factors[0] in symbol.factors:
                proj_f0_idx = self._projections[symbol.idx][factors[0]]
                proj_f0 = self[proj_f0_idx]
                return self.project(proj_f0, factors[1:])
            else:
                return self.project(symbol, factors[1:])

    def get_active_symbol_indices(self, observation: np.ndarray,
                                  max_samples: int = 100) -> np.ndarray:
        """
        Get the index of the active symbol for each factor.

        Parameters
        ----------
        observation : np.ndarray
            The observation to evaluate.
        max_samples : int, optional
            The maximum number of samples to use for scoring. Default is 100.

        Returns
        -------
        np.ndarray
            An array of size [n_sample, n_factors] or [n_sample, n_obj, n_factors] containing the index of the
            active symbol for each factor.
        """
        assert self.mutex_groups is not None, "Mutually exclusive factors are not defined."
        n_factors = len(self.mutex_groups)

        if observation.ndim == 3:
            n_sample, n_obj, _ = observation.shape
            object_factored = True
        elif observation.ndim == 2:
            n_sample, _ = observation.shape
            object_factored = False
        else:
            raise ValueError("Invalid observation shape.")

        if object_factored:
            indices = np.zeros((n_sample, n_obj, n_factors), dtype=int)
        else:
            indices = np.zeros((n_sample, n_factors), dtype=int)

        for f_i, factor in enumerate(self.mutex_groups):
            group = self.mutex_groups[factor]
            n_sym = len(group)
            if n_sym == 0:
                raise ValueError("This shouldn't happen?!")

            masked_obs = observation[..., factor.variables]
            if object_factored:
                scores = np.zeros((n_sample, n_obj, n_sym))
                masked_obs = masked_obs.reshape(n_sample * n_obj, -1)
            else:
                scores = np.zeros((n_sample, n_sym))

            for p_i, idx in enumerate(group):
                prop = self._list[idx]
                s = prop.estimator.score_samples(masked_obs, max_samples_used=max_samples)
                if object_factored:
                    s = s.reshape(n_sample, n_obj)
                scores[..., p_i] = s

            indices[..., f_i] = np.argmax(scores, axis=-1)
        return indices

    def get_topk_symbol_indices(self, observation: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the most probable k list of symbols in a beam search fashion.

        Parameters
        ----------
        observation : np.ndarray
            The observation to evaluate. shape: (n_obj, n_feature) or (n_feature,)
        k : int, optional
            The number of list of symbols to return. Default is 5.

        Returns
        -------
        indices : np.ndarray
            An array of size (k, n_factors) or (k, n_obj, n_factors) containing the index of the
            active symbol for each factor.
        scores : np.ndarray
            An array of size (k,) containing the scores of the top-k symbols.
        """
        assert self.mutex_groups is not None, "Mutually exclusive factors are not defined."

        if observation.ndim == 2:
            n_obj, _ = observation.shape
            object_factored = True
        elif observation.ndim == 1:
            object_factored = False
        else:
            raise ValueError("Invalid observation shape.")

        fringe = []
        for factor in self.mutex_groups:
            group = self.mutex_groups[factor]
            n_sym = len(group)

            masked_obs = observation[..., factor.variables]
            if object_factored:
                scores = np.zeros((n_obj, n_sym))
            else:
                scores = np.zeros(n_sym)

            for p_i, idx in enumerate(group):
                prop = self[idx]
                s = prop.estimator.score_samples(masked_obs)
                scores[..., p_i] = s

            new_fringe = []
            if len(fringe) == 0:
                for i in range(n_sym):
                    heapq.heappush(new_fringe, (-scores[i], [i]))
            else:
                for i in range(n_sym):
                    for p_k, path_k in fringe:
                        heapq.heappush(new_fringe, (-scores[i]+p_k, path_k + [i]))

            fringe = heapq.nsmallest(k, new_fringe)

        indices = np.array([path for _, path in fringe])
        # scores = np.array([-p for p, _ in fringe])
        return indices

    def fill_mutex_groups(self, factors: list[Factor]) -> None:
        """
        Fill the mutex groups for each factor.

        Parameters
        ----------
        factors : list[Factor]
            The factors to consider.

        Returns
        -------
        None
        """
        self.mutex_groups = defaultdict(list)
        self.factors = factors
        for factor in factors:
            for i, prop in enumerate(self._list):
                # only consider propositions with a single factor for now
                if (len(prop.factors) == 1) and (prop.factors[0] == factor):
                    self.mutex_groups[factor].append(i)
            # add empty list if there are no predicates for this factor
            if len(self.mutex_groups[factor]) == 0:
                self.mutex_groups[factor] = []

    def get_by_index(self, factor_index: int, symbol_index: int) -> Proposition:
        factor = self.factors[factor_index]
        group = self.mutex_groups[factor]
        return self[group[symbol_index]]

    def _get_or_none(self, item, parameters) -> int:
        for i, x in enumerate(self._list):
            if self._comparator(item, x.estimator) and (parameters == x.parameters):
                return i
        return -1

    def __getitem__(self, item: Union[int, slice, list, tuple]) -> Union[Proposition, list[Proposition]]:
        if isinstance(item, int):
            return self._list[item]
        elif isinstance(item, slice):
            return self._list[item]
        elif isinstance(item, list):
            return [self._list[i] for i in item]
        elif isinstance(item, tuple):
            return [self._list[i] for i in item]

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self) -> Proposition:
        if self.__idx < 0 or self.__idx >= len(self):
            raise StopIteration
        x = self._list[self.__idx]
        self.__idx += 1
        return x

    def __len__(self):
        return len(self._list)


class LiftedDecisionTree:
    def __init__(self,
                 vocabulary: UniquePredicateList,
                 referencable_indices: list[int],
                 min_samples_split: int = 10):
        self.ref_indices = referencable_indices
        self.vocabulary = vocabulary
        self.min_samples_split = min_samples_split
        self.nodes = {}
        self.leaves = {}

    def fit(self, x, y):
        x = self._translate(x)
        self._build_tree(x, y)

    def extract_preconditions(self):
        assert len(self.nodes) > 0, "Decision tree not trained yet."
        preconditions = []
        pos_leaves = [x[0] for x in self.leaves.values() if x[1] > 0.6]
        for branch in pos_leaves:
            pre = []
            a_it = 0
            for node in branch:
                (o_id, f_i), s_i = node
                sym = self.vocabulary.get_by_index(f_i, s_i)

                if o_id == -1:
                    sym = sym.substitute([(f"a{a_it}", None)])
                    a_it += 1
                else:
                    sym = sym.substitute([(f"x{o_id}", None)])
                pre.append(sym)
            preconditions.append(pre)
        return preconditions

    # def decision_path(self, x):
    #     x = self._translate(x)
    #     decisions = []
    #     for v in self.nodes:
    #         masks = self._get_decision_masks(self.nodes[v], x)
    #         decisions.append(masks.astype(int))
    #     return np.stack(decisions, axis=1)

    # def predict(self, x):
    #     g = self.decision_path(x)
    #     y = np.zeros(len(x))
    #     for i, path in enumerate(g):
    #         v = 0
    #         for d_j in path:
    #             if v not in self.nodes:
    #                 break
    #             if d_j == 0:
    #                 v = 2*v + 1
    #             else:
    #                 v = 2*v + 2
    #         y[i] = self.leaves[v][1]
    #     return y

    def _translate(self, x):
        x = self.vocabulary.get_active_symbol_indices(x)
        symbols = np.zeros_like(x)
        for i, f_i in enumerate(self.vocabulary.factors):
            group = np.array(self.vocabulary.mutex_groups[f_i])
            symbols[..., i] = group[x[..., i]]
        return symbols

    def _build_tree(self, x, y):
        queue = [((0, -1), [], x, y)]
        v_count = 1
        while len(queue) > 0:
            (v_idx, parent), rules, x, y = queue.pop(0)
            rule, child_masks = self._compute_best_rule(x, y)
            if rule is None:
                self.leaves[(v_idx, parent)] = (rules, y.mean())
                continue

            self.nodes[(v_idx, parent)] = rule
            for c_i, mask in enumerate(child_masks):
                child_rule = (rule, c_i)
                n_child = mask.sum()
                if n_child >= self.min_samples_split:
                    queue.append(((v_count, v_idx), rules + [child_rule], x[mask], y[mask]))
                    v_count += 1
                elif n_child > 0:
                    self.leaves[(v_count, v_idx)] = (rules + [child_rule], y[mask].mean())
                    v_count += 1

    def _compute_best_rule(self, x, y):
        best_rule = None
        best_child_masks = None
        best_info_gain = 1e-8
        for rule in self._get_valid_rules():
            info_gain, child_masks = self._compute_info_gain(rule, x, y)
            if info_gain > best_info_gain:
                best_rule = rule
                best_child_masks = child_masks
                best_info_gain = info_gain
        return best_rule, best_child_masks

    def _get_valid_rules(self):
        rules = []
        for i, _ in enumerate(self.vocabulary.factors):
            for o_x in self.ref_indices:
                rules.append((o_x, i))
            rules.append((-1, i))
        return rules

    def _compute_info_gain(self, rule, x, y):
        child_masks = self._get_decision_masks(rule, x)
        n = len(y)
        child_entropy = 0.0
        for mask in child_masks:
            n_i = len(y[mask])
            if n_i == 0:
                continue
            child_entropy += (n_i/n)*self._entropy(y[mask])
        info_gain = self._entropy(y) - child_entropy
        return info_gain, child_masks

    def _get_decision_masks(self, rule, x):
        o_i, f_i = rule
        syms = self.vocabulary.mutex_groups[self.vocabulary.factors[f_i]]
        child_masks = []
        for s_i in syms:
            if o_i == -1:
                mask = (x[:, :, f_i] == s_i).any(axis=1)
            else:
                mask = (x[:, o_i, f_i] == s_i)
            child_masks.append(mask)
        return child_masks

    @staticmethod
    def _entropy(y):
        if len(y) == 0:
            return 0
        p = np.mean(y)
        if (p < 1e-8) or (p > 1 - 1e-8):
            return 0
        return -p*np.log2(p) - (1-p)*np.log2(1-p)


class ActionSchema:
    """
    An action schema in PDDL. An action schema is a template for an action that can be instantiated
    with different objects.
    """
    def __init__(self, name: str):
        """
        Create a new action schema.

        Parameters
        ----------
        name : str
            The name of the action schema.

        Returns
        -------
        None
        """
        self.name = name.replace(' ', '-')
        self.preconditions = []
        self.effects = []

    def add_preconditions(self, predicates: list[Proposition]) -> None:
        """
        Add preconditions to the action schema.

        Parameters
        ----------
        predicates : list[Proposition]
            The preconditions to add.

        Returns
        -------
        None
        """
        self.preconditions.extend(predicates)

    def add_effects(self, effects: list[Proposition]) -> None:
        """
        Add effects to the action schema.

        Parameters
        ----------
        effect : list[Proposition]
            The effects to add.

        Returns
        -------
        None
        """
        self.effects.extend(effects)

    def is_probabilistic(self):
        # TODO: no probabilistic effects for now
        return False
        # return len(self.effects) > 1

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        precondition = ""
        effect = ""
        if len(self.preconditions) > 0:
            precondition += _proposition_to_str(self.preconditions)
        if len(self.effects) > 0:
            effect += _proposition_to_str(self.effects)
        params = []
        for prop in self.preconditions + self.effects:
            if prop.parameters is not None:
                for param in prop.parameters:
                    if param not in params:
                        params.append(param)

        parameters = []
        for param in params:
            if param[1] is not None:
                parameters.append(f"?{param[0]} - {param[1]}")
            else:
                parameters.append(f"?{param[0]}")
        parameters = " ".join(parameters)

        schema = f"(:action {self.name}\n" + \
                 f"\t:parameters ({parameters})\n" + \
                 f"\t:precondition (and {precondition})\n" + \
                 f"\t:effect (and {effect})\n)"
        return schema


def _proposition_to_str(proposition: Union[Proposition, list[Proposition]]) -> str:
    if isinstance(proposition, Proposition):
        return f"({proposition})"
    elif isinstance(proposition, list):
        return " ".join([f"({p})" for p in proposition])


class PDDLDomain:
    """
    A PDDL domain. A domain is a set of predicates, actions, and operators that define the state space
    and the actions that can be taken in that state space.
    """
    def __init__(self, name: str, vocabulary: UniquePredicateList, operators: list[ActionSchema], lifted: bool = False):
        """
        Create a new PDDL domain.

        Parameters
        ----------
        name : str
            The name of the domain.
        vocabulary : UniquePredicateList
            The vocabulary of the domain.
        operators : list[ActionSchema]
            The action schemas of the domain.
        lifted : bool, optional
            Whether the domain is lifted or not. Default is False.

        Returns
        -------
        None
        """
        self.name = name
        self.vocabulary = vocabulary
        self.num_operators = len(operators)
        self.operator_str = "\n\n".join([str(x) for x in operators])
        self.lifted = lifted

        self._comment = f";Automatically generated {self.name} domain PDDL file."
        self._definition = f"define (domain {self.name})"
        self._requirements = "\t(:requirements :strips)"

    def active_symbols(self, observation: np.ndarray) -> list[Proposition]:
        """
        Get the active symbols in the observation.

        Parameters
        ----------
        observation : np.ndarray
            The observation to evaluate.

        Returns
        -------
        active_symbols : list[Proposition]
            The active symbols in the observation.
        """
        assert self.vocabulary.mutex_groups is not None, "Mutually exclusive factors are not defined."

        active_symbols = {}
        if observation.ndim == 1:
            # global observation
            active_symbols["global"] = self._get_active_symbols(observation)
        else:
            # object-factored observation
            for o_i in range(observation.shape[0]):
                name = f"obj{o_i}"
                active_symbols[name] = self._get_active_symbols(observation[o_i])
        return active_symbols

    def _get_active_symbols(self, observation: np.ndarray) -> list[Proposition]:
        active_symbols = []
        for factor in self.vocabulary.mutex_groups:
            group = self.vocabulary.mutex_groups[factor]
            if len(group) == 0:
                continue

            scores = np.zeros(len(group))
            masked_obs = observation[factor.variables].reshape(1, -1)
            for p_i, idx in enumerate(group):
                prop = self.vocabulary[idx]
                scores[p_i] = prop.estimator.score_samples(masked_obs)[0]
            active_symbols.append(self.vocabulary[group[np.argmax(scores)]])
        return active_symbols

    def __str__(self):
        symbols = "\t\t "
        for i, p in enumerate(self.vocabulary):
            # TODO: need to understand lifted propositions
            # fixed to lifted propositions for now
            if self.lifted:
                symbols += f"({p} ?x)"
            else:
                symbols += f"({p})"

            if (i+1) % 6 == 0:
                symbols += "\n\t\t"
            else:
                symbols += " "

        predicates = f"\t(:predicates\n{symbols}\n\t)"

        description = f"{self._comment}\n" + \
                      f"({self._definition}\n" + \
                      f"{self._requirements}\n" + \
                      f"{predicates}\n\n" + \
                      f"{self.operator_str}\n)"
        return description

    def __repr__(self) -> str:
        return f"{self._comment}\n({self._definition}\n{self._requirements}\n" + \
               f"\t...{len(self.vocabulary)} symbols...\n\t...{self.num_operators} actions...\n)"


class PDDLProblem:
    """
    A PDDL problem. A problem is a specific instance of a domain. It defines the initial state of the
    environment and the goal state that the agent should reach.
    """
    def __init__(self, problem_name: str, domain_name: str):
        """
        Create a new PDDL problem.

        Parameters
        ----------
        problem_name : str
            The name of the problem.
        domain_name : str
            The name of the domain.

        Returns
        -------
        None
        """
        self.name = problem_name
        self.domain = domain_name
        self.init_propositions = []
        self.goal_propositions = []

    def add_init_proposition(self, proposition: Proposition, name: str = None):
        self.init_propositions.append((proposition, name))

    def add_goal_proposition(self, proposition: Proposition, name: str = None):
        self.goal_propositions.append((proposition, name))

    def get_object_names(self) -> list[str]:
        obj_names = []
        for prop, name in self.init_propositions:
            if name is not None and name not in obj_names:
                obj_names.append(name)
        for prop, name in self.goal_propositions:
            if name is not None and name not in obj_names:
                obj_names.append(name)
        return obj_names

    def __str__(self):
        init = ""
        for i, (prop, name) in enumerate(self.init_propositions):
            if name is not None:
                init += f"({prop} {name})"
            else:
                init += f"({prop})"
            if (i+1) % 6 == 0:
                init += "\n\t\t"
            elif i < len(self.init_propositions) - 1:
                init += " "
        goal = ""
        for i, (prop, name) in enumerate(self.goal_propositions):
            if name is not None:
                goal += f"({prop} {name})"
            else:
                goal += f"({prop})"
            if (i+1) % 6 == 0:
                goal += "\n\t\t"
            elif i < len(self.goal_propositions) - 1:
                goal += " "
        description = f"(define (problem {self.name})\n" + \
                      f"\t(:domain {self.domain})\n" + \
                      f"\t(:objects {' '.join(self.get_object_names())})\n" + \
                      f"\t(:init {init})\n" + \
                      f"\t(:goal (and {goal}))\n)"
        return description

    def __repr__(self) -> str:
        return self.__str__()


def sort_dataset(dataset: S2SDataset, mask_full_obj: bool = False,
                 mask_pos_feats: bool = False, flatten: bool = False,
                 shuffle_only_nonmask: bool = False) -> S2SDataset:
    """
    Given a dataset with object-factored states, convert it to a canonical form
    where objects that are affected by the action are followed by the non-affected
    objects. This reduces the invariance between samples. Note that there might
    still be invariances within affected objects and/or non-affected objects. Those
    invariances should be resolved by merging equivalent schemas.

    Parameters
    ----------
    dataset : S2SDataset
        The dataset to be converted.
    mask_full_obj : bool, optional
        Whether to mask the full object or just the effected part of the object.
    mask_pos_feats : bool, optional
        Whether to mask features that are related to positions all together.
    flatten : bool, optional
        Whether to flatten the state into a fixed-size vector instead of a collection
        of objects.
    shuffle_only_nonmask : bool, optional
        Whether to shuffle only the non-masked objects.

    Returns
    -------
    S2SDataset
        The dataset with a flattened state.
    """

    # flat already
    if dataset.state.ndim == 2:
        return dataset

    assert dataset.state.ndim == 3, "State should be 3D: (n_samples, n_objects, n_features)"

    n_sample, n_obj, n_feat = dataset.state.shape
    state = np.zeros((n_sample, n_obj, n_feat), dtype=dataset.state.dtype)
    next_state = np.zeros((n_sample, n_obj, n_feat), dtype=dataset.next_state.dtype)
    mask = np.zeros((n_sample, n_obj, n_feat), dtype=bool)

    for i in range(n_sample):
        order = []

        # add other objects that are affected by the action
        obj_mask = np.any(dataset.mask[i], axis=1)
        effected_objs, = np.where(obj_mask)
        if not shuffle_only_nonmask:
            np.random.shuffle(effected_objs)
        order.extend(effected_objs)

        # add other objects that are not affected by the action
        uneffected_objs, = np.where(np.logical_not(obj_mask))
        np.random.shuffle(uneffected_objs)
        order.extend(uneffected_objs)

        for j, o_i in enumerate(order):
            state[i, j] = dataset.state[i, o_i]
            next_state[i, j] = dataset.next_state[i, o_i]
            if mask_full_obj:
                mask[i, j] = np.any(dataset.mask[i, o_i], axis=-1)
            elif mask_pos_feats:
                mask[i, j] = dataset.mask[i, o_i]
                mask[i, j, -2:] = np.any(dataset.mask[i, o_i, -2:])
            else:
                mask[i, j] = dataset.mask[i, o_i]

    if flatten:
        state = state.reshape(n_sample, -1)
        next_state = next_state.reshape(n_sample, -1)
        mask = mask.reshape(n_sample, -1)

    return S2SDataset(state, dataset.option, dataset.reward, next_state, mask)
