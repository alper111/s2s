from typing import NamedTuple, Callable
import copy

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KernelDensity


__author__ = 'Steve James and George Konidaris'
# Modified by Alper Ahmetoglu. Original source:
# https://github.com/sd-james/skills-to-symbols/tree/master


S2SDataset = NamedTuple('S2SDataset', [
    ('state', np.ndarray),
    ('option', np.ndarray),
    ('reward', np.ndarray),
    ('next_state', np.ndarray),
    ('mask', np.ndarray),
])


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
        c_range = kwargs.get('precondition_c_range', np.logspace(-2, 2, 10))
        gamma_range = kwargs.get('precondition_gamma_range', np.logspace(-2, 2, 10))

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
        bandwidth_range = kwargs.get('effect_bandwidth_range', np.logspace(-3, 1, 20))
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


Operator = NamedTuple('Operator', [
    ('option', int),
    ('partition', int),
    ('precondition', SupportVectorClassifier),
    ('effect', KernelDensityEstimator),  # TODO: this will be a list in the probabilistic setting
])


class Proposition:
    """
    A non-typed, non-lifted predicate (i.e. a proposition).
    """

    def __init__(self, name: str, kde: KernelDensityEstimator | None):
        self._name = name
        self._kde = kde
        self.sign = 1  # whether true or the negation of the predicate

    @property
    def estimator(self) -> KernelDensityEstimator | None:
        return self._kde

    @property
    def name(self) -> str:
        return self._name

    @property
    def mask(self):
        assert isinstance(self._kde, KernelDensityEstimator)
        return self._kde.mask

    def sample(self, n_samples):
        assert isinstance(self._kde, KernelDensityEstimator)
        return self._kde.sample(n_samples)

    def is_grounded(self) -> bool:
        return False

    def negate(self) -> 'Proposition':
        """"
        Creates a negated copy of the predicate.
        """
        clone = copy.copy(self)
        clone.sign *= -1
        return clone

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        if self.sign < 0:
            return 'not ({})'.format(self.name)
        return self.name

    @staticmethod
    def not_failed():
        return Proposition("notfailed", None)


class UniquePredicateList:
    """
    A class that wraps a list. The user add density estimators to the list,
    and they are wrapped in PDDL predicates.  The list automatically deals
    with duplicates.
    """

    def __init__(self, comparator: Callable[[KernelDensityEstimator, KernelDensityEstimator], bool] | None = None):
        """
        Create a list data structure that ensures no duplicates are added to the list.

        Parameters
        ----------
        comparator : callable, optional
            A function that accepts two objects and returns whether they are equal.
        """
        self._comparator = comparator if comparator is not None else lambda x, y: x is y
        self._list = []
        self.mutex_groups = None
        self.__idx = 0

    def append(self, item: KernelDensityEstimator) -> Proposition:
        """
        Add an item to the list

        Parameters
        ----------
        item : KernelDensityEstimator
            The item to add.

        Returns
        -------
        predicate : Proposition
            The predicate in the list. If the item is a duplicate, the predicate will refer to the
            existing item in the list.
        """
        for x in self._list:
            if self._comparator(item, x.estimator):
                return x
        idx = len(self._list)
        predicate = Proposition('symbol_{}'.format(idx), item)
        self._list.append(predicate)
        return predicate

    def fill_mutex_groups(self, factors: list[list[int]]) -> None:
        self.mutex_groups = [[] for _ in range(len(factors))]
        for f_i, factor in enumerate(factors):
            for i, pred in enumerate(self._list):
                if set(pred.mask) == set(factor):
                    self.mutex_groups[f_i].append(i)

    def __getitem__(self, item: int | slice | list | tuple) -> Proposition | list[Proposition]:
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


class ActionSchema:
    def __init__(self, operator: Operator, name: str | None = None):
        self.operator = operator

        if name is not None:
            self.name = name.replace(' ', '-')
        else:
            self.name = f"option-{self.operator.option}-partition-{self.operator.partition}"
        self.preconditions = [Proposition.not_failed()]
        self.effects = []

    def add_preconditions(self, predicates: list[Proposition]):
        self.preconditions.extend(predicates)

    def add_effect(self, effect: list[Proposition], probability: float = 1):
        self.effects.append((probability, effect))

    def is_probabilistic(self):
        return len(self.effects) > 1

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        precondition = _proposition_to_str(self.preconditions)

        if self.is_probabilistic():
            # Probabilistic effects not supported yet. Defaulting to the most probable effect.
            effects = [max(self.effects, key=lambda x: x[0])]  # get most probable
            # effects = self.effects
        else:
            effects = [max(self.effects, key=lambda x: x[0])]  # get most probable

        if len(effects) == 1:
            effect = _proposition_to_str(effects[0][1])
        else:
            # Probabilistic effects not supported yet. Defaulting to the most probable effect.
            effect = _proposition_to_str(effects[0][1])

        schema = f"(:action {self.name}\n" + \
                 "\t:parameters ()\n" + \
                 f"\t:precondition (and {precondition})\n" + \
                 f"\t:effect (and {effect})\n)"
        return schema


def _proposition_to_str(proposition: Proposition | list[Proposition]) -> str:
    if isinstance(proposition, Proposition):
        return str(proposition)
    elif isinstance(proposition, list):
        assert len(proposition) > 0
        prop_str = list(map(str, proposition))
        return f"{' '.join([f'({prop})' for prop in prop_str])}"


class PDDLDomain:
    def __init__(self, name: str, vocabulary: UniquePredicateList, operators: list[ActionSchema]):
        self.name = name
        self.vocabulary = vocabulary
        self.num_operators = len(operators)
        self.operator_str = "\n\n".join([str(x) for x in operators])

        self._comment = f";Automatically generated {self.name} domain PDDL file."
        self._definition = f"define (domain {self.name})"
        self._requirements = "\t(:requirements :strips)"

    def get_active_symbols(self, observation: np.ndarray) -> list[Proposition]:
        assert self.vocabulary.mutex_groups is not None, "Mutually exclusive factors are not defined."

        active_symbols = []
        for _, group in enumerate(self.vocabulary.mutex_groups):
            scores = np.zeros(len(group))
            prop = self.vocabulary[group[0]]
            assert isinstance(prop, Proposition)
            masked_obs = observation[prop.mask].reshape(1, -1)
            for p_i, idx in enumerate(group):
                prop = self.vocabulary[idx]
                scores[p_i] = prop.estimator._kde.score_samples(masked_obs)[0]
            active_symbols.append(group[np.argmax(scores)])
        return active_symbols

    def __str__(self):
        symbols = f"\t\t({Proposition.not_failed()}) "
        for i, p in enumerate(self.vocabulary):
            symbols += f"({p})"
            if (i+1) % 6 == 0:
                symbols += "\n\t\t"
            else:
                symbols += " "

        predicates = f"\t(:predicates\n{symbols}\n\t)"

        description = f"{self._comment}\n({self._definition}\n{self._requirements}\n{predicates}\n\n{self.operator_str}\n)"
        return description

    def __repr__(self) -> str:
        return f"{self._comment}\n({self._definition}\n{self._requirements}\n" + \
               f"\t...{len(self.vocabulary)} symbols...\n\t...{self.num_operators} actions...\n)"


class PDDLProblem:
    def __init__(self, problem_name: str, domain_name: str):
        self.name = problem_name
        self.domain = domain_name
        self.init_propositions = []
        self.goal_propositions = []

    def add_init_proposition(self, proposition: Proposition):
        self.init_propositions.append(proposition)

    def add_goal_proposition(self, proposition: Proposition):
        self.goal_propositions.append(proposition)

    def __str__(self):
        init = ""
        for i, p in enumerate(self.init_propositions):
            init += f"({p})"
            if (i+1) % 6 == 0:
                init += "\n\t\t"
            elif i < len(self.init_propositions) - 1:
                init += " "
        goal = ""
        for i, p in enumerate(self.goal_propositions):
            goal += f"({p})"
            if (i+1) % 6 == 0:
                goal += "\n\t\t"
            elif i < len(self.goal_propositions) - 1:
                goal += " "
        description = f"(define (problem {self.name})\n" + \
                      f"\t(:domain {self.domain})\n" + \
                      f"\t(:init {init})\n" + \
                      f"\t(:goal (and {goal}))\n)"
        return description

    def __repr__(self) -> str:
        return self.__str__()
