from typing import Callable
from collections import defaultdict
import copy

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KernelDensity


__author__ = 'Steve James and George Konidaris'
# Modified by Alper Ahmetoglu. Original source:
# https://github.com/sd-james/skills-to-symbols/tree/master


class Factor:
    """
    A factor is a minimal set of state variables that can be changed by an option.
    """
    def __init__(self, indices):
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

    def is_independent(self, other: 'Factor', dataset: 'S2SDataset') -> bool:
        """
        Check if this factor is independent of another factor in a dataset.

        Parameters
        ----------
        other : Factor
            The other factor.
        dataset : S2SDataset
            The dataset.

        Returns
        -------
        bool
            Whether the factors are independent.
        """
        # set to true for now
        return True

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Factor({self._indices})"

    def __hash__(self):
        return hash(tuple(sorted(self.variables)))


class S2SDataset:
    def __init__(self, state: np.ndarray, option: np.ndarray, reward: np.ndarray,
                 next_state: np.ndarray, mask: np.ndarray, factors: list[Factor] = None):
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

    def __len__(self):
        return len(self.state)

    def __getitem__(self, item):
        return S2SDataset(self.state[item], self.option[item], self.reward[item],
                          self.next_state[item], self.mask[item])

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        n_factor = len(self.factors) if self.factors is not None else "unset"
        if self.state.ndim == 2:
            return f"S2SDataset(n_sample={len(self)}, n_feature={self.state.shape[1]}, n_factor={n_factor})"
        return f"S2SDataset(n_sample={len(self)}, n_object={self.state.shape[1]}, " + \
               f"n_feature={self.state.shape[2]}, n_factor={n_factor})"


class SupportVectorClassifier:
    """
    An implementation of a probabilistic classifier that
    uses support vector machines with Platt scaling.
    """

    def __init__(self, factor: Factor, probabilistic=True):
        """
        Create a new SVM classifier for preconditions

        Parameters
        ----------
        factor : Factor
            The factor that the classifier models.
        probabilistic : bool, optional
            Whether the classifier is probabilistic
        """
        self._factor = factor
        self._probabilistic = probabilistic
        self._classifier: SVC | None = None

    @property
    def factor(self) -> Factor:
        """
        Get the precondition mask
        """
        return self.factor

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
        data = X[:, self.factor.variables]
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

        masked_states = states[:, self.factor.variables]
        if self._probabilistic:
            return np.mean(self._classifier.predict_proba(masked_states)[0][1])
        else:
            return self._classifier.predict(masked_states)[0]


class KernelDensityEstimator:
    """
    A density estimator that models a distribution over a factor (a set of low-level states).
    """

    def __init__(self, factor: Factor):
        """
        Initialize a new estimator.

        Parameters
        ----------
        factor : Factor
            The factor that the estimator models.
        """
        self._factor = factor
        self._kde: KernelDensity | None = None

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
            data = X[:, self.factor.variables]
        bandwidth_range = kwargs.get('effect_bandwidth_range', np.logspace(-3, 1, 20))
        params = {'bandwidth': bandwidth_range}
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=3)
        grid.fit(data)
        # print("Best bandwidth hyperparameter: {}".format(grid.best_params_['bandwidth']))
        self._kde = grid.best_estimator_

    @property
    def factor(self) -> Factor:
        """
        Get the effect factor.
        """
        return self._factor

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
            return np.reshape(data, (data.shape[0], 1))
        return data

    def integrate_out(self, variable_list: list[int], **kwargs) -> 'KernelDensityEstimator':
        """
        Integrate out the given variables from the distribution.

        Parameters
        ----------
        variable_list : list[int]
            A list of variables to be marginalized out from the distribution.

        Returns
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


class Operator:
    def __init__(self, option: int, partition: int, precondition: list[SupportVectorClassifier],
                 effect: list[KernelDensityEstimator]):
        self.option = option
        self.partition = partition
        self.precondition = precondition
        self.effect = effect


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
    def factor(self):
        assert isinstance(self._kde, KernelDensityEstimator)
        return self._kde.factor

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
        self.factors = None
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
            self.factors.append(factor)

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
        self.preconditions = []
        self.effects = []
        self.obj_preconditions = {}
        self.obj_effects = defaultdict(list)

    def add_preconditions(self, predicates: list[Proposition]):
        self.preconditions.extend(predicates)

    def add_obj_preconditions(self, obj_idx: str, predicates: list[Proposition]):
        if obj_idx not in self.obj_preconditions:
            self.obj_preconditions[obj_idx] = [Proposition.not_failed()]
        self.obj_preconditions[obj_idx].extend(predicates)

    def add_effect(self, effect: list[Proposition], probability: float = 1):
        self.effects.append((probability, effect))

    def add_obj_effect(self, obj_idx: str, effect: list[Proposition], probability: float = 1):
        self.obj_effects[obj_idx].append((probability, effect))

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
            effects = max(self.effects, key=lambda x: x[0])
            effect += _proposition_to_str(effects[1], None)
        parameters = []
        for name in self.obj_preconditions:
            parameters.append(name)
            if len(precondition) > 0:
                precondition += " "
            precondition += _proposition_to_str(self.obj_preconditions[name], name)

        for name in self.obj_effects:
            if name not in parameters:
                parameters.append(name)
            if len(effect) > 0:
                effect += " "
            max_effect = max(self.obj_effects[name], key=lambda x: x[0])
            effect += _proposition_to_str(max_effect[1], name)
        parameters = " ".join([f"?{name}" for name in parameters])

        schema = f"(:action {self.name}\n" + \
                 f"\t:parameters ({parameters})\n" + \
                 f"\t:precondition (and {precondition})\n" + \
                 f"\t:effect (and {effect})\n)"
        return schema


def _proposition_to_str(proposition: Proposition | list[Proposition], name: str = None) -> str:
    if isinstance(proposition, Proposition):
        prop = proposition.name
        if name is not None:
            prop = f"{prop} ?{name}"
        if proposition.sign < 0:
            return f"(not ({prop}))"
    elif isinstance(proposition, list):
        assert len(proposition) > 0
        props = []
        for prop in proposition:
            p = prop.name
            if name is not None:
                p = f"{p} ?{name}"
            if prop.sign < 0:
                props.append(f"(not ({p}))")
            else:
                props.append(f"({p})")
        return " ".join(props)


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

        active_symbols = {}
        if observation.ndim == 1:
            # global observation
            active_symbols["global"] = []
            for _, group in enumerate(self.vocabulary.mutex_groups):
                if len(group) == 0:
                    continue

                scores = np.zeros(len(group))
                prop = self.vocabulary[group[0]]
                assert isinstance(prop, Proposition)
                masked_obs = observation[prop.mask].reshape(1, -1)
                for p_i, idx in enumerate(group):
                    prop = self.vocabulary[idx]
                    scores[p_i] = prop.estimator._kde.score_samples(masked_obs)[0]
                active_symbols["global"].append(group[np.argmax(scores)])
        else:
            # object-factored observation
            for o_i in range(observation.shape[0]):
                name = f"obj{o_i}"
                active_symbols[name] = []
                for _, group in enumerate(self.vocabulary.mutex_groups):
                    if len(group) == 0:
                        continue

                    scores = np.zeros(len(group))
                    prop = self.vocabulary[group[0]]
                    assert isinstance(prop, Proposition)
                    masked_obs = observation[o_i][prop.mask].reshape(1, -1)
                    for p_i, idx in enumerate(group):
                        prop = self.vocabulary[idx]
                        scores[p_i] = prop.estimator._kde.score_samples(masked_obs)[0]
                    active_symbols[name].append(group[np.argmax(scores)])
        return active_symbols

    def __str__(self):
        symbols = f"\t\t({Proposition.not_failed()} ?x) "
        for i, p in enumerate(self.vocabulary):
            # TODO: need to understand lifted propositions
            # fixed to lifted propositions for now
            symbols += f"({p} ?x)"
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
    def __init__(self, problem_name: str, domain_name: str):
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


def sort_dataset(dataset: S2SDataset, mask_full_obj: bool = False, flatten: bool = False) -> S2SDataset:
    """
    Given a dataset with object-factored states, convert it to a canonical form
    where objects are ordered based on the action parameters, followed by objects
    that are affected by the action, and finally objects that are not affected by
    the action.

    Parameters
    ----------
    dataset : S2SDataset
        The dataset to be converted.
    mask_full_obj : bool, optional
        Whether to mask the full object or just the effected part of the object.
    flatten : bool, optional
        Whether to flatten the state into a fixed-size vector instead of a collection
        of objects.

    Returns
    -------
    S2SDataset
        The dataset with a flattened state.
    """

    # Note:
    # This can be arbitrarily different for different domains.
    # E.g., one can concatenate objects in an order based on the action params, or
    # proximity to the acted object, etc. This is a domain-specific choice.
    # For now, we implement it for domains in which the action is applied to an object,
    # and that object is the first object in the state. Next, objects that have non-zero
    # mask values are concatenated in a random order. Lastly, other objects are concatenated
    # in a random order to ensure invariance to object ordering.

    # if the state is not object-factored
    # require that the state consists of objects (i.e., 3d array)
    assert dataset.state.ndim == 3, "State must be object-factored"

    n_sample, n_obj, n_feat = dataset.state.shape
    state = np.zeros((n_sample, n_obj, n_feat))
    next_state = np.zeros((n_sample, n_obj, n_feat))
    mask = np.zeros((n_sample, n_obj, n_feat))

    for i in range(n_sample):
        order = []
        # get the object index that the action was applied to
        order.append(dataset.option[i][1])

        # add other objects that are affected by the action
        obj_mask = np.any(dataset.mask[i], axis=1)
        effected_objs, = np.where(obj_mask)
        effected_objs = effected_objs[effected_objs != order[0]]
        np.random.shuffle(effected_objs)
        order.extend(effected_objs)

        # add other objects that are not affected by the action
        uneffected_objs, = np.where(np.logical_not(obj_mask))
        uneffected_objs = uneffected_objs[uneffected_objs != order[0]]
        np.random.shuffle(uneffected_objs)
        order.extend(uneffected_objs)

        for j, o_i in enumerate(order):
            state[i, j] = dataset.state[i, o_i]
            next_state[i, j] = dataset.next_state[i, o_i]
            if mask_full_obj:
                mask[i, j] = np.any(dataset.mask[i, o_i], axis=-1)
            else:
                mask[i, j] = dataset.mask[i, o_i]

    if flatten:
        state = state.reshape(n_sample, -1)
        next_state = next_state.reshape(n_sample, -1)
        mask = mask.reshape(n_sample, -1)

    return S2SDataset(state, dataset.option, dataset.reward, next_state, mask)
