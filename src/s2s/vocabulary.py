from itertools import chain, combinations
from typing import Union
import logging

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

from s2s.structs import (KernelDensityEstimator, KNNDensityEstimator, UniquePredicateList,
                         Factor, Proposition, ActionSchema, S2SDataset)

__author__ = 'Steve James and George Konidaris'
# Modified by Alper Ahmetoglu. Original source:
# https://github.com/sd-james/skills-to-symbols/tree/master

logger = logging.getLogger(__name__)


def build_vocabulary(partitions: dict[tuple[int, int], S2SDataset], factors: list[Factor]) \
        -> tuple[UniquePredicateList,
                 dict[tuple[int, int], list[Proposition]],
                 dict[tuple[int, int], list[Proposition]]]:
    """
    Build the vocabulary of propositions from the given partitions.

    Parameters
    ----------
        partitions : dict[tuple[int, int], S2SDataset]
            The partitions of the dataset.
        factors : list[Factor]
            The factors learned from the partitions.

    Returns
    -------
        vocabulary : UniquePredicateList
            The vocabulary of propositions.
        pre_props : dict[tuple[int, int], list[Proposition]]
            The preconditions for each partition.
        eff_props : dict[tuple[int, int], list[Proposition]]
            The effects for each partition.
    """
    # vocabulary = UniquePredicateList(_overlapping_dists)
    vocabulary = UniquePredicateList(_knn_overlapping, density_type="knn")
    pre_props = {}
    eff_props = {}

    # first add the factors to the partitions (mutates partitions!)
    add_factors_to_partitions(partitions, factors, threshold=0.9)

    # create effect symbols
    for key in partitions:
        partition_k = partitions[key]
        vocabulary, preds = create_effect_clause(vocabulary, partition_k)
        logger.info(f"Processed Eff({key[0]}-{key[1]}); {len(preds)} predicates found.")
        eff_props[key] = preds

    # compute symbols over factors that are mutually exclusive.
    # important: this assumes the partition semantics, not distribution!
    vocabulary.fill_mutex_groups(factors)

    # learn symbols for constant factors in case we need them for preconditions
    for f_i, factor in enumerate(vocabulary.mutex_groups):
        group = vocabulary.mutex_groups[factor]
        # no effect changes this factor, but there might be preconditions that depend on this factor.
        # so, learn a density estimator for each precondition just to be safe.
        # there should be only one such factor.
        if len(group) == 0:
            logger.info(f"Factor {f_i} is constant. Learning a density estimator in case a precond depends on it.")
            for key in partitions:
                logger.debug(f"Learning a density estimator for Pre({key[0]}-{key[1]}) on factor {f_i}...")
                partition_k = partitions[key]
                if partition_k.is_object_factored:
                    for j in range(partition_k.n_objects):
                        vocabulary.append(partition_k.state[:, j], [factor], [("x", None)])
                else:
                    vocabulary.append(partition_k.state, [factor])
            for i, pred in enumerate(vocabulary):
                if pred.factors[0] == factor:
                    group.append(i)

    # now learn preconditions in terms of the vocabulary found in the previous step
    # important: this part learns the most probably precondition and disregards others,
    # possibly breaking the soundness, but makes it efficient!
    for key in partitions:
        x_pos = partitions[key].state
        n = len(x_pos)
        other_options = [o for o in partitions if o != key]
        pre_count = {}
        for _ in range(10):
            x_neg = []
            for _ in range(n):
                j = np.random.randint(len(other_options))
                o_ = other_options[j]
                ds_neg = partitions[o_]
                sample_neg = ds_neg.state[np.random.randint(len(ds_neg.state))]
                x_neg.append(sample_neg)
            x_neg = np.array(x_neg)
            x = np.concatenate([x_pos, x_neg])
            y = np.concatenate([np.ones(n), np.zeros(n)])
            pre = _compute_preconditions(x, y, vocabulary, delta=1e-8)
            # count the number of times each proposition is selected
            for i, prop in enumerate(pre):
                if prop not in pre_count:
                    pre_count[prop] = 0
                pre_count[prop] += 1

        pre = []
        for prop in pre_count:
            if pre_count[prop] > 8:
                pre.append(prop)

        pre_props[key] = pre
        logger.info(f"Processed Pre({key[0]}-{key[1]}); {len(pre)} predicates found.")
    return vocabulary, pre_props, eff_props


def build_schemata(vocabulary: UniquePredicateList,
                   pre_props: dict[tuple[int, int], Union[list[Proposition], list[list[Proposition]]]],
                   eff_props: dict[tuple[int, int], Union[list[Proposition], list[list[Proposition]]]]) \
                    -> list[ActionSchema]:
    """
    Build the action schemata from the given preconditions and effects.

    Parameters
    ----------
        vocabulary : UniquePredicateList
            The vocabulary of propositions.
        pre_props : dict[tuple[int, int], list[Proposition] | list[list[Proposition]]]
            The preconditions for each partition.
        eff_props : dict[tuple[int, int], list[Proposition] | list[list[Proposition]]]
            The effects for each partition.

    Returns
    -------
        schemata : list[ActionSchema]
            The action schemata.
    """
    schemata = []
    for key in pre_props:
        pre = pre_props[key]
        eff = eff_props[key]
        if len(eff) == 0:
            continue
        action_schema = ActionSchema(f"a_{key[0]}_{key[1]}")
        action_schema = create_action_schema(action_schema, vocabulary, pre, eff)
        schemata.append(action_schema)
    return schemata


def create_action_schema(action_schema: ActionSchema, vocabulary: UniquePredicateList, pre_prop: list[Proposition],
                         eff_prop: list[Proposition]) -> ActionSchema:
    """
    Create an action schema from the given preconditions and effects.

    Parameters
    ----------
        action_schema : ActionSchema
            The action schema to add the preconditions and effects to.
        vocabulary : UniquePredicateList
            The vocabulary of propositions.
        pre_prop : list[Proposition]
            The preconditions.
        eff_prop : list[Proposition]
            The effects.

    Returns
    -------
        action_schema : ActionSchema
            The action schema with the preconditions and effects added.
    """
    action_schema.add_preconditions(pre_prop)

    # TODO: add probabilities...
    f_pre = set()
    f_eff = set()
    for prop in pre_prop:
        f_pre = f_pre.union(set(prop.factors))
    for prop in eff_prop:
        f_eff = f_eff.union(set(prop.factors))

    eff_neg = [x.negate() for x in pre_prop if set(x.factors).issubset(f_eff)]
    eff_proj_neg = [x for x in pre_prop if (not set(x.factors).issubset(f_eff)) and
                                           (not set(x.factors).isdisjoint(f_eff))]
    eff_proj_pos = [vocabulary.project(x, list(f_eff)) for x in eff_proj_neg]
    eff_proj_neg = [x.negate() for x in eff_proj_neg]

    action_schema.add_effects(eff_prop + eff_proj_pos + eff_neg + eff_proj_neg)
    return action_schema


def create_effect_clause(vocabulary: UniquePredicateList, partition: S2SDataset) \
        -> Union[list[KernelDensityEstimator], list[list[KernelDensityEstimator]]]:
    """
    Create the effect clause for the given partition.

    Parameters
    ----------
        vocabulary : UniquePredicateList
            The vocabulary of propositions.
        partition : S2SDataset
            The partition to create the effect clause for.

    Returns
    -------
        vocabulary : UniquePredicateList
            The updated vocabulary of propositions.
        effect_clause : list[KernelDensityEstimator] | list[list[KernelDensityEstimator]]
            The effect clause, which is a list of densities.
    """
    effect_clause = []
    if partition.factors is None:
        # partition does not contain any state changes. return empty list.
        return vocabulary, []

    if partition.is_object_factored:
        for i, obj_factors in enumerate(partition.factors):
            if len(obj_factors) == 0:
                continue
            densities = _create_factored_densities(partition.next_state[:, i], vocabulary, obj_factors,
                                                   lifted=True)
            for prop in densities:
                prop = prop.substitute([(f"x{i}", None)])
                effect_clause.append(prop)
    else:
        if len(partition.factors) != 0:
            densities = _create_factored_densities(partition.next_state, vocabulary, partition.factors)
            effect_clause.extend(densities)
    return vocabulary, effect_clause


def add_factors_to_partitions(partitions: dict[tuple[int, int], S2SDataset], factors: list[Factor],
                              threshold: float = 0.9) -> None:
    """
    Add factors to the partitions. Mutates the partitions in place.

    Parameters
    ----------
    partitions : dict[tuple[int, int], S2SDataset]
        Subgoal partitions.
    factors : list[Factor]
        Factors that represent the state space.
    threshold : float
        Minimum proportion of samples that must be modified
        for a factor to be considered as modified by the partition.

    Returns
    -------
    None
    """
    partition_keys = list(partitions.keys())
    for p_i in partition_keys:
        partition = partitions[p_i]

        if partition.is_object_factored:
            partition.factors = [[] for _ in range(partition.n_objects)]
        else:
            partition.factors = []

        for factor in factors:
            factor_mask = partition.mask[..., factor.variables].mean(axis=0)
            # By construction, if any of the variables is higher than the threshold,
            # then all of them should be higher. But if factors are provided from
            # some other procedure, then this might not be the case. Therefore,
            # check if any of the variables are modified enough.
            if partition.is_object_factored:
                for obj_i in range(partition.n_objects):
                    if (factor_mask[obj_i] > threshold).any():
                        partition.factors[obj_i].append(factor)
            else:
                if (factor_mask > threshold).any():
                    partition.factors.append(factor)


def _compute_preconditions(x: np.ndarray, y: np.ndarray, vocabulary: UniquePredicateList,
                           delta: float = 0.01) -> Union[list[Proposition], list[list[Proposition]]]:
    """
    Compute the preconditions from the given data.

    Parameters
    ----------
        x : np.ndarray
            The data.
        y : np.ndarray
            The labels.
        vocabulary : UniquePredicateList
            The vocabulary of propositions.
        delta : float
            The threshold for the factor selection.

    Returns
    -------
        preconditions : list[Proposition] | list[list[Proposition]]
            The preconditions.
    """
    symbol_indices = vocabulary.get_active_symbol_indices(x)
    if symbol_indices.ndim == 3:
        object_factored = True
    else:
        object_factored = False

    n = symbol_indices.shape[0]
    pos_symbols = symbol_indices[:n//2]

    # get the most frequent symbol for each factor
    mode, _ = stats.mode(pos_symbols, axis=0, keepdims=False)
    preds = []

    current_mask = np.ones(n, dtype=bool)
    last_score = 0.5
    if object_factored:
        for j in range(symbol_indices.shape[1]):
            for f_i in range(symbol_indices.shape[2]):
                sym = int(mode[j, f_i])
                mask = symbol_indices[:, j, f_i] == sym
                mask = mask & current_mask
                if not np.any(mask):
                    continue
                score = np.mean(y[mask] == 1)
                if score > (last_score + delta):
                    prop = vocabulary.get_by_index(f_i, sym)
                    prop = prop.substitute([(f"x{j}", None)])
                    preds.append(prop)
                    current_mask = mask
                    last_score = score
    else:
        for f_i in range(symbol_indices.shape[1]):
            sym = int(mode[f_i])
            mask = symbol_indices[:, f_i] == sym
            mask = mask & current_mask
            if not np.any(mask):
                continue
            score = np.mean(y[mask] == 1)
            if score > (last_score + delta):
                prop = vocabulary.get_by_index(f_i, sym)
                preds.append(prop)
                current_mask = mask
                last_score = score
    return preds


def _create_factored_densities(data: np.ndarray, vocabulary: UniquePredicateList,
                               factors: list[Factor], lifted: bool = False) \
        -> list[KernelDensityEstimator]:
    """
    Create the factored densities from the given data.

    Parameters
    ----------
        data : np.ndarray
            The data.
        vocabulary : UniquePredicateList
            The vocabulary of propositions.
        factors : list[Factor]
            The factors.

    Returns
    -------
        densities : list[KernelDensityEstimator]
            The factored densities.
    """
    # use validation samples for independency tests
    n_val = max(int(len(data) * 0.1), 3)
    densities = []
    dependency_groups = _compute_factor_dependencies(data[-n_val:], factors, method="knn")
    _n_expected_symbol = sum([2**len(x)-1 for x in dependency_groups])
    logger.info(f"n_factors={len(factors)}, n_groups={len(dependency_groups)}, n_expected_symbol={_n_expected_symbol}")
    for group in dependency_groups:
        params = None
        if lifted:
            params = [("x", None)]
        predicate = vocabulary.append(data, list(group), params)
        densities.append(predicate)
    return densities


def _compute_factor_dependencies(data: np.ndarray, factors: list[Factor], method: str = "independent") \
        -> list[list[Factor]]:
    """
    Compute the factor dependencies.

    Parameters
    ----------
        data : np.ndarray
            The data.
        factors : list[Factor]
            The factors.
        method : str
            The method to use for computing the factor dependencies.

    Returns
    -------
        independent_factor_groups : list[list[Factor]]
            The independent factor groups.
    """
    if len(factors) == 1:
        return [factors]

    if method == "gaussian":
        return _gaussian_independent_factor_groups(data, factors)
    elif method == "knn":
        return _knn_independent_factor_groups(data, factors)
    elif method == "independent":
        return [[f] for f in factors]


def _gaussian_independent_factor_groups(data: np.ndarray, factors: list[Factor]) -> list[list[Factor]]:
    """
    Compute linear dependencies of factors.

    Parameters
    ----------
        data : np.ndarray
            The data.
        factors : list[Factor]
            The factors.

    Returns
    -------
        independent_factor_groups : list[list[Factor]]
            The independent factor groups.
    """
    n_factors = len(factors)
    independent_factor_groups = []
    factor_vars = []
    factor_indices = []
    it = 0
    for f in factors:
        factor_vars.extend(f.variables)
        n_vars = len(f.variables)
        factor_indices.append(np.arange(it, it+n_vars))
        it += n_vars
    assert len(factor_vars) > 0

    cov = np.cov(data[:, factor_vars], rowvar=False) + 1e-6

    std = cov.diagonal()**0.5
    abs_corr = abs(cov / (std.reshape(-1, 1) @ std.reshape(1, -1)))
    ind_map = abs_corr > 0.7
    ind_factors = np.zeros((n_factors, n_factors), dtype=bool)
    for i in range(n_factors):
        i_vars = factor_indices[i]
        i_changing = std[i_vars] > 0.1
        if not np.any(i_changing):
            ind_factors[i, i] = True
            continue

        for j in range(i, n_factors):
            j_vars = factor_indices[j]
            j_changing = std[j_vars] > 0.1
            if not np.any(j_changing):
                continue

            ij_dep = np.any(ind_map[i_vars[i_changing]][:, j_vars[j_changing]])
            ind_factors[i, j] = ij_dep
            ind_factors[j, i] = ij_dep

    remaining_factors = list(range(n_factors))
    while len(remaining_factors) > 0:
        ind_partial = ind_factors[remaining_factors][:, remaining_factors]
        group_idx, = np.where(ind_partial[0])
        group = [remaining_factors[g_i] for g_i in group_idx]
        independent_factor_groups.append([factors[g_i] for g_i in group])
        remaining_factors = [f for f in remaining_factors if f not in group]
    return independent_factor_groups


def _knn_independent_factor_groups(data: np.ndarray, factors: list[Factor]) -> list[list[Factor]]:
    """
    Compute the independent factor groups using k-NN.

    Parameters
    ----------
        data : np.ndarray
            The data.
        factors : list[Factor]
            The factors.

    Returns
    -------
        independent_factor_groups : list[list[Factor]]
            The independent factor groups.
    """
    n_factors = len(factors)
    independent_factor_groups = []
    remaining_factors = [f for f in factors]
    max_comb_size = n_factors // 2
    for comb_size in range(1, max_comb_size+1):
        if len(remaining_factors) <= comb_size:
            break

        comb_queue = list(combinations(remaining_factors, comb_size))
        while len(comb_queue) > 0:
            comb = comb_queue.pop(0)
            remaining_factors = [f for f in remaining_factors if f not in comb]
            f_vars = list(chain.from_iterable([f.variables for f in comb]))
            other_vars = list(chain.from_iterable([f.variables for f in remaining_factors]))
            data_comb = data[:, f_vars]
            data_other = data[:, other_vars]
            data_real = np.concatenate([data_comb, data_other], axis=1)
            data_fake = np.concatenate([data_comb, np.random.permutation(data_other)], axis=1)
            x_acc, y_acc = _knn_accuracy(data_real, data_fake)
            if abs(x_acc-y_acc) < 0.075:
                independent_factor_groups.append(comb)
                remaining_factors = [f for f in remaining_factors if f not in comb]
                comb_queue = [c for c in comb_queue if not set(c).intersection(set(comb))]
    if len(independent_factor_groups) == 0:
        independent_factor_groups.append(factors)
    return independent_factor_groups


def _overlapping_dists(x: KernelDensityEstimator, y: KernelDensityEstimator) -> bool:
    """
    A measure of similarity from the original paper that compares means, mins and maxes.

    Parameters
    ----------
        x : KernelDensityEstimator
            The first distribution.
        y : KernelDensityEstimator
            The second distribution.

    Returns
    -------
        bool: True if the distributions are similar, False otherwise.
    """
    if set(x.factors) != set(y.factors):
        return False

    dat1 = x.sample(100)
    dat2 = y.sample(100)

    mean1 = np.mean(dat1)
    mean2 = np.mean(dat2)
    if np.linalg.norm(mean1 - mean2) > 0.1:
        return False

    ndims = len(x.variables)
    for n in range(ndims):
        if np.min(dat1[:, n]) > np.max(dat2[:, n]) or np.min(dat2[:, n]) > np.max(dat1[:, n]):
            return False
    return True


def _knn_overlapping(x: KNNDensityEstimator, y: KNNDensityEstimator, k: int = 5, threshold=0.1) -> bool:
    if set(x.factors) != set(y.factors):
        return False
    x_acc, y_acc = _knn_accuracy(x._samples, y._samples, k)
    x_diff = abs(x_acc - 0.5)
    y_diff = abs(y_acc - 0.5)
    diff = (x_diff + y_diff) / 2
    return diff < threshold


def _knn_accuracy(x: np.ndarray, y: np.ndarray, k: int = 5, max_samples: int = 100) -> tuple[float, float]:
    """
    Compute classifier 2-sample test with k-NN.

    Parameters
    ----------
        x : np.ndarray
            The first dataset.
        y : np.ndarray
            The second dataset.
        k : int
            The number of nearest neighbors.

    Returns
    -------
        x_acc : float
            The accuracy of the classifier on the first dataset.
        y_acc : float
            The accuracy of the classifier on the second dataset.
    """
    n_sample = min(x.shape[0], y.shape[0], max_samples)
    x_idx = np.random.choice(x.shape[0], n_sample, replace=False)
    y_idx = np.random.choice(y.shape[0], n_sample, replace=False)
    # add noise to avoid ties
    x = x[x_idx].reshape(n_sample, -1) + 1e-6 * np.random.randn(n_sample, x.shape[1])
    y = y[y_idx].reshape(n_sample, -1) + 1e-6 * np.random.randn(n_sample, y.shape[1])
    xy = np.concatenate([x, y], axis=0)
    dists = cdist(xy, xy) + np.eye(2*n_sample) * 1e12
    indexes = np.argsort(dists, axis=1)[:, :k]
    x_decisions = np.sum(indexes < n_sample, axis=1) / k
    x_acc = np.sum(x_decisions[:n_sample]) / n_sample
    y_acc = 1 - np.sum(x_decisions[n_sample:]) / n_sample
    return x_acc, y_acc
