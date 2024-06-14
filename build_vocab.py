from itertools import product, chain, combinations
import logging

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

from structs import (KernelDensityEstimator, Operator, UniquePredicateList, Factor,
                     Proposition, ActionSchema, SupportVectorClassifier, S2SDataset)

__author__ = 'Steve James and George Konidaris'
# Modified by Alper Ahmetoglu. Original source:
# https://github.com/sd-james/skills-to-symbols/tree/master

logger = logging.getLogger(__name__)


def build_vocabulary(partitions: dict[tuple[int, int], S2SDataset], factors: list[Factor]) \
        -> tuple[UniquePredicateList,
                 dict[tuple[int, int], list[Proposition]],
                 dict[tuple[int, int], list[Proposition]]]:
    vocabulary = UniquePredicateList(_overlapping_dists)
    pre_props = {}
    eff_props = {}

    for key in partitions:
        partition_k = partitions[key]
        vocabulary, preds = create_effect_clause(vocabulary, partition_k)
        n_pred = len(preds) if not partition_k.is_object_factored else sum([len(x) for x in preds])
        logger.info(f"Processed Eff({key[0]}-{key[1]}); {n_pred} predicates found.")
        eff_props[key] = preds

    vocabulary.fill_mutex_groups(factors)
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
                        vocabulary.append(partition_k.state[:, j], [factor])
                        # density = _create_factored_densities(partition_k.state[:, j], vocabulary, [factor])
                        # vocabulary.append(density[0], group)
                else:
                    vocabulary.append(partition_k.state, [factor])
                    # density = _create_factored_densities(partition_k.state, vocabulary, [factor])
                    # vocabulary.append(density[0], group)
            for i, pred in enumerate(vocabulary):
                if pred.factors[0] == factor:
                    group.append(i)

    # now learn preconditions in terms of the vocabulary found in the previous step
    for key in partitions:
        x_pos = partitions[key].state
        n = len(x_pos)
        x_neg = []
        other_options = [o for o in partitions if o != key]
        for _ in range(n):
            j = np.random.randint(len(other_options))
            o_ = other_options[j]
            ds_neg = partitions[o_]
            sample_neg = ds_neg.state[np.random.randint(len(ds_neg.state))]
            x_neg.append(sample_neg)
        x_neg = np.array(x_neg)
        x = np.concatenate([x_pos, x_neg])
        y = np.concatenate([np.ones(n), np.zeros(n)])
        pre = _compute_preconditions(x, y, vocabulary)
        pre_props[key] = pre
        n_pred = len(pre) if not partitions[key].is_object_factored else sum([len(x) for x in pre])
        logger.info(f"Processed Pre({key[0]}-{key[1]}); {n_pred} predicates found.")
    return vocabulary, pre_props, eff_props


def build_schemata(vocabulary: UniquePredicateList,
                   pre_props: dict[tuple[int, int], list[Proposition] | list[list[Proposition]]],
                   eff_props: dict[tuple[int, int], list[Proposition] | list[list[Proposition]]]) \
                    -> list[ActionSchema]:
    schemata = []
    for key in pre_props:
        pre = pre_props[key]
        eff = eff_props[key]
        if len(eff) == 0:
            continue
        object_factored = True if isinstance(eff[0], list) else False
        action_schema = ActionSchema(f"p{key[0]}_{key[1]}")
        if object_factored:
            for j in range(len(pre)):
                if len(eff) == 0:
                    eff_j = []
                else:
                    eff_j = eff[j]
                action_schema = create_action_schema(action_schema, vocabulary, pre[j], eff_j, f"obj{j}")
        else:
            action_schema = create_action_schema(action_schema, vocabulary, pre, eff)
        schemata.append(action_schema)
    return schemata


def create_action_schema(action_schema: ActionSchema, vocabulary: UniquePredicateList, pre_prop: list[Proposition],
                         eff_prop: list[Proposition], obj_name: str = "") -> ActionSchema:
    # propositional case (i.e., no objects)
    if obj_name == "":
        action_schema.add_preconditions(pre_prop)
    # object-centric version
    else:
        action_schema.add_obj_preconditions(obj_name, pre_prop)

    # TODO: add probabilities...
    f_pre = set()
    f_eff = set()
    for prop in pre_prop:
        f_pre = f_pre.union(set(prop.factors))
    for prop in eff_prop:
        f_eff = f_eff.union(set(prop.factors))

    eff_neg = [x.negate() for x in pre_prop if set(x.factors).issubset(f_eff)]
    # P_nr = [x for x in vocabulary if x not in eff_prop]
    eff_proj_neg = [x for x in pre_prop if (not set(x.factors).issubset(f_eff)) and
                                           (not set(x.factors).isdisjoint(f_eff))]
    eff_proj_pos = [vocabulary.project(x, list(f_eff)) for x in eff_proj_neg]
    eff_proj_neg = [x.negate() for x in eff_proj_neg]
    # there should be projected eff propositions for dependents etc.

    # eff_neg = [x for x in vocabulary if {x.factor}.issubset(f_eff)]
    # eff_neg = [x for x in eff_neg if x not in eff_prop]
    # eff_neg = [x for x in eff_neg if not ({x.factor}.issubset(f_pre) and (x not in pre_prop))]
    # eff_neg = [x.negate() for x in eff_neg]

    if obj_name == "":
        action_schema.add_effect(eff_prop + eff_proj_pos + eff_neg + eff_proj_neg, 1)
    else:
        action_schema.add_obj_effect(obj_name, eff_prop + eff_proj_pos + eff_neg + eff_proj_neg, 1)
    return action_schema


def create_effect_clause(vocabulary: UniquePredicateList, partition: S2SDataset) \
        -> list[KernelDensityEstimator] | list[list[KernelDensityEstimator]]:
    effect_clause = []
    if partition.factors is None:
        # partition does not contain any state changes. return empty list.
        return vocabulary, []

    if partition.is_object_factored:
        for i, obj_factors in enumerate(partition.factors):
            if len(obj_factors) == 0:
                effect_clause.append([])
                continue
            densities = _create_factored_densities(partition.next_state[:, i], vocabulary, obj_factors)
            effect_clause.append(densities)
    else:
        if len(partition.factors) != 0:
            densities = _create_factored_densities(partition.next_state, vocabulary, partition.factors)
            effect_clause.extend(densities)
    return vocabulary, effect_clause


def _compute_preconditions(x: np.ndarray, y: np.ndarray, vocabulary: UniquePredicateList) \
        -> list[Proposition] | list[list[Proposition]]:
    symbol_indices = vocabulary.get_active_symbol_indices(x)
    if symbol_indices.ndim == 3:
        object_factored = True
    else:
        object_factored = False

    n = symbol_indices.shape[0]
    pos_symbols = symbol_indices[:n//2]

    # get the most frequent symbol for each factor
    mode, _ = stats.mode(pos_symbols, axis=0, keepdims=False)
    if object_factored:
        preds = [[] for _ in range(symbol_indices.shape[1])]
    else:
        preds = []

    if object_factored:
        for j in range(symbol_indices.shape[1]):
            for f_i in range(symbol_indices.shape[2]):
                sym = int(mode[j, f_i])
                mask = symbol_indices[:, j, f_i] == sym
                if np.mean(y[mask] == 1) > 0.7:
                    factor = vocabulary.factors[f_i]
                    group = vocabulary.mutex_groups[factor]
                    prop = vocabulary[group[sym]]
                    preds[j].append(prop)
    else:
        for f_i in range(symbol_indices.shape[1]):
            sym = int(mode[f_i])
            mask = symbol_indices[:, f_i] == sym
            if np.mean(y[mask] == 1) > 0.7:
                factor = vocabulary.factors[f_i]
                group = vocabulary.mutex_groups[factor]
                prop = vocabulary[group[sym]]
                preds.append(prop)
    return preds


def _create_factored_densities(data, vocabulary, factors):
    # use validation samples for independency tests
    n_val = max(int(len(data) * 0.1), 3)
    densities = []
    dependency_groups = _compute_factor_dependencies(data[-n_val:], factors)
    _n_expected_symbol = sum([2**len(x)-1 for x in dependency_groups])
    logger.info(f"n_factors={len(factors)}, n_groups={len(dependency_groups)}, n_expected_symbol={_n_expected_symbol}")
    for group in dependency_groups:
        predicate = vocabulary.append(data, group)
        densities.append(predicate)
    return densities


def _compute_factor_dependencies(data, factors, method="independent"):
    if len(factors) == 1:
        return [factors]

    if method == "gaussian":
        return _gaussian_independent_factor_groups(data, factors)
    elif method == "knn":
        return _knn_independent_factor_groups(data, factors)
    elif method == "independent":
        return [[f] for f in factors]


def _gaussian_independent_factor_groups(data, factors):
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


def _knn_independent_factor_groups(data, factors):
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


def _find_precond_propositions(vocabulary: UniquePredicateList,
                               precondition: SupportVectorClassifier) -> tuple[list[Proposition], float]:
    # in this part, express preconditions in terms of density symbols
    pre_factors = _mask_to_factors(precondition.mask, vocabulary.factors)
    # candidates are all possible proposition pairs that we need to consider
    # [(p1, p2, p3), (p4, p5), ...] where each element is a list of propositions
    # changing the same factor.
    candidates = []
    for f_i in pre_factors:
        candidates.append(vocabulary[vocabulary.mutex_groups[f_i]])

    # TODO: if such a case arises, then there is probably a factor
    # that is not being changed by any operator but is part of the precondition.
    # For those factor(s), we can fit a density estimator right here and then
    # add it to the vocabulary.
    assert len(candidates) > 0, "No candidate propositions"

    best_prob = 0
    best_cands = None

    # find the best set of candidate propositions in each factor group
    # that maximizes the probability of being in the precondition
    # i.e., represents the precondition best
    combs = product(*candidates)
    for cand_props in combs:
        cand_props = list(cand_props)
        prop_masks = sorted(list(chain.from_iterable([prop.mask for prop in cand_props])))
        assert set(prop_masks) == set(precondition.mask), \
            f"why would this happen? props: {prop_masks} pre: {precondition.mask}"

        prob = _probability_in_precondition(cand_props, precondition, allow_fill_in=True)
        if prob > best_prob:
            best_prob = prob
            best_cands = cand_props

    return best_cands, best_prob


def _fill_action_schema(action_schema: ActionSchema, precondition: SupportVectorClassifier,
                        effect: KernelDensityEstimator, eff_o: list[Proposition],
                        vocabulary: UniquePredicateList, obj_name: str = "") -> ActionSchema:
    props, prob = _find_precond_propositions(vocabulary, precondition)

    prob = round(prob, 3)
    # propositional case (i.e., no objects)
    if obj_name == "":
        action_schema.add_preconditions(props)
    # object-centric version
    else:
        action_schema.add_obj_preconditions(obj_name, props)

    success_prob = 1.0
    # add fail effect if the probability is not high enough
    if prob < 0.95:
        success_prob = prob
        action_schema.add_obj_effect(obj_name, [Proposition.not_failed().negate()], 1-success_prob)

    eff_prob = 1 * success_prob

    # 1. eff_o: Eff(o) = Inter_i Proj(X, mask(o) \ f_i) are set to true.
    # 2. Propositions with mask subseteq effect.mask are set to false.
    eff_vars = set(effect.mask)
    pre_vars = set(precondition.mask)
    neg_effects = [x for x in vocabulary if set(x.mask).issubset(eff_vars)]
    neg_effects = [x for x in neg_effects if x not in eff_o]

    # 3. Propositions with mask not subseteq effect.mask and
    #    mask intersect effect.mask is not empty are set to false.
    # ? didn't quite understand the below one.
    neg_effects = [x for x in neg_effects if not (set(x.mask).issubset(pre_vars) and (x not in props))]
    neg_effects = [x.negate() for x in neg_effects]

    if obj_name == "":
        action_schema.add_effect(eff_o + neg_effects, eff_prob)
    else:
        action_schema.add_obj_effect(obj_name, eff_o + neg_effects, eff_prob)
    return action_schema, prob


def _probability_in_precondition(propositions: list[Proposition], precondition: SupportVectorClassifier,
                                 allow_fill_in: bool = False) -> float:
    """
    Calculate the probability of samples drawn from the estimators being in the precondition.

    Parameters:
    -----------
        estimators : list[Proposition]
            The list of propositions (density estimators).
        precondition : SupportVectorClassifier
            The precondition classifier.
        allow_fill_in : bool, optional
            Whether to allow randomly filling in missing state variables. Defaults to False.

    Returns:
    --------
        prob : float
            The probability of samples drawn from the estimators being in the precondition.
    """
    mask = []
    for predicate in propositions:
        mask.extend(predicate.mask)

    # if we are not allowed to randomly sample, and we are missing state variables, then return 0
    if not allow_fill_in and not set(mask).issuperset(set(precondition.mask)):
        return 0

    keep_indices = [i for i in range(len(mask)) if mask[i] in precondition.mask]

    # Bail if no overlap.
    if len(keep_indices) == 0:
        return 0

    n_samples = 100
    samples = np.hstack([predicate.sample(n_samples) for predicate in propositions])
    samples = samples[:, keep_indices]

    # if the estimators are a subset of the precondition, randomly add data to fill in
    add_list = [m for m in precondition.mask if m not in mask]
    if len(add_list) > 0:
        if not allow_fill_in:
            return 0
        logger.info("Must randomly fill in data from {} to intersect with precondition".format(add_list))
        raise NotImplementedError

    total_mask = np.array(mask)[keep_indices]
    s_prob = 0
    for pos in range(n_samples):
        point = samples[pos, :]
        t_point = np.zeros([np.max(total_mask) + 1])
        t_point[total_mask] = point
        s_prob += precondition.probability(t_point)
    return s_prob / n_samples


def _overlapping_dists(x: KernelDensityEstimator, y: KernelDensityEstimator) -> bool:
    """
    A measure of similarity from the original paper that compares means, mins and maxes.

    Parameters:
    -----------
        x : KernelDensityEstimator
            The first distribution.
        y : KernelDensityEstimator
            The second distribution.

    Returns:
    --------
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


def _modifies(operators: list[Operator], n_variables: int) -> dict[int, list[int]]:
    """
    Determine which options modify each state variable.

    Parameters
    ----------
    operators : list[Operator]
        The list of operators.
    n_variables : int
        The number of state variables.

    Returns
    -------
    modifies : dict[int, list[tuple[int, int]]]
        For each state variable, a list of option-effect pairs that modify it.
    """
    modifies = {}
    for x in range(n_variables):
        modifying_ops = []
        for i, operator in enumerate(operators):
            for eff in operator.effect:
                if x in eff.mask:
                    modifying_ops.append(i)  # modifies[s] -> [op1, op2, ...]
                    break
        modifies[x] = modifying_ops

    return modifies


def _factorise(operators: list[Operator], n_variables: int) -> list[list[int]]:
    """
    Factorise the state space based on what variables are changed by the options.

    Parameters:
    ----------
        operators : list[Operator]
            The learned operators.
        n_variables : int
            The number of state-space variables.

    Returns:
    -------
        factors : list[list[int]]
            For each factor, the list of state variables.
    """
    modifies = _modifies(operators, n_variables)  # check which variables are modified by each operator
    factors = []
    options = []

    for i in range(n_variables):
        found = False
        for x in range(len(factors)):
            f = factors[x]
            if options[x] == modifies[i]:
                f.append(i)
                found = True

        if not found:
            factors.append([i])
            options.append(modifies[i])

    for i, f_i in enumerate(factors):
        logger.info(f"Factor {i}: {len(f_i)} # modifying options: {len(options[i])}")

    return factors


def _extract_factors(mask: list[int], factors: list[list[int]]) -> list[list[int]]:
    """
    Extract the factors referred to by the mask.

    Parameters
    ----------
    mask : list[int]
        The mask.
    factors : list[list[int]]
        The factors.

    Returns
    -------
    ret : list[list[int]]
        The extracted factors.
    """
    ret = []
    mask_set = set(mask)
    for factor in factors:
        f_set = set(factor)
        if not f_set.isdisjoint(mask_set):
            # we are sure that f_set \subseteq mask_set,
            # so the intersection is the factor itself
            part = list(f_set.intersection(mask_set))
            ret.append(sorted(part))  # masks always sorted!!
            mask_set = mask_set - f_set

            if len(mask_set) == 0:
                return ret
    return ret


def _mask_to_factors(mask: list[int], factors: list[list[int]]) -> list[int]:
    """
    Convert a mask to factors.

    Parameters
    ----------
    mask : list[int]
        The mask.
    factors : list[list[int]]
        The factors.

    Returns
    -------
    f : list[int]
        The extracted factors.
    """
    # return factors containing at least one variable present in the mask
    f = []
    for i, factor in enumerate(factors):
        if not set(mask).isdisjoint(set(factor)):
            f.append(i)
    return f


def _knn_accuracy(x, y, k=5):
    n_sample = x.shape[0]
    x = x.reshape(n_sample, -1)
    y = y.reshape(n_sample, -1)
    xy = np.concatenate([x, y], axis=0)
    dists = cdist(xy, xy) + np.eye(2*n_sample) * 1e12
    indexes = np.argsort(dists, axis=1)[:, :k]
    x_decisions = np.sum(indexes < n_sample, axis=1) / k
    x_acc = np.sum(x_decisions[:n_sample]) / n_sample
    y_acc = 1 - np.sum(x_decisions[n_sample:]) / n_sample
    return x_acc, y_acc
