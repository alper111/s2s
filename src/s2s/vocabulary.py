from itertools import chain, combinations, product
from typing import Union, Optional
import logging
from copy import deepcopy
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeClassifier, _tree

from s2s.structs import (KernelDensityEstimator, KNNDensityEstimator, UniquePredicateList,
                         Factor, Proposition, ActionSchema, S2SDataset, SupportVectorClassifier,
                         LiftedDecisionTree)

__author__ = 'Steve James and George Konidaris'
# Modified by Alper Ahmetoglu. Original source:
# https://github.com/sd-james/skills-to-symbols/tree/master

logger = logging.getLogger(__name__)


def build_vocabulary(partitions: dict[tuple[int, int], S2SDataset],
                     factors: list[Factor],
                     symbol_prefix: str = "symbol_") \
        -> tuple[UniquePredicateList,
                 dict[tuple[int, int], list[list[Proposition]]],
                 dict[tuple[int, int], list[Proposition]]]:
    """
    Build the vocabulary of propositions from the given partitions.

    Parameters
    ----------
        partitions : dict[tuple[int, int], S2SDataset]
            The partitions of the dataset.
        factors : list[Factor]
            The factors learned from the partitions.
        symbol_prefix : str
            The prefix for the symbols.

    Returns
    -------
        vocabulary : UniquePredicateList
            The vocabulary of propositions.
        pre_props : dict[tuple[int, int], list[list[Proposition]]]
            The preconditions for each partition.
        eff_props : dict[tuple[int, int], list[Proposition]]
            The effects for each partition.
    """
    # vocabulary = UniquePredicateList(_overlapping_dists)
    vocabulary = UniquePredicateList(_knn_overlapping, density_type="knn")
    pre_props = {}
    eff_props = {}

    # first add the factors to the partitions (mutates partitions!)
    add_factors_to_partitions(partitions, factors, threshold=0.01)

    # create effect symbols
    for key in partitions:
        partition_k = partitions[key]
        vocabulary, preds = create_effect_clause(vocabulary, partition_k)
        logger.info(f"Processed Eff({key[0]}-{key[1]}); {len(preds)} predicates found.")
        eff_props[key] = preds

    # merge equivalent eff_props if there is a valid
    # substitution for the object-factored case
    merge_map = {}
    if partition_k.is_object_factored:
        merge_map = merge_equivalent_effects(partitions, eff_props)

    # compute symbols over factors that are mutually exclusive.
    # important: this assumes the partition semantics, not distribution!
    vocabulary.fill_mutex_groups(factors)

    # learn constant symbols (if there are any) in case we need them for preconditions
    for f_i, factor in enumerate(vocabulary.mutex_groups):
        group = vocabulary.mutex_groups[factor]
        # no effect changes this factor, but there might be preconditions that depend on this factor.
        # so, learn a density estimator for each precondition just to be safe.
        # I think there should be only one such factor because they should be grouped together.
        if len(group) == 0:
            logger.info(f"Factor {f_i} is constant. Learning a constant symbol for each partition "
                        f"in case a precondition depends on it.")
            vocabulary = append_constant_symbols(vocabulary, partitions, factor)
    logger.info(f"Found {len(vocabulary)} unique symbols in total.")

    # now learn preconditions in terms of the vocabulary found in the previous step
    for key in partitions:
        preds = create_lifted_precondition(key, partitions, vocabulary, k_cross=20, lower_threshold=0.2)
        pre_props[key] = preds
        logger.info(f"Processed Pre({key[0]}-{key[1]}); {len(preds)} subpartitions, "
                    f"{sum([len(p) for p in preds])} predicates found in total.")
    return vocabulary, pre_props, eff_props, merge_map


def build_schemata(vocabulary: UniquePredicateList,
                   pre_props: dict[tuple[int, int], list[list[Proposition]]],
                   eff_props: dict[tuple[int, int], Union[list[Proposition], list[list[Proposition]]]]) \
                    -> list[ActionSchema]:
    """
    Build the action schemata from the given preconditions and effects.

    Parameters
    ----------
        vocabulary : UniquePredicateList
            The vocabulary of propositions.
        pre_props : dict[tuple[int, int], list[list[Proposition]]]
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
        name = "_".join([str(k) for k in key])
        action_schema = create_action_schema(f"a_{name}", vocabulary, pre, eff)
        schemata.extend(action_schema)
    return schemata


def build_typed_schemata(vocabulary: UniquePredicateList, schemata: list[ActionSchema]) -> list[ActionSchema]:
    """
    Build typed action schemata from the given action schemata.

    Parameters
    ----------
        vocabulary : UniquePredicateList
            The vocabulary of propositions.
        schemata : list[ActionSchema]
            The action schemata.

    Returns
    -------
        typed_schemata : list[ActionSchema]
            The typed action schemata.
    """
    # Find all possible symbolic transitions starting from a symbol.
    # Symbols that have the same possible set of transitions are
    # defined as the same type.
    symbol_affordances = {}
    for w in vocabulary:
        symbol_affordances[w] = set(_find_factored_symbol_affordance(w, schemata, len(vocabulary)))

    types = {}
    groups = {}
    sym_to_type = {}
    it = 0
    factor_types = defaultdict(list)
    for s in symbol_affordances:
        for t in types:
            if (symbol_affordances[s], tuple(s.factors)) == types[t]:
                groups[t].append(s)
                sym_to_type[s] = t
                break
        else:
            name = f"type{it}"
            types[name] = (symbol_affordances[s], tuple(s.factors))
            groups[name] = [s]
            sym_to_type[s] = name
            factor_types[tuple(s.factors)].append(name)
            it += 1

    all_types = []
    otype_idx = 0
    object_types = {}
    for comb in product(*tuple(factor_types.values())):
        otype = f"otype{otype_idx}"
        for t in comb:
            all_types.append((otype, t))
        otype_idx += 1
        object_types[otype] = set(comb)
    for t in types:
        all_types.append((t, 'object'))

    typed_vocabulary = UniquePredicateList(_l2_norm_overlapping, symbol_prefix=vocabulary._prefix)
    typed_schemata = []
    for w in vocabulary:
        params = [("x", sym_to_type[w])]
        typed_vocabulary.append(w.estimator._samples, w.factors, params, masked=True, forced=True)
    typed_vocabulary.fill_mutex_groups(vocabulary.factors)

    for action_schema in schemata:
        typed_action = ActionSchema(action_schema.name)
        for x in action_schema.preconditions:
            typed_x = typed_vocabulary[x.idx]
            typed_x = typed_x.substitute([(x.parameters[0][0], typed_x.parameters[0][1])])
            typed_x.sign = x.sign
            typed_action.add_preconditions([typed_x])
        for x in action_schema.effects:
            typed_x = typed_vocabulary[x.idx]
            typed_x = typed_x.substitute([(x.parameters[0][0], typed_x.parameters[0][1])])
            typed_x.sign = x.sign
            typed_action.add_effects([typed_x])
        typed_schemata.append(typed_action)
    return typed_schemata, all_types, groups, sym_to_type, object_types


def append_to_schemata(schemata, appendix):
    s_keys = {}
    a_keys = {}
    for i, s_i in enumerate(schemata):
        name = tuple(s_i.name.split("_")[1:])
        pre_id = name[2]
        key = (name[0], name[1])
        if key not in s_keys:
            s_keys[key] = []
        s_keys[key].append((i, pre_id))
    for i, s_i in enumerate(appendix):
        name = tuple(s_i.name.split("_")[1:])
        part_id = name[2]
        pre_id = name[3]
        key = (name[0], name[1])
        if key not in a_keys:
            a_keys[key] = []
        a_keys[key].append((i, part_id, pre_id))
    new_schemata = []
    for s_i in s_keys:
        base_partitions = s_keys[s_i]
        for (i, _) in base_partitions:
            if s_i not in a_keys:
                new_schemata.append(schemata[i])
            else:
                global_partitions = a_keys[s_i]
                for (j, part_j, pre_j) in global_partitions:
                    base_op = schemata[i]
                    global_op = appendix[j]
                    name = f"{base_op.name}_{part_j}_{pre_j}"
                    schema = ActionSchema(name)
                    schema.add_preconditions(base_op.preconditions)
                    schema.add_preconditions(global_op.preconditions)
                    schema.add_effects(base_op.effects)
                    schema.add_effects(global_op.effects)
                    new_schemata.append(schema)
    return new_schemata


def create_action_schema(name: str, vocabulary: UniquePredicateList, pre_prop: list[list[Proposition]],
                         eff_prop: list[Proposition]) -> list[ActionSchema]:
    """
    Create an action schema from the given preconditions and effects.

    Parameters
    ----------
        name : str
            The name of the action schema.
        vocabulary : UniquePredicateList
            The vocabulary of propositions.
        pre_prop : list[list[Proposition]]
            The preconditions.
        eff_prop : list[Proposition]
            The effects.

    Returns
    -------
        action_schemas : list[ActionSchema]
            A list of action schemas with the preconditions and effects added for each
            alternative precondition.
    """
    schemas = []
    for i, pre_i in enumerate(pre_prop):
        action_schema = ActionSchema(f"{name}_{i}")
        action_schema.add_preconditions(pre_i)

        # TODO: add probabilities...
        f_pre = set()
        f_eff = set()
        for prop in pre_i:
            for f_i in prop.factors:
                params = tuple(prop.parameters) if prop.parameters is not None else None
                f_pre.add((f_i, params))
        for prop in eff_prop:
            for f_i in prop.factors:
                params = tuple(prop.parameters) if prop.parameters is not None else None
                f_eff.add((f_i, params))

        eff_neg = []
        eff_proj_neg = []
        eff_proj_pos = []
        for x in pre_i:
            f_x = set()
            for f_i in x.factors:
                params = tuple(x.parameters) if x.parameters is not None else None
                f_x.add((f_i, params))
            if f_x.issubset(f_eff):
                eff_neg.append(x.negate())
            elif not f_x.isdisjoint(f_eff):
                eff_proj_neg.append(x.negate())
                proj_factors = [f_i for f_i, _ in f_eff]
                eff_proj_pos.append(vocabulary.project(x, proj_factors))

        # eff_neg = [x.negate() for x in pre_i if (set(x.factors).issubset(f_eff) and
        #                                          x.sign == 1)]
        # eff_proj_neg = [x for x in pre_i if (not set(x.factors).issubset(f_eff)) and
        #                                     (not set(x.factors).isdisjoint(f_eff))]
        # eff_proj_pos = [vocabulary.project(x, list(f_eff)) for x in eff_proj_neg]
        # eff_proj_neg = [x.negate() for x in eff_proj_neg if x.sign == 1]

        action_schema.add_effects(eff_prop + eff_proj_pos + eff_neg + eff_proj_neg)
        schemas.append(action_schema)
    return schemas


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


def append_constant_symbols(vocabulary: UniquePredicateList,
                            partitions: dict[tuple[int, int], S2SDataset],
                            factor: Factor) -> UniquePredicateList:
    group = vocabulary.mutex_groups[factor]
    for key in partitions:
        logger.debug(f"Learning a constant symbol for Pre({key[0]}-{key[1]}) on {factor}...")
        partition_k = partitions[key]
        if partition_k.is_object_factored:
            for j in range(partition_k.n_objects):
                vocabulary.append(partition_k.state[:, j], [factor], [("x", None)])
        else:
            vocabulary.append(partition_k.state, [factor])
    for i, pred in enumerate(vocabulary):
        if pred.factors[0] == factor:
            group.append(i)
    return vocabulary


def create_lifted_precondition(partition_key: tuple[int, int],
                               partitions: dict[tuple[int, int], S2SDataset],
                               vocabulary: UniquePredicateList,
                               k_cross: int = 50,
                               lower_threshold: float = 0.2) \
                                -> tuple[UniquePredicateList, list[list[Proposition]]]:
    pre_count = []
    p_k = partitions[partition_key]
    x_pos = p_k.state
    if x_pos.ndim == 2:
        x_pos = x_pos[:, np.newaxis, :]
        mask_indices = [0]
    else:
        mask_indices = np.where(np.any(p_k.mask.mean(axis=0) > 0.95, axis=1))[0].tolist()

    for _ in range(k_cross):
        x_neg = _generate_negative_data(partition_key, partitions, len(x_pos))
        if x_neg.ndim == 2:
            x_neg = x_neg[:, np.newaxis, :]
        y = np.concatenate([np.ones(x_pos.shape[0]), np.zeros(x_neg.shape[0])])
        x = np.concatenate([x_pos, x_neg])
        min_samples_split = max(int(len(x)*0.05), 3)
        tree = LiftedDecisionTree(vocabulary,
                                  referencable_indices=mask_indices,
                                  min_samples_split=min_samples_split)
        tree.fit(x, y)
        preconditions = tree.extract_preconditions()
        for pre in preconditions:
            found = False
            for p_i in pre_count:
                if set(pre) == set(p_i[0]):
                    p_i[1] += 1
                    found = True
                    break
            if not found:
                pre_count.append([pre, 1])
    pre_count = [p_i[0] for p_i in pre_count if p_i[1] > k_cross * lower_threshold]
    if len(pre_count) == 0:
        return [[]]
    return pre_count


def create_precondition_clause(partition_key: tuple[int, int],
                               partitions: dict[tuple[int, int], S2SDataset],
                               vocabulary: UniquePredicateList,
                               k_cross: int = 50,
                               lower_threshold: Optional[float] = None) -> list[list[Proposition]]:
    """
    Create the precondition clause for the given partition.

    Parameters
    ----------
        partition_key : tuple[int, int]
            The partition key.
        partitions : dict[tuple[int, int], S2SDataset]
            The partitions.
        vocabulary : UniquePredicateList
            The vocabulary of propositions.
        k_cross : int
            The number of cross-validation folds.
        lower_threshold : float
            The lower threshold for the number of times a precondition must be found.

    Returns
    -------
        precondition_clause : list[Proposition]
            The precondition clause.
    """
    factors = find_precondition_factors(partition_key, partitions, vocabulary,
                                        k_cross=k_cross, n_sample=100, sym_samples=10)
    if len(factors) == 0:
        logger.info(f"No factors found for Pre({partition_key[0]}-{partition_key[1]}).")
        return [[]]
    logger.info(f"Found {len(factors)} factors for Pre({partition_key[0]}-{partition_key[1]}).")

    x_pos = partitions[partition_key].state
    x_neg = _generate_negative_data(partition_key, partitions, len(x_pos))
    x = np.concatenate([x_pos, x_neg])
    n_vars = x.shape[-1]
    x = x.reshape(x.shape[0], -1)
    y = np.concatenate([np.ones(x_pos.shape[0]), np.zeros(x_neg.shape[0])])

    svm_factors = [Factor([f_i+n_vars*f[1] for f_i in f[0].variables]) for f in factors]
    svm = SupportVectorClassifier(factors=svm_factors)
    svm.fit(x, y)

    scores = svm.probability(x)
    pos_scores = scores[:len(x_pos)]
    neg_scores = scores[len(x_pos):]
    pos_mean = pos_scores.mean()
    neg_mean = neg_scores.mean()
    pos_std = pos_scores.std()
    neg_std = neg_scores.std()
    if pos_mean < 0.55:
        logger.info(f"Precondition for Pre({partition_key[0]}-{partition_key[1]}) is too weak. "
                    f"Mean: {pos_mean}, std: {pos_std}. Assuming no preconditions.")
        return [[]]
    if lower_threshold is None:
        lower_threshold = max(pos_mean-2*pos_std, (pos_mean+neg_mean)/2)
    logger.info(f"Lower threshold: {lower_threshold:.2f}, pos mean: {pos_mean:.2f}, std: {pos_std:.2f}, "
                f"neg mean: {neg_mean:.2f}, std: {neg_std:.2f}")

    candidates = []
    obj_idx = []
    for svm_f in svm_factors:
        obj_i = svm_f.variables[0] // n_vars
        obj_idx.append(obj_i)
        f = Factor([v_i % n_vars for v_i in svm_f.variables])
        candidates.append(vocabulary[vocabulary.mutex_groups[f]])
    n_loop = np.prod([len(c) for c in candidates])
    if n_loop > 1000:
        logger.info(f"Finding preconditions for {n_loop} combinations of propositions...")
    it = 0
    combs = product(*candidates)

    preconditions = []
    for cand_props in combs:
        samples = np.zeros((100, x.shape[1]))
        for o_i, p in zip(obj_idx, cand_props):
            variables = [v_i + o_i*n_vars for v_i in p.variables]
            samples[:, variables] = p.sample(100)
        y_prob = svm.probability(samples)
        if y_prob.mean() > lower_threshold:
            subbed_props = []
            for o_i, p in zip(obj_idx, cand_props):
                if p.is_grounded:
                    subbed_props.append(p)
                else:
                    subbed_props.append(p.substitute([(f"x{o_i}", None)]))
            preconditions.append(subbed_props)
        it += 1
        if n_loop > 1000:
            print(f"Processed {it}/{n_loop} combinations.", end="\r")
    return preconditions


def find_precondition_factors(partition_key: tuple[int, int],
                              partitions: dict[tuple[int, int], S2SDataset],
                              vocabulary: UniquePredicateList,
                              k_cross: int = 100,
                              n_sample: int = 100,
                              sym_samples: int = 100) -> list[Factor]:
    x_pos = partitions[partition_key].state
    n_sample = min(n_sample, len(x_pos))
    fac_count = {}
    x_arr = []
    for _ in range(k_cross):
        x_pos_ = x_pos[np.random.choice(len(x_pos), size=(n_sample,), replace=False)]
        x_neg = _generate_negative_data(partition_key, partitions, len(x_pos_))
        x = np.concatenate([x_pos_, x_neg])
        x_arr.append(x)
    x_arr = np.concatenate(x_arr)
    x_sym = vocabulary.get_active_symbol_indices(x_arr, max_samples=sym_samples)
    x_sym = x_sym.reshape(k_cross, -1, *x_sym.shape[1:])
    y = np.concatenate([np.ones(x_pos_.shape[0]), np.zeros(x_neg.shape[0])])
    for k in range(k_cross):
        factor_usage = _find_factor_usage(x_sym[k], y, vocabulary)
        # count the number of times each proposition is selected
        for factor in factor_usage:
            if factor not in fac_count:
                fac_count[factor] = 0
            fac_count[factor] += factor_usage[factor]

    factors = []
    for fac in fac_count:
        if fac_count[fac] > k_cross * 0.5:
            factors.append(fac)
    return factors


def merge_equivalent_effects(partitions: dict[tuple[int, int], S2SDataset],
                             eff_props: dict[tuple[int, int], list[Proposition]]):
    all_keys = list(partitions.keys())
    options = []
    merge_map = {}
    for (opt, _) in all_keys:
        if opt not in options:
            options.append(opt)

    for opt in options:
        for i, key_i in enumerate(all_keys):
            if key_i[0] != opt:
                continue
            if key_i not in eff_props:
                continue
            for j, key_j in enumerate(all_keys):
                if key_j[0] != opt:
                    continue
                if key_j not in eff_props:
                    continue
                if i >= j:
                    continue
                subs = _find_substitution(eff_props[key_i], eff_props[key_j])
                if len(subs) > 0:
                    logger.info(f"Merging Eff({key_i[0]}-{key_i[1]}) and Eff({key_j[0]}-{key_j[1]})...")
                    partition_i = _merge_partitions(partitions[key_i], partitions[key_j], subs)
                    partitions[key_i] = partition_i
                    merge_map[key_j] = key_i
                    # remove the merged partition
                    del partitions[key_j]
                    del eff_props[key_j]
    logger.info(f"Found {len(partitions)} partitions after merging equivalent effects.")
    n_partitions_option = {}
    for key in partitions:
        if key[0] not in n_partitions_option:
            n_partitions_option[key[0]] = 0
        n_partitions_option[key[0]] += 1
    for key in n_partitions_option:
        logger.info(f"Number of partitions for {key}={n_partitions_option[key]}.")
    return merge_map


def lift_vocabulary_and_schemata(vocabulary, schemata, vars_per_obj, max_objects):
    types = find_typed_groups(schemata, vars_per_obj, max_objects)
    lifted_vocabulary = UniquePredicateList(_knn_overlapping, density_type="knn")
    lifted_schemata = []
    lift_map = {}
    for w in vocabulary:
        objects = []
        factors = []
        prev_objects = []
        for f_i in w.factors:
            obj_idx = f_i.variables[0] // vars_per_obj
            variables = [v_i % vars_per_obj for v_i in f_i.variables]
            # for now, only works with KNNDensityEstimator
            f = Factor(variables)
            factors.append(f)
            for t in types:
                if (obj_idx in types[t]) and (obj_idx not in prev_objects):
                    objects.append(t)
                    prev_objects.append(obj_idx)
                    break
        if len(objects) != 0:
            params = [("x", f"type{t}") for t in objects]
        lifted_pred = lifted_vocabulary.append(w.estimator._samples, factors, params, masked=True)
        lift_map[w] = lifted_pred

    for rule in schemata:
        lifted_rule = ActionSchema(rule.name)
        obj_map = {}
        i = 0
        for p_i in rule.preconditions:
            current_obj = p_i.variables[0] // vars_per_obj
            if current_obj not in obj_map:
                obj_map[current_obj] = i
                i += 1
            lifted_pred = deepcopy(lift_map[p_i])
            lifted_pred.sign = p_i.sign
            temp_params = lifted_pred.parameters
            params = [(f"x{obj_map[current_obj]}", p[1]) for p in temp_params]
            lifted_pred = lifted_pred.substitute(params)
            lifted_rule.add_preconditions([lifted_pred])

        for p_i in rule.effects:
            current_obj = p_i.variables[0] // vars_per_obj
            if current_obj not in obj_map:
                obj_map[current_obj] = i
                i += 1
            lifted_pred = deepcopy(lift_map[p_i])
            lifted_pred.sign = p_i.sign
            temp_params = lifted_pred.parameters
            params = [(f"x{obj_map[current_obj]}", p[1]) for p in temp_params]
            lifted_pred = lifted_pred.substitute(params)
            lifted_rule.add_effects([lifted_pred])
        lifted_schemata.append(lifted_rule)
    return lifted_vocabulary, lifted_schemata


def find_typed_groups(schemata, vars_per_obj, n_max):
    typed_groups = {}
    options = {rule.name.split("_")[0] for rule in schemata}
    profiles = {i: _get_effect_profile(options, i, schemata, vars_per_obj) for i in range(n_max)}
    j = 0
    for i in range(n_max):
        for type_idx in typed_groups:
            an_instance = typed_groups[type_idx][0]
            if _is_profile_same(profiles[i], profiles[an_instance], vars_per_obj):
                typed_groups[type_idx].append(i)
                break
        else:
            typed_groups[j] = [i]
            j += 1
    return typed_groups


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
    dependency_groups = _compute_factor_dependencies(data[-n_val:], factors, method="gaussian")
    _n_expected_symbol = sum([2**len(x)-1 for x in dependency_groups])
    logger.info(f"n_factors={len(factors)}, n_groups={len(dependency_groups)}, n_expected_symbol={_n_expected_symbol}")
    for group in dependency_groups:
        params = None
        if lifted:
            params = [("x", None)]
        predicate = vocabulary.append(data, list(group), params)
        densities.append(predicate)
    return densities


def _find_factor_usage(symbol_indices: np.ndarray, y: np.ndarray, vocabulary: UniquePredicateList) \
        -> Union[list[Proposition], list[list[Proposition]]]:
    """
    Find the factor usage from the given data.

    Parameters
    ----------
        x : np.ndarray
            The data.
        y : np.ndarray
            The labels.
        vocabulary : UniquePredicateList
            The vocabulary of propositions.

    Returns
    -------
        factors : dict[tuple[Factor, int], float]
            The usage of the (factor, object) pairs. The object index is 0
            if the data is not object factored.
    """
    y = y.astype(int)
    n = len(y) // 2
    # symbol_indices = vocabulary.get_active_symbol_indices(x)
    if symbol_indices.ndim == 3:
        _, n_obj, n_vars = symbol_indices.shape
        symbol_indices = symbol_indices.reshape(-1, n_vars*n_obj)
    else:
        n_vars = symbol_indices.shape[1]
    tree = DecisionTreeClassifier(min_samples_leaf=max(int(n*0.01), 1))
    tree.fit(symbol_indices, y)
    pos_symbol_indices = symbol_indices[:n]
    counts = tree.decision_path(pos_symbol_indices).toarray().sum(axis=0)
    rules = _parse_tree(tree, counts)
    rules = [r_i for r_i in rules if r_i[0][0] == 1]
    factors = {}
    for r_i in rules:
        factor_covered = {}
        for decision in r_i[1:]:
            feat, _, _ = decision
            obj_i = feat // n_vars
            feat = feat % n_vars
            factor = [factor for factor in vocabulary.factors if feat in factor.variables][0]
            key = (factor, obj_i)
            if key not in factors:
                factors[key] = 0
            if key not in factor_covered:
                factor_covered[key] = 1
                factors[key] += r_i[0][1] / n
    return factors


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
            other_factors = [f for f in remaining_factors if f not in comb]
            if len(other_factors) == 0:
                independent_factor_groups.append(comb)
                break
            f_vars = list(chain.from_iterable([f.variables for f in comb]))
            other_vars = list(chain.from_iterable([f.variables for f in other_factors]))
            data_comb = data[:, f_vars]
            data_other = data[:, other_vars]
            data_real = np.concatenate([data_comb, data_other], axis=1)
            data_fake = np.concatenate([data_comb, np.random.permutation(data_other)], axis=1)
            x_acc, y_acc = _knn_accuracy(data_real, data_fake)
            if abs(x_acc-y_acc) < 0.075:
                independent_factor_groups.append(comb)
                remaining_factors = [f for f in remaining_factors if f not in comb]
                if len(remaining_factors) == 1:
                    independent_factor_groups.append(tuple(remaining_factors))
                    break
                comb_queue = [c for c in comb_queue if not set(c).intersection(set(comb))]
    if len(independent_factor_groups) == 0:
        independent_factor_groups.append(factors)
    return independent_factor_groups


def _generate_negative_data(partition_key: tuple[int, int],
                            partitions: dict[tuple[int, int], S2SDataset],
                            n_samples: int) -> np.ndarray:
    x_neg = []
    other_subgoals = [o for o in partitions if o != partition_key]
    for i in range(n_samples):
        j = np.random.randint(len(other_subgoals))
        o_ = other_subgoals[j]
        ds_neg = partitions[o_]
        sample_neg = ds_neg.state[np.random.randint(len(ds_neg.state))]
        if sample_neg.ndim == 2 and i > (n_samples // 2):
            sample_neg = sample_neg[np.random.permutation(len(sample_neg))]
        x_neg.append(sample_neg)
    return np.array(x_neg)


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


def _l2_norm_overlapping(x: KNNDensityEstimator, y: KNNDensityEstimator, threshold: float = 1) -> bool:
    """
    A measure of similarity that compares the L2 norm of the means.

    Parameters
    ----------
        x : KNNDensityEstimator
            The first distribution.
        y : KNNDensityEstimator
            The second distribution.
        threshold : float
            The threshold for the L2 norm of the means.

    Returns
    -------
        bool: True if the distributions are similar, False otherwise.
    """
    if set(x.factors) != set(y.factors):
        return False

    dat1 = x.sample(100)
    dat2 = y.sample(100)

    mean1 = np.mean(dat1, axis=0)
    mean2 = np.mean(dat2, axis=0)
    return np.linalg.norm(mean1 - mean2) < threshold


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


def _find_factored_symbol_affordance(symbol, schemata, n_symbol):
    covered = np.zeros(n_symbol, dtype=bool)
    queue = [symbol]
    while queue:
        symbol = queue.pop()
        for schema in schemata:
            for p in schema.preconditions:
                if p.idx == symbol.idx:
                    for e in schema.effects:
                        if (not covered[e.idx]) and (e.sign > 0) and \
                            (e.parameters == p.parameters) and \
                                (set(e.factors) == set(p.factors)):
                            queue.append(e)
                            covered[e.idx] = True
    indices = np.where(covered)[0].tolist()
    return indices


def _get_option_effect(option, object_idx, schemata, vars_per_obj):
    symbols = []
    for rule in schemata:
        rule_option = rule.name.split("_")[0]
        if rule_option == option and rule.effects:
            effects = []
            for e_i in rule.effects:
                if e_i.variables[0] // vars_per_obj == object_idx:
                    if e_i.sign == 1:
                        effects.append(e_i)
            if len(effects) != 0:
                symbols.append(effects)
    return symbols


def _get_effect_profile(options, object_idx, schemata, vars_per_obj):
    effect_profile = {}
    for option in options:
        effect_profile[option] = _get_option_effect(option, object_idx, schemata, vars_per_obj)
    return effect_profile


def _is_profile_same(eff_profile1, eff_profile2, vars_per_obj):
    if eff_profile1.keys() != eff_profile2.keys():
        return False

    for o in eff_profile1.keys():
        # they should have the same number of subgoals
        if len(eff_profile1[o]) != len(eff_profile2[o]):
            return False

        subgoals1 = eff_profile1[o]
        subgoals2 = eff_profile2[o]
        is_used1 = np.zeros(len(subgoals1), dtype=bool)
        is_used2 = np.zeros(len(subgoals2), dtype=bool)
        for i, sg1 in enumerate(subgoals1):
            available_indices = np.where(~is_used2)[0]
            if len(available_indices) == 0:
                return False

            for j in available_indices:
                sg2 = subgoals2[j]
                if _is_subgoal_same(sg1, sg2, vars_per_obj):
                    is_used1[i] = True
                    is_used2[j] = True
                    break
        if not is_used1.all() or not is_used2.all():
            return False
    return True


def _is_subgoal_same(subgoal1, subgoal2, vars_per_obj):
    if len(subgoal1) != len(subgoal2):
        return False

    is_used1 = np.zeros(len(subgoal1), dtype=bool)
    is_used2 = np.zeros(len(subgoal2), dtype=bool)
    for i, prop1 in enumerate(subgoal1):
        available_indices = np.where(~is_used2)[0]
        if len(available_indices) == 0:
            return False
        vars1 = {v_i % vars_per_obj for v_i in prop1.variables}

        for j in available_indices:
            prop2 = subgoal2[j]
            vars2 = {v_i % vars_per_obj for v_i in prop2.variables}
            if vars1 != vars2:
                continue

            sample1 = prop1.sample(100)
            sample2 = prop2.sample(100)
            acc1, acc2 = _knn_accuracy(sample1, sample2)
            diff1 = abs(acc1 - 0.5)
            diff2 = abs(acc2 - 0.5)
            diff = (diff1 + diff2) / 2
            if diff < 0.1:
                # they are the same
                is_used1[i] = True
                is_used2[j] = True
                break
    return is_used1.all() and is_used2.all()


def _find_substitution(props1, props2) -> dict[int, int]:
    """
    Given two lists of propositions, find a substitution
    that makes them equal if there are any.

    Parameters
    ----------
    props1 : list[Proposition]
        The first list of propositions.
    props2 : list[Proposition]
        The second list of propositions.

    Returns
    -------
    substitution : dict[int, int]
        The mapping from the indices of the first list
        to the indices of the second list if the substitution
        exists, i.e., ```props1 == [props2[i] for i in substitution]```
        is True. Otherwise, an empty dictionary is returned.
    """
    substitution = {}
    is_used1 = np.zeros(len(props1), dtype=bool)
    is_used2 = np.zeros(len(props2), dtype=bool)
    for i, prop1 in enumerate(props1):
        available_indices = np.where(~is_used2)[0]
        if len(available_indices) == 0:
            return {}

        for j in available_indices:
            prop2 = props2[j]
            if prop1.idx == prop2.idx:
                is_used1[i] = True
                is_used2[j] = True
                o_i = int(prop1.parameters[0][0][1:])
                o_j = int(prop2.parameters[0][0][1:])
                substitution[o_i] = o_j
                break
    if not is_used1.all() or not is_used2.all():
        return {}
    return substitution


def _merge_partitions_by_map(partitions: dict[tuple[int, int], S2SDataset],
                             merge_map: dict[tuple[int, int], tuple[int, int]]) -> dict[tuple[int, int], S2SDataset]:
    """
    Merge partitions by the given map.

    Parameters
    ----------
    partitions : dict[tuple[int, int], S2SDataset]
        The partitions.
    merge_map : dict[tuple[int, int], tuple[int, int]]
        The merge map.

    Returns
    -------
    merged_partitions : dict[tuple[int, int], S2SDataset]
        The merged partitions.
    """

    merged_partitions = {}
    for key in partitions:
        if key in merge_map:
            new_key = merge_map[key]
            if new_key not in merged_partitions:
                merged_partitions[new_key] = deepcopy(partitions[key])
            else:
                merged_partitions[new_key] = _merge_partitions(merged_partitions[new_key], partitions[key])
        else:
            merged_partitions[key] = deepcopy(partitions[key])
    return merged_partitions


def _merge_partitions(partition_i: S2SDataset, partition_j: S2SDataset,
                      subs: Optional[dict[int, int]] = None) -> S2SDataset:
    """
    Merge two partitions. The first partition is mutated in place.

    Parameters
    ----------
    partition_i : S2SDataset
        The first partition.
    partition_j : S2SDataset
        The second partition.
    subs : dict[int, int], optional
        The substitution mapping.

    Returns
    -------
    partition_i : S2SDataset
        The merged partition.
    """
    if subs is None:
        subs = {}
    sub_arr = []
    for i in range(partition_i.state.shape[1]):
        if i in subs:
            sub_arr.append(subs[i])
        else:
            sub_arr.append(i)
    partition_i.state = np.concatenate([partition_i.state, partition_j.state[:, sub_arr]])
    partition_i.next_state = np.concatenate([partition_i.next_state, partition_j.next_state[:, sub_arr]])
    partition_i.mask = np.concatenate([partition_i.mask, partition_j.mask[:, sub_arr]])
    partition_i.option = np.concatenate([partition_i.option, partition_j.option])
    partition_i.reward = np.concatenate([partition_i.reward, partition_j.reward])
    return partition_i


def _parse_tree(tree, counts):
    tree_ = tree.tree_

    def recurse(node, rules):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            left = rules.copy()
            right = rules.copy()
            left.append((tree_.feature[node], tree_.threshold[node], "<="))
            right.append((tree_.feature[node], tree_.threshold[node], ">"))
            rules_from_left = recurse(tree_.children_left[node], left)
            rules_from_right = recurse(tree_.children_right[node], right)
            rules = rules_from_left + rules_from_right
            return rules
        else:
            leaf = rules.copy()
            leaf.insert(0, (np.argmax(tree_.value[node][0]), counts[node]))
            return [leaf]
    return recurse(0, [])
