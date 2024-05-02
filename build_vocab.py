from itertools import combinations, product, chain

import numpy as np

from structs import (KernelDensityEstimator, Operator, UniquePredicateList,
                     Proposition, ActionSchema, SupportVectorClassifier)

__author__ = 'Steve James and George Konidaris'
# Modified by Alper Ahmetoglu. Original source:
# https://github.com/sd-james/skills-to-symbols/tree/master


def build_vocabulary(operators: list[Operator], n_variables: int) \
               -> tuple[UniquePredicateList,
                        dict[tuple[int, int], list[Proposition] | list[list[Proposition]]]]:
    vocabulary = UniquePredicateList(_overlapping_dists)
    factors = _factorise(operators, n_variables)
    print(f"Number of factors={len(factors)}")

    propositions = {}
    for i, operator in enumerate(operators):
        print(f"Processing operator {i}")
        predicates = []
        if isinstance(operator.effect, KernelDensityEstimator):
            vocabulary, preds = _compute_possible_projections(vocabulary, operator.effect, factors)
            predicates.extend(preds)
        else:
            for op_obj_eff in operator.effect:
                vocabulary, preds = _compute_possible_projections(vocabulary, op_obj_eff, factors)
                predicates.append(preds)
        propositions[(operator.option, operator.partition)] = predicates
    vocabulary.fill_mutex_groups(factors)
    return vocabulary, propositions


def build_schemata(operators: list[Operator], vocabulary: UniquePredicateList,
                   op_propositions: dict[tuple[int, int], list[Proposition] | list[list[Proposition]]]) -> \
                    list[ActionSchema]:
    schemata = []
    for operator in operators:
        action_schema = ActionSchema(operator)

        total_prob = 1.0
        if isinstance(operator.precondition, list):
            for i, obj_pre in enumerate(operator.precondition):
                name = f"x{i}"
                # TODO: consider something else if objects in the precondition
                # do not match the objects in the effect.
                # (don't know if such a case can happen)
                eff_o = op_propositions[(operator.option, operator.partition)][i]
                effect = operator.effect[i]
                action_schema, prob = _fill_action_schema(action_schema, obj_pre, effect, eff_o, vocabulary, name)
                total_prob *= prob
        else:
            eff_o = op_propositions[(operator.option, operator.partition)]
            effect = operator.effect
            action_schema, prob = _fill_action_schema(action_schema, operator.precondition, effect, eff_o, vocabulary)
            total_prob *= prob

        print(f"Best found with p={total_prob:.3f} for {operator.option}-{operator.partition}")
        schemata.append(action_schema)

    return schemata


def _compute_possible_projections(vocabulary: UniquePredicateList, effect: KernelDensityEstimator,
                                  factors: list[list[int]]) -> list[KernelDensityEstimator]:
    projections = []
    op_factors = _extract_factors(effect.mask, factors)
    # single factor -- the graph case
    if len(op_factors) == 1:
        predicate = vocabulary.append(effect)
        projections.append(predicate)
    # the below combination is only valid if factors are independent.
    # if factors are not independent, we need to consider all possible combinations.
    # i.e., for comb_size in range(1, len(op_factors)-1):
    #           for subset in combinations(op_factors, comb_size):
    else:
        comb_size = len(op_factors) - 1
        for subset in combinations(op_factors, comb_size):
            variable_subset = np.concatenate(subset).tolist()
            marginal = effect.integrate_out(variable_subset)
            predicate = vocabulary.append(marginal)
            projections.append(predicate)
    return vocabulary, projections


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
        print("Must randomly fill in data from {} to intersect with precondition".format(add_list))
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
    if set(x.mask) != set(y.mask):
        return False

    dat1 = x.sample(100)
    dat2 = y.sample(100)

    mean1 = np.mean(dat1)
    mean2 = np.mean(dat2)
    if np.linalg.norm(mean1 - mean2) > 0.1:
        return False

    ndims = len(x.mask)
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
        print(f"Factor {i}: {len(f_i)} # modifying options: {len(options[i])}")

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
