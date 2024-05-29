from collections import defaultdict

from structs import S2SDataset, Factor


def factors_from_partitions(partitions: dict[tuple[int, int], S2SDataset], threshold: float = 0.9) \
        -> list[Factor]:
    """
    Factorise the state space based on what variables are changed by the options induced
    by the subgoal partitions. This function mutates the partitions in place by adding
    a list of factors modified by each partition.

    Parameters
    ----------
    partitions : dict[tuple[int, int], S2SDataset]
        Subgoal partitions.
    threshold : float
        Minimum proportion of samples that must be modified
        for a partition to be considered a modifier.

    Returns
    -------
    factors : list[Factor]
        Factors that represent the state space.
    """
    modifies = _modifies(partitions, threshold)
    partition_keys = list(partitions.keys())
    factors = []
    options = []
    n_variables = partitions[partition_keys[0]].mask.shape[-1]
    empty_factor_idx = None

    for i in range(n_variables):
        found = False
        for x in range(len(factors)):
            f = factors[x]
            if options[x] == modifies[i]:
                f.append(i)
                found = True
                break
        if not found:
            factors.append([i])
            options.append(modifies[i])
            if len(modifies[i]) == 0:
                empty_factor_idx = len(factors) - 1

    factors = [Factor(f) for f in factors]

    # mutates partitions!
    for f_i, modif_opts in zip(factors, options):
        for o_i in modif_opts:
            if partitions[o_i].factors is None:
                partitions[o_i].factors = []
            partitions[o_i].factors.append(f_i)
    # fill partitions that are not modified with the empty factor
    if empty_factor_idx is not None:
        for p_i in partition_keys:
            if partitions[p_i].factors is None:
                partitions[p_i].factors = [factors[empty_factor_idx]]

    return factors


def _modifies(partitions: dict[tuple[int, int], S2SDataset], threshold: float = 0.9) \
        -> dict[int, list[tuple[int, int]]]:
    """
    Determine which partitions modify each state variable.

    Parameters
    ----------
    partitions : dict[tuple[int, int], S2SDataset]
        Subgoal partitions.
    threshold : float
        Minimum proportion of samples that must be modified
        for a partition to be considered a modifier.

    Returns
    -------
    modifies : dict[int, list[tuple[int, int]]]
        For each state variable, a list of option-effect pairs that modify it.
    """
    partition_keys = list(partitions.keys())
    n_variables = partitions[partition_keys[0]].mask.shape[-1]
    modifies = defaultdict(list)
    for p_i in partition_keys:
        partition = partitions[p_i]
        avg_mask = partition.mask.mean(axis=0)
        for x in range(n_variables):
            if avg_mask[x] > threshold:
                modifies[x].append(p_i)  # modifies[s] -> [(o1, p1), (o2, p2), ...]
    return modifies
