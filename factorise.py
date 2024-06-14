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

    factors = [Factor(f) for f in factors]

    return factors


def add_factors_to_partitions(partitions: dict[tuple[int, int], S2SDataset], factors: list[Factor],
                              threshold: float = 0.9):
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
