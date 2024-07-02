from collections import defaultdict
import logging

from s2s.structs import S2SDataset, Factor

__author__ = 'Steve James and George Konidaris'
# Modified by Alper Ahmetoglu. Original source:
# https://github.com/sd-james/skills-to-symbols/tree/master
logger = logging.getLogger(__name__)


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
            if (avg_mask[..., x] > threshold).any():
                modifies[x].append(p_i)  # modifies[s] -> [(o1, p1), (o2, p2), ...]
    return modifies
