from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist

from data import S2SDataset


def partition_to_subgoal(dataset: S2SDataset) -> dict[tuple[int, int], S2SDataset]:
    """
    Partition a dataset such that each partition corresponds to a subgoal.

    Parameters
    ----------
    dataset : S2SDataset
        A dataset to be partitioned.

    Returns
    -------
    partitions : dict[tuple[int, int], S2SDataset]
        A dictionary of partitioned datasets with keys as options and subgoals.
    """
    partitions = {}

    # split by options
    option_partitions = _split_by_options(dataset)

    # partition each option by mask and abstract effect
    for o_i, partition_k in option_partitions.items():
        # compute masked effect
        abstract_effect = (partition_k.next_state * partition_k.mask)

        # partition by options' mask
        mask_partitions = _partition(partition_k.mask.astype(float))

        # partition by abstract effect
        it = 0
        for mask in mask_partitions:
            idx_i = mask_partitions[mask]
            eff_i = abstract_effect[idx_i]
            eff_partitions = _partition(eff_i)
            for eff in eff_partitions:
                idx_j = eff_partitions[eff]
                partition = S2SDataset(
                    partition_k.state[idx_i][idx_j],
                    partition_k.option[idx_i][idx_j],
                    partition_k.reward[idx_i][idx_j],
                    partition_k.next_state[idx_i][idx_j],
                    partition_k.mask[idx_i][idx_j]
                )
                partitions[(o_i, it)] = partition
                it += 1

    # TODO: merge partitions with intersecting initiation sets
    # partitions = _merge_partitions(partitions)
    return partitions


def _split_by_options(dataset: S2SDataset) -> dict[int, S2SDataset]:
    """
    Split a dataset by options.

    Parameters
    ----------
    dataset : S2SDataset
        A dataset to be split.

    Returns
    -------
    datasets : list[S2SDataset]
        A list of datasets split by options.
    """
    datasets = defaultdict(list)
    for i in range(len(dataset.state)):
        o_i = dataset.option[i]
        datasets[o_i].append(i)
    datasets = {k: S2SDataset(
        dataset.state[v],
        dataset.option[v],
        dataset.reward[v],
        dataset.next_state[v],
        dataset.mask[v]
    ) for k, v in datasets.items()}
    return datasets


def _merge_partitions(partitions: list[S2SDataset]) -> list[S2SDataset]:
    """
    Merge two partitions if their initiation sets intersect above a threshold.
    These partitions will introduce probabilistic effects.

    Parameters
    ----------
    partitions : list[S2SDataset]
        A list of partitioned datasets.

    Returns
    -------
    merged_partitions : list[S2SDataset]
        A list of merged datasets.
    """
    raise NotImplementedError


def _partition(x: np.ndarray) -> dict[int, list]:
    """
    Partition a given numpy array with x-means clustering.

    Parameters
    ----------
    x : np.ndarray
        A numpy array to be partitioned.

    Returns
    -------
    partitions : dict[int, list]
        A dictionary of partitioned indices.
    """
    # TODO: make this portion deep?
    _, labels, _ = _x_means(x, 1, 50)
    partitions = {}
    for i in range(max(labels) + 1):
        indices = np.where(labels == i)[0].tolist()
        if len(indices) > 0:
            partitions[i] = indices
    return partitions


def _k_means(x: np.ndarray, k: int, centroids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Perform k-means clustering.

    Parameters
    ----------
    x : np.ndarray
        A numpy array to be clustered.
    k : int
        The number of clusters.
    centroids : np.ndarray, optional
        Initial centroids.

    Returns
    -------
    centroids : np.ndarray
        The final centroids.
    assigns : np.ndarray
        The cluster assignments.
    mse : float
        The mean squared error.
    """
    if centroids is None:
        centroids = np.zeros((k, x.shape[1]))
        prev_assigns = np.random.randint(0, k, x.shape[0])
    else:
        distances = cdist(centroids, x)
        prev_assigns = np.argmin(distances, axis=0)
    while True:
        for i in range(k):
            if (prev_assigns == i).sum() > 0:
                centroids[i] = x[prev_assigns == i].mean(axis=0)
        distances = cdist(centroids, x)
        next_assigns = np.argmin(distances, axis=0)
        if (next_assigns == prev_assigns).all():
            break
        else:
            prev_assigns = next_assigns
    mse = distances[next_assigns, np.arange(x.shape[0])].mean()
    return centroids, next_assigns, mse


def _bic(x: np.ndarray, mu: np.ndarray) -> float:
    """
    Compute the Bayesian Information Criterion (BIC).

    Parameters
    ----------
    x : np.ndarray
        A numpy array of data points.
    mu : np.ndarray
        A numpy array of centroids.

    Returns
    -------
    bic : float
        The Bayesian Information Criterion.
    """
    K = mu.shape[0]
    M = mu.shape[1]
    N = x.shape[0]
    distances = cdist(mu, x)
    assigns = np.argmin(distances, axis=0)
    R = np.zeros(K)
    for i in range(K):
        R[i] = (assigns == i).sum()
    assert R.sum() == N
    squared_error = distances[assigns, np.arange(N)].sum()
    variance = squared_error / (M*(N-K))
    logl = (R * np.log(R+1e-12) - R * np.log(N) - 0.5 * R * M *
            np.log(2 * np.pi * variance+1e-12) - 0.5 * (R-1) / M).sum()
    complexity = 0.5 * K * (M+1) * np.log(N)
    bic = logl - complexity
    return bic


def _extend_or_keep(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Extend a cluster or keep it as is.

    Parameters
    ----------
    x : np.ndarray
        A numpy array of data points.
    mu : np.ndarray
        A numpy array of centroids.

    Returns
    -------
    new_centroids : np.ndarray
        The new centroids.
    """
    radius = np.linalg.norm(x-mu, axis=1).max()
    direction = np.random.randn(x.shape[1])
    direction /= np.linalg.norm(direction)
    child_centroids = np.zeros((2, x.shape[1]))
    child_centroids[0] = mu + radius * direction
    child_centroids[1] = mu - radius * direction
    child_centroids, _, _ = _k_means(x, 2, centroids=child_centroids)
    parent_bic = _bic(x, mu)
    children_bic = _bic(x, child_centroids)
    if parent_bic > children_bic:
        return mu
    else:
        return child_centroids


def _x_means(x: np.ndarray, k_min: int, k_max: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Perform x-means clustering.

    Parameters
    ----------
    x : np.ndarray
        A numpy array to be clustered.
    k_min : int
        The minimum number of clusters.
    k_max : int
        The maximum number of clusters.

    Returns
    -------
    centroids : np.ndarray
        The final centroids.
    assigns : np.ndarray
        The cluster assignments.
    mse : float
        The mean squared error.
    """
    k = k_min
    assert k_min < k_max and k_min > 0
    while k < (k_max+1):
        if k == k_min:
            centroids = None
        centroids, assigns, _ = _k_means(x, k, centroids=centroids)
        centroid_list = []
        change = False
        for i in range(k):
            if (assigns == i).sum() < 1:
                continue
            new_centroid = _extend_or_keep(x[(assigns == i)], centroids[i].reshape(1, -1))
            for m in new_centroid:
                centroid_list.append(m.tolist())
            if new_centroid.shape[0] == 2:
                k += 1
                change = True
            if k == (k_max+1):
                break
        if not change:
            break
        centroids = np.array(centroid_list)
    distances = cdist(centroids, x)
    assigns = np.argmin(distances, axis=0)
    mse = distances[assigns, np.arange(x.shape[0])].mean()
    return centroids, assigns, mse
