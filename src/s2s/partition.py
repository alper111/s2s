from collections import defaultdict
from typing import Optional
import logging

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from s2s.structs import S2SDataset, sort_dataset

logger = logging.getLogger(__name__)


def partition_discrete_set(dataset: S2SDataset, min_samples: int = 1) -> dict[tuple[int, int], S2SDataset]:
    """
    Partition a discrete dataset created by Markov state abstractions such that
    each partitions corresponds to a subgoal.

    Parameters
    ----------
    dataset : S2SDataset
        A discrete dataset to be partitioned.
    min_samples : int, optional
        The minimum number of samples for a partition to be considered as subgoal.
        Default is 1.

    Returns
    -------
    partitions : dict[tuple[int, int], S2SDataset]
        A dictionary of partitioned datasets with keys as option-subgoals.
    """
    partitions = {}

    # split by options
    option_partitions = _split_by_options(dataset)

    # partition each option by its abstract effect
    for o_i, partition_k in option_partitions.items():
        unique_effects = {}
        for j in range(len(partition_k)):
            m_ij = partition_k.mask[j]
            e_ij = partition_k.next_state[j]
            e_str = []
            for ri, row in enumerate(m_ij):
                row_str = []
                for ci, digit in enumerate(row):
                    if digit:
                        # this is finite since we assume a discrete set
                        row_str.append(str(e_ij[ri, ci]))
                    else:
                        # meaning this digit has not been changed
                        row_str.append('x')
                row_str = "".join(row_str)
                e_str.append(row_str)
            indices = [x[0] for x in sorted(enumerate(e_str), key=lambda x: x[1])]
            e_sorted = [e_str[x] for x in indices]
            key = tuple([x for x in e_sorted if x != 'x'*len(x)])
            if key not in unique_effects:
                unique_effects[key] = {
                    "state": [],
                    "option": [],
                    "reward": [],
                    "next_state": [],
                    "mask": []
                }

            s, o, r, s_, m = partition_k[j]
            subgoal = unique_effects[key]
            subgoal["state"].append(s[indices])
            subgoal["option"].append(o)
            subgoal["reward"].append(r)
            subgoal["next_state"].append(s_[indices])
            subgoal["mask"].append(m[indices])

        for p_i, key in enumerate(unique_effects):
            subgoal = unique_effects[key]
            partition = S2SDataset(
                np.stack(subgoal["state"]),
                np.array(subgoal["option"]),
                np.array(subgoal["reward"]),
                np.stack(subgoal["next_state"]),
                np.stack(subgoal["mask"])
            )
            partitions[(o_i, p_i)] = partition

    if min_samples > 1:
        partitions_filtered = {}
        for p_i in partitions:
            if len(partitions[p_i]) > min_samples:
                partitions_filtered[p_i] = partitions[p_i]
        return partitions_filtered

    return partitions


def partition_to_subgoal(dataset: S2SDataset,
                         other_dataset: Optional[S2SDataset] = None,
                         eps: float = 0.5,
                         mask_eps: float = 0.1,
                         min_samples: Union[int, float] = 10,
                         mask_threshold: float = 0.05) \
                         -> tuple[dict[tuple[int, int], S2SDataset],
                                  dict[tuple[int, int], S2SDataset]]:
    """
    Partition a dataset such that each partition corresponds to a subgoal.

    Parameters
    ----------
    dataset : S2SDataset
        A dataset to be partitioned.
    other_dataset : S2SDataset, optional
        Optional dataset to be partitioned with the same subgoals.
    eps : float, optional
        The epsilon value for DBSCAN clustering. Default is 0.5.
    mask_eps : float, optional
        The epsilon value for DBSCAN clustering on the mask. Default is 0.1.
    min_samples : Union[int, float], optional
        The minimum number of samples for a partition to be considered as subgoal.
        If float, it is the fraction of the dataset. Otherwise, it is the number of samples.
    mask_threshold : float, optional
        The threshold for the change mask to be considered as a
        changing effect. Default is 0.05.

    Returns
    -------
    partitions : dict[tuple[int, int], S2SDataset]
        A dictionary of partitioned datasets with keys as option-subgoals.
    partitions_other : dict[tuple[int, int], S2SDataset]
        A dictionary of partitioned datasets from the other dataset.
    """
    partitions = {}
    other_partitions = {}

    # split by options
    option_partitions = _split_by_options(dataset)
    if other_dataset is not None:
        option_partitions_other = _split_by_options(other_dataset)

    # partition each option by mask and abstract effect
    for o_i, partition_i in option_partitions.items():
        partition_i_sorted = sort_dataset(partition_i)
        flat_mask = partition_i_sorted.mask.reshape(len(partition_i_sorted.mask), -1).astype(float)
        # eps is fixed to 0.1 because the mask is binary
        # and we want a fine-grained partition
        mask_partitions, mask_centroids = _partition(flat_mask, eps=mask_eps, min_samples=min_samples)
        it = 0
        for m_i, m_idx in mask_partitions.items():
            mask = mask_centroids[m_i] > mask_threshold
            if not any(mask):
                continue
            partition_ij = S2SDataset(*partition_i_sorted[m_idx])
            mask_reshaped = mask.reshape(partition_ij.mask[:].shape[1:])
            partition_ij.mask[:] = mask_reshaped
            abstract_effect = partition_ij.next_state[:, mask_reshaped]
            abs_eff_partitions, _ = _partition(abstract_effect, eps, min_samples)
            for _, e_idx in abs_eff_partitions.items():
                partition_ijk = S2SDataset(*partition_ij[e_idx])
                partitions[(o_i, it)] = partition_ijk
                if other_dataset is not None:
                    other_i = option_partitions_other[o_i]
                    other_ij = S2SDataset(*other_i[m_idx])
                    other_ijk = S2SDataset(*other_ij[e_idx])
                    other_partitions[(o_i, it)] = other_ijk
                it += 1

        logger.info(f"Option {o_i} has {it} abstract effects.")

    # TODO: merge partitions with intersecting initiation sets
    # partitions = _merge_partitions(partitions)
    return partitions, other_partitions


def _split_by_options(dataset: S2SDataset) -> dict[int, S2SDataset]:
    """
    Split a dataset by options.

    Parameters
    ----------
    dataset : S2SDataset
        A dataset to be split.

    Returns
    -------
    datasets : dict[int, S2SDataset]
        A list of datasets split by options.
    """
    datasets = defaultdict(list)
    for i in range(len(dataset.state)):
        o_i = dataset.option[i]
        # if the option contains object index, just use the option index
        if isinstance(o_i, np.ndarray):
            o_i = o_i[0]
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


def _partition(x: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> dict[int, list]:
    """
    Partition a given numpy array with x-means clustering.

    Parameters
    ----------
    x : np.ndarray
        A numpy array to be partitioned.
    eps : float, optional
        The epsilon value for DBSCAN clustering. Default is 0.5.

    Returns
    -------
    partitions : dict[int, list]
        A dictionary of partitioned indices.
    centroids : np.ndarray
        The final centroids.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
    labels = clustering.labels_

    partitions = {}
    centroids = {}
    for i in range(max(labels) + 1):
        indices = np.where(labels == i)[0].tolist()
        if len(indices) > 0:
            partitions[i] = indices
            centroids[i] = x[indices].mean(axis=0)
    return partitions, centroids


def _k_means(x: np.ndarray, k: int, centroids: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, float]:
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
    child_centroids, assigns, _ = _k_means(x, 2, centroids=child_centroids)

    # require at least 10 points in each cluster
    if (assigns == 0).sum() < 10 or (assigns == 1).sum() < 10:
        return mu

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
