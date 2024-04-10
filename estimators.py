import numpy as np

from structs import S2SDataset, SupportVectorClassifier, KernelDensityEstimator


__author__ = 'Steve James and George Konidaris'
# Modified by Alper Ahmetoglu. Original source:
# https://github.com/sd-james/skills-to-symbols/tree/master


def learn_preconditions(subgoals: dict[tuple[int, int], S2SDataset]) -> dict[tuple[int, int], SupportVectorClassifier]:
    options = list(subgoals.keys())
    preconditions = {}
    for option in options:
        dataset = subgoals[option]
        mask = np.where(np.any(dataset.mask, axis=0))[0].tolist()
        if len(mask) == 0:
            print(f"Skipping option {option} because it has no mask")
            preconditions[option] = None
            continue

        data_pos = dataset.state
        n_pos = len(data_pos)

        other_options = [o for o in options if o != option]
        data_neg = []
        for _ in range(n_pos):
            j = np.random.randint(len(other_options))
            o_ = other_options[j]
            ds_neg = subgoals[o_]
            sample_neg = ds_neg.state[np.random.randint(len(ds_neg.state))]
            data_neg.append(sample_neg)
        data_neg = np.array(data_neg)

        x = np.concatenate([data_pos, data_neg])
        y = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])
        svm = SupportVectorClassifier(mask)
        svm.fit(x, y)
        preconditions[option] = svm

    return preconditions


def learn_effects(subgoals: dict[tuple[int, int], S2SDataset]) -> dict[tuple[int, int], KernelDensityEstimator]:
    options = list(subgoals.keys())
    effects = {}
    for option in options:
        # TODO: dataset might be a list in the probabilistic setting
        dataset = subgoals[option]
        mask = np.where(np.any(dataset.mask, axis=0))[0].tolist()
        if len(mask) == 0:
            print(f"Skipping option {option} because it has no mask")
            effects[option] = None
            continue

        data = dataset.next_state
        kde = KernelDensityEstimator(mask)
        kde.fit(data)
        effects[option] = kde

    return effects
