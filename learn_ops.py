import numpy as np

from structs import S2SDataset, SupportVectorClassifier, KernelDensityEstimator, Operator


__author__ = 'Steve James and George Konidaris'
# Modified by Alper Ahmetoglu. Original source:
# https://github.com/sd-james/skills-to-symbols/tree/master


def learn_operators(subgoals: dict[tuple[int, int], S2SDataset]) -> list[Operator]:
    preconditions = _learn_preconditions(subgoals)
    effects = _learn_effects(subgoals)
    operators = _combine(preconditions, effects)
    return operators


def _learn_preconditions(subgoals: dict[tuple[int, int], S2SDataset]) -> \
        dict[tuple[int, int], SupportVectorClassifier | list[SupportVectorClassifier]]:
    options = list(subgoals.keys())
    preconditions = {}

    for option in options:
        dataset = subgoals[option]
        any_mask = np.any(dataset.mask, axis=0)
        if any_mask.ndim == 2:
            obj_mask, feat_mask = np.where(any_mask)
            n_objs = len(obj_mask)
        else:
            feat_mask, = np.where(any_mask)

        if len(feat_mask) == 0:
            print(f"Skipping option {option} because it has no mask")
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

        if any_mask.ndim == 1:
            mask = np.where(any_mask)[0].tolist()
            x = np.concatenate([data_pos, data_neg])
            y = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])
            svm = SupportVectorClassifier(mask)
            svm.fit(x, y)
            preconditions[option] = svm
        else:
            preconditions[option] = []
            for i in range(n_objs):
                mask = np.where(any_mask[i])[0].tolist()
                x = np.concatenate([data_pos[:, i], data_neg[:, i]])
                y = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])
                svm = SupportVectorClassifier(mask)
                svm.fit(x, y)
                preconditions[option].append(svm)

    return preconditions


def _learn_effects(subgoals: dict[tuple[int, int], S2SDataset]) -> \
        dict[tuple[int, int], KernelDensityEstimator | list[KernelDensityEstimator]]:
    options = list(subgoals.keys())
    effects = {}

    for option in options:
        # TODO: dataset will be a list in the probabilistic setting
        dataset = subgoals[option]
        any_mask = np.any(dataset.mask, axis=0)
        if any_mask.ndim == 2:
            obj_mask, feat_mask = np.where(any_mask)
            n_objs = len(obj_mask)
        else:
            feat_mask, = np.where(any_mask)

        if len(feat_mask) == 0:
            print(f"Skipping option {option} because it has no mask")
            continue

        if any_mask.ndim == 1:
            mask = np.where(any_mask)[0].tolist()
            data = dataset.next_state
            kde = KernelDensityEstimator(mask)
            kde.fit(data)
            effects[option] = kde
        else:
            effects[option] = []
            for i in range(n_objs):
                mask = np.where(any_mask[i])[0].tolist()
                data = dataset.next_state[:, i]
                kde = KernelDensityEstimator(mask)
                kde.fit(data)
                effects[option].append(kde)

    return effects


def _combine(preconditions: dict[tuple[int, int], SupportVectorClassifier | list[SupportVectorClassifier]],
             effects: dict[tuple[int, int], KernelDensityEstimator | list[KernelDensityEstimator]]) -> list[Operator]:
    assert sorted(preconditions.keys()) == sorted(effects.keys())
    operators = []
    for (option, partition) in preconditions:
        precond = preconditions[(option, partition)]
        effect = effects[(option, partition)]
        operators.append(Operator(option, partition, precond, effect))
    return operators
