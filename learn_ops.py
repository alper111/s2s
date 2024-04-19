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


def _learn_preconditions(subgoals: dict[tuple[int, int], S2SDataset]) -> dict[tuple[int, int], SupportVectorClassifier]:
    options = list(subgoals.keys())
    preconditions = {}

    ###
    # hard code just for testing
    # TODO: definitely remove this
    rng = np.arange(84*84).reshape(84, 84)
    factors = [
        rng[:28, :28].flatten().tolist(),
        rng[:28, 28:56].flatten().tolist(),
        rng[:28, 56:].flatten().tolist(),
        rng[28:56, :28].flatten().tolist(),
        rng[28:56, 28:56].flatten().tolist(),
        rng[28:56, 56:].flatten().tolist(),
        rng[56:, :28].flatten().tolist(),
        rng[56:, 28:56].flatten().tolist(),
        rng[56:, 56:].flatten().tolist()
    ]
    ###

    for option in options:
        dataset = subgoals[option]
        mask = np.where(np.any(dataset.mask, axis=0))[0].tolist()
        if len(mask) == 0:
            print(f"Skipping option {option} because it has no mask")
            continue

        ###
        # TODO: remove this
        found_factor = False
        for factor_i in factors:
            for factor_j in factors:
                factor = factor_i + factor_j
                if set(mask).issubset(set(factor)):
                    mask = factor
                    found_factor = True
                    break
        if not found_factor:
            raise ValueError(f"Could not find a factor for option {option}")
        ###

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


def _learn_effects(subgoals: dict[tuple[int, int], S2SDataset]) -> dict[tuple[int, int], KernelDensityEstimator]:
    options = list(subgoals.keys())
    effects = {}

    ###
    # hard code just for testing
    # TODO: definitely remove this
    rng = np.arange(84*84).reshape(84, 84)
    factors = [
        rng[:28, :28].flatten().tolist(),
        rng[:28, 28:56].flatten().tolist(),
        rng[:28, 56:].flatten().tolist(),
        rng[28:56, :28].flatten().tolist(),
        rng[28:56, 28:56].flatten().tolist(),
        rng[28:56, 56:].flatten().tolist(),
        rng[56:, :28].flatten().tolist(),
        rng[56:, 28:56].flatten().tolist(),
        rng[56:, 56:].flatten().tolist()
    ]
    ###

    for option in options:
        # TODO: dataset will be a list in the probabilistic setting
        dataset = subgoals[option]
        mask = np.where(np.any(dataset.mask, axis=0))[0].tolist()
        if len(mask) == 0:
            print(f"Skipping option {option} because it has no mask")
            continue

        ###
        # TODO: remove this
        found_factor = False
        for factor_i in factors:
            for factor_j in factors:
                factor = factor_i + factor_j
                if set(mask).issubset(set(factor)):
                    mask = factor
                    found_factor = True
                    break
        if not found_factor:
            raise ValueError(f"Could not find a factor for option {option}")
        ###
        data = dataset.next_state
        kde = KernelDensityEstimator(mask)
        kde.fit(data)
        effects[option] = kde

    return effects


def _combine(preconditions: dict[tuple[int, int], SupportVectorClassifier],
             effects: dict[tuple[int, int], KernelDensityEstimator]) -> list[Operator]:
    assert sorted(preconditions.keys()) == sorted(effects.keys())
    operators = []
    for (option, partition) in preconditions:
        precond = preconditions[(option, partition)]
        effect = effects[(option, partition)]
        operators.append(Operator(option, partition, precond, effect))
    return operators
