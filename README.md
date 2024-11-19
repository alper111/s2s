# Markov State Abstractions + Skills to Symbols
This repository contains the implementation of core methods in [From Skills to Symbols: Learning Symbolic Representations for Abstract High-Level Planning](https://jair.org/index.php/jair/article/view/11175) [[1](#1)] and [Markov State Abstractions](https://arxiv.org/abs/2106.04379) [[2](#2)]. Skills to symbols implementation is a modified version of [Steve's implementation](https://github.com/sd-james/skills-to-symbols).

## Install requirements
```
conda create -n s2s python=3.9.20 setuptools=65.5.0 wheel=0.38.4 pip=24.0
conda activate s2s
pip install -r requirements.txt
```

## Outline of the code
```
|-- docs
|-- src
    |-- abstraction    # Markov state abstraction
    |-- environments   # Environment and dataset definitions
    |-- s2s            # Skills to symbols core
    agent.py           # Agent class that wraps s2s and MSA
```

## Prepare the dataset
The codebase expects a dataset class derived from `s2s.structs.FlatDataset` for fixed-size observations and `s2s.structs.UnorderedDataset` for object-factored representations.

### Flat Dataset
Save your $(s, a, s')$ transitions in `state.npy`, `action.pkl`, `next_state.npy` with the same order. `state.npy` and `next_state.npy` should be 2d `np.ndarray`s, and `action.pkl` should be a list of executed actions. You can modify these and extend `s2s.structs.FlatDataset` class based on your requirements.

### Unordered Dataset
This is for cases where we have access to object-factored observations and can track objects across timesteps. The only difference is the state representation:
```json
state = {
    "modality1": {
        entity1_id: np.ndarray,
        entity2_id: np.ndarray,
        ...
    },
    "modality2": {
        entity1_id: np.ndarray,
        entity3_id: np.ndarray
    },
    ...
    "global": {0: nd.array}
    "dimensions": {
        "modality1": 3072,
        "modality2": 128,
        ...
        "global": 3
    }
}
```
The code expects that entity ids are the same in a single $(s, a, s')$ tuple. If there is a key mismatch, the code treats that difference as a removal or addition of a new element and creates skolem objects. For instance in Minecraft, imagine we place a block from the inventory and there is a new block object in the environment, and the item in the inventory is removed. The dictionaries will be as follows.
```python
state = {
  "objects": {
    (0, 4, 3): <obj1img>,
    (3, 4, 2): <obj2img>
  },
  "inventory": {
    "slot1": <item1>,
    "slot2": <item2>
  }
  "global": {
    0: np.array([7, 5])
  }
  "dimensions": {
    "objects": 3072,
    "inventory": 3,
    "global": 2
  }
}
next_state = {
  "objects": {
    (0, 4, 3): <obj1img>,
    (3, 4, 2): <obj2img>,
    (7, 4, 5): <newobjimg>
  },
  "inventory": {
    "slot1": <item1>,
  }
  "global": {
    0: np.array([7, 5])
  }
  "dimensions": {
    "objects": 3072,
    "inventory": 3,
    "global": 2
  }
}
```
For these key mismatches (e.g., `(7, 4, 5)` is not found in `state`, and `"slot2"` is not found in `next_state`), we assume that there has been an object addition/removal, and add skolem vectors for the missing keys:
```python
# skolem for (7, 4, 5) in state
state["objects"][(7, 4, 5)] = np.zeros(state["dimensions"]["objects"])
# skolem for "slot2" in next_state
next_state["inventory"]["slot2"] = np.zeros(state["dimensions"]["inventory"])
```
These are automatically added when there is a key mismatch in the tuple---you don't need to manually add these vectors (see `s2s.helpers` for details).

## Running the code
Once the dataset class is ready, add its definition to `agent._get_loader` and use `python main.py <yaml_file>` to train and run MSA&S2S pipeline. All hyperparameters of MSA and S2S are collected in a single yaml file.

### Example YAML configuration
```yaml
env: sokoban
save_path: save/sokoban/model1
fast_downward_path: "~/downward/fast-downward.py"

abstraction:
  method: "msa"                             # the abstraction method. only `msa` atm.
  parameters:
    input_dims:
      - [agent, 3072]                       # input dimensionality of the `agent` modality.
      - [inventory, 3]
      - [objects, 3073]
    action_classification_type: "sigmoid"   # or `softmax` if vectors are one-hot.
    n_hidden: 256                           # number of hidden units
    n_latent: 16                            # dimensionality of encoder outputs
    n_layers: 4                             # number of MLP layers
    action_dim: 26                          # dimensionality of action vectors
  training:
    batch_size: 128
    epoch: 1000
    lr: 0.0001
    device: "cuda"
    save_freq: 1

s2s:
  partition:
    eps: 0.5                                # DBSCAN eps for effect clustering
    mask_eps: 2                             # DBSCAN eps for mask clustering
    mask_threshold: 0.05                    # threshold to decide if a mask is changed by an option
    min_samples: 10                         # DBSCAN min_samples
  factor_threshold: 0.05                    # threshold to decide if a factor is changed by an option
  density_type: "knn"                       # `knn` or `kde`.
  comparison: "l2"                          # `l2`, `knn`, or `orig`.
  independency_test: "gaussian"             # `gaussian`, `knn`, or `independent`
  k_cross: 20                               # number of DT fits for the precondition
  pre_threshold: 0.7                        # threshold to select a precondition from DT fits
  min_samples_split: 0.05                   # DT split hyperparameter. Can be int (number) or float (ratio)
  pos_threshold: 0.9                        # threshold to decide if a leaf is positive
  negative_rate: 10                         # ratio of negative samples while learning preconds
```


### Skills to Symbols core
`factorise.py`: Contains a single method, `factors_from_partitions`, where we learn the factors of the environment directly from the option-mask changes. If the raw sensory input high-dimensional (such as pixels), this method might produce a lot of factors, which in turn increases the number of learned symbols. Ideally, a separate procedure should produce a factored representation, and this method should only find what those factors are.

`partition.py`: The skills to symbols framework requires options to satisfy the strong subgoal option property: $\text{Im}(X, o) = \text{Eff}(o)$, that is, effect of an option does not depend on its initiation vector (see Def. 11 in [[1](#1)]). This can be practically achieved by partitioning the dataset based on the effect set. The current implementation first partitions option masks, then masked effects for each unique mask based on DBSCAN algorithm [[3](#3)].

`structs.py`: Contains useful data abstractions. The most important one is `UniquePredicateList` which is a list of `Proposition`s. When we add a new proposition, it checks whether the proposition is already in the list or not by comparing the similarity with other propositions (see `vocabulary._overlapping_dists`). If the proposition is over multiple factors (which means those factors are dependent), it automatically adds all possible projections to the list (see the case 2 in Section 3.2.3 in [[1](#1)]).

`vocabulary.py`: The part we build the propositions (symbols) and the high-level MDP definition. The two most important methods are:
  - `build_vocabulary`: Given a list of partitions each approximately satisfying the strong subgoal property and factors of the environment (a factor is an index set over low-level variables), this method  
    1. Finds and assigns factors that are modified by each partition,
    2. Tests dependencies between factors[*](#*) for each effect set,
    3. Learns propositions over independent and dependent factors (together with possibly projections for dependents),
    4. Adds propositions to the vocabulary together with the mutual exclusiveness information (i.e., symbols over a factor should be mutually exclusive),
    5. Learns propositions over the factor that is not modified by any partition but might be needed for preconditions,
    6. Finds the set of propositions for each precondition by fitting a decision tree on positive and negative samples. The input to the decision tree are the most active symbols for each factor. The decision tree works for unordered set of inputs and fixed-size vectors.
  - `build_schemata`: Given the precondition and effect propositions for a partition, this method builds the action schema that covers which symbols should be active before the execution, and which symbols should be turned-off and on after execution (see Section 3.2.4 in [[1](#1)]).

*<a name="*"></a>Currently, there are three options for the factor independency: (1) a linear dependency test by checking the covariance, (2) k-nearest neighbor two-sample test which can possibly find non-linear relations [[4](#4)], and (3) assuming that they are independent to reduce the number of symbols (possibly losing soundness).


## References
[1]<a name="1"></a> Konidaris, G., Kaelbling, L. P., & Lozano-Perez, T. (2018). From skills to symbols: Learning symbolic representations for abstract high-level planning. Journal of Artificial Intelligence Research, 61, 215-289.  
[2]<a name="2"></a> Allen, C., Parikh, N., Gottesman, O., & Konidaris, G. (2021). Learning markov state abstractions for deep reinforcement learning. Advances in Neural Information Processing Systems, 34, 8229-8241.
[3]<a name="3"></a> https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html.  
[4]<a name="4"></a> Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv preprint arXiv:1610.06545.
