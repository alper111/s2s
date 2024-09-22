# From Skills to Symbols
This repository contains the implementation of core methods in [From Skills to Symbols: Learning Symbolic Representations for Abstract High-Level Planning](https://jair.org/index.php/jair/article/view/11175) [[1](#1)] paper. Some of the parts are based on [Steve's implementation](https://github.com/sd-james/skills-to-symbols).

## Install requirements
```
conda create -n s2s python=3.9 setuptools=65.5.0 wheel==0.38.4
conda activate s2s
pip install -r requirements.txt
```

## Example YAML configuration
```yaml
env: sokoban
save_path: save/sokoban/model1

abstraction:
  method: "msa"
  parameters:
    input_dims:
      - [objects, 9216]
    action_classification_type: "softmax"
    n_hidden: 256
    n_latent: 16
    n_layers: 4
    action_dim: 4
  training:
    batch_size: 128
    epoch: 1000
    lr: 0.0001
    device: "mps"
    beta: 0.0
    save_freq: 1

s2s:
  partition:
    eps: 0.5
    mask_threshold: 0.05
  factor_threshold: 0.05
  density_type: "knn"
  comparison: "l2"
  independency_test: "gaussian"
  k_cross: 20
  pre_threshold: 0.20
  min_samples_split: 0.05
  pos_threshold: 0.6

s2s_global:
  partition:
    eps: 0.5
    mask_threshold: 0.05
  factor_threshold: 0.05
  density_type: "knn"
  comparison: "l2"
  independency_test: "gaussian"
  k_cross: 20
  pre_threshold: 0.20
  min_samples_split: 0.05
  pos_threshold: 0.6
```

## Outline of the code
```
|-- docs
|-- src
    |-- abstraction    # Markov state abstraction
    |-- environments
    |-- s2s            # Skills to symbols core
```

### Skills to Symbols core
`factorise.py`: Contains a single method, `factors_from_partitions`, where we learn the factors of the environment directly from the option-mask changes. If the raw sensory input high-dimensional (such as pixels), this method might produce a lot of factors, which in turn increases the number of learned symbols.

`partition.py`: The skills to symbols framework requires options to satisfy the strong subgoal option property: $\text{Im}(X, o) = \text{Eff}(o)$, that is, effect of an option does not depend on its initiation vector (see Def. 11 in [[1](#1)]). This can be practically achieved by partitioning the dataset based on the effect set. The current implementation partitions based on X-means algorithm [[2](#2)].

`structs.py`: Contains useful data abstractions. The most important one is `UniquePredicateList` which is a list of `Proposition`s. When we add a new proposition, it checks whether the proposition is already in the list or not by comparing the similarity with other propositions (see `vocabulary._overlapping_dists`). If the proposition is over multiple factors (which means those factors are dependent), it automatically adds all possible projections to the list (see the case 2 in Section 3.2.3 in [[1](#1)]).

`vocabulary.py`: The part we build the propositions (symbols) and the high-level MDP definition. The two most important methods are:
  - `build_vocabulary`: Given a list of partitions each approximately satisfying the strong subgoal property and factors of the environment (a factor is an index set over low-level variables), this method  
    1. Finds and assigns factors that are modified by each partition,
    2. Tests dependencies between factors[*](#*) for each effect set,
    3. Learns propositions over independent and dependent factors (together with possibly projections for dependents),
    4. Adds propositions to the vocabulary together with the mutual exclusiveness information (i.e., symbols over a factor should be mutually exclusive),
    5. Learns propositions over the factor that is not modified by any partition but might be needed for preconditions,
    6. Finds the correct set of propositions for each precondition set by greedily finding the best proposition for each factor. If factors are independent, the greedy method should also give the correct factors. In case of dependent factors, one needs to loop over all possible combinations of propositions for those factors. In the current implementation, preconditions are found by only considering independent factors.
  - `build_schemata`: Given the precondition and effect propositions for a partition, this method builds the action schema that covers which symbols should be active before the execution, and which symbols should be turned-off and on after execution (see Section 3.2.4 in [[1](#1)]).

*<a name="*"></a>Currently, there are three options for the factor independency: (1) a linear dependency test by checking the covariance, (2) k-nearest neighbor two-sample test which can possibly find non-linear relations [[3](#3)], and (3) assuming that they are independent to reduce the number of symbols (possibly losing soundness).

### Environments
Environments follow the `gym` interface with some specifications on the observation space. The main differences are that (1) a state observation is object-centric (2) with (possibly) multiple modalities. For instance, in the Minecraft example, there is an agent observation, object-centric observations, and an inventory observation.

A state shall be a dictionary of the following structure:
```
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
    "dimensions": {
        "modality1": 3072,
        "modality2": 128,
        ...
    }
}
```
An action shall be an `int` indicating which action has been executed (i.e., `gym.spaces.Discrete`).
A Markov state abstraction can be trained on top of these `(state, action, next_state)` tuples.

## References
[1]<a name="1"></a> Konidaris, G., Kaelbling, L. P., & Lozano-Perez, T. (2018). From skills to symbols: Learning symbolic representations for abstract high-level planning. Journal of Artificial Intelligence Research, 61, 215-289.  
[2]<a name="2"></a> Pelleg, D., & Moore, A. (2000, June). X-means: Extending K-means with Efficient Estimation of the Number of Clusters. In ICML’00 (pp. 727-734).  
[3]<a name="3"></a> Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv preprint arXiv:1610.06545.