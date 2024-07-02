# From Skills to Symbols
This repository contains the implementation of core methods in [From Skills to Symbols: Learning Symbolic Representations for Abstract High-Level Planning](https://jair.org/index.php/jair/article/view/11175) [[1](#1)] paper. Some of the parts are based on [Steve's implementation](https://github.com/sd-james/skills-to-symbols).

## Install requirements
```
conda create -n s2s python=3.9 setuptools=65.5.0
conda activate s2s
pip install -r requirements.txt
```

## Outline of the code
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

## References
[1]<a name="1"></a> Konidaris, G., Kaelbling, L. P., & Lozano-Perez, T. (2018). From skills to symbols: Learning symbolic representations for abstract high-level planning. Journal of Artificial Intelligence Research, 61, 215-289.  
[2]<a name="2"></a> Pelleg, D., & Moore, A. (2000, June). X-means: Extending K-means with E cient Estimation of the Number of Clusters. In ICML’00 (pp. 727-734).  
[3]<a name="3"></a> Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv preprint arXiv:1610.06545.