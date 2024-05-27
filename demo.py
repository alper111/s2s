import pickle

from environment import MNIST8Tile, ObjectCentricEnv
from collect import collect
from partition import partition_to_subgoal
from learn_ops import learn_operators
from build_vocab import build_vocabulary, build_schemata

if __name__ == "__main__":
    # initialize the environment
    env = ObjectCentricEnv(MNIST8Tile(random=False, max_steps=1))

    # collect data
    print("Collecting data...")
    dataset = collect(10000, env, None)
    print("Data collected.")

    # partition the data s.t. subgoal property is satisfied
    print("Partitioning data...")
    partitions = partition_to_subgoal(dataset)
    print(f"Number of partitions={len(partitions)}.")
    with open("partitions.pkl", "wb") as f:
        pickle.dump(partitions, f)

    # learn precondition classifiers and effect predictors
    print("Learning operators...")
    ops = learn_operators(partitions)
    with open("ops.pkl", "wb") as f:
        pickle.dump(ops, f)
    print(f"Number of operators={len(ops)}.")
    # ops = pickle.load(open("ops.pkl", "rb"))

    print("Building vocabulary...")
    vocabulary, propositions = build_vocabulary(ops, 786)
    with open("vocabulary.pkl", "wb") as f:
        pickle.dump(vocabulary, f)
    with open("propositions.pkl", "wb") as f:
        pickle.dump(propositions, f)
    print(f"Vocabulary size={len(vocabulary)}.")
    print(f"Number of propositions={len(propositions)}.")
    # factors = pickle.load(open("factors.pkl", "rb"))
    # vocabulary = pickle.load(open("vocabulary.pkl", "rb"))
    # propositions = pickle.load(open("propositions.pkl", "rb"))

    print("Building schemata...")
    schemata = build_schemata(ops, vocabulary, propositions)
    with open("schemata.pkl", "wb") as f:
        pickle.dump(schemata, f)
    for schema in schemata:
        print(schema)
    print(f"Number of action schemas={len(schemata)}.")
