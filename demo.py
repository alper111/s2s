import pickle

from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.transform_observation import TransformObservation

from environment import MNIST8Tile
from collect import collect
from partition import partition_to_subgoal
from learn_ops import learn_operators
from build_vocab import build_vocabulary, build_schemata


# initialize the environment
env = FlattenObservation(MNIST8Tile(random=True, max_steps=1))
env = TransformObservation(env, lambda x: x / 255)

# collect data
print("Collecting data...")
dataset = collect(50000, env, None)
print("Data collected.")

# partition the data s.t. subgoal property is satisfied
print("Partitioning data...")
partitions = partition_to_subgoal(dataset)
print(f"Number of partitions={len(partitions)}.")

# learn precondition classifiers and effect predictors
print("Learning operators...")
ops = learn_operators(partitions)
with open("ops.pkl", "wb") as f:
    pickle.dump(ops, f)
print(f"Number of operators={len(ops)}.")
# ops = pickle.load(open("ops.pkl", "rb"))

print("Building vocabulary...")
factors, vocabulary, propositions = build_vocabulary(ops, 84*84)
with open("factors.pkl", "wb") as f:
    pickle.dump(factors, f)
with open("vocabulary.pkl", "wb") as f:
    pickle.dump(vocabulary, f)
with open("propositions.pkl", "wb") as f:
    pickle.dump(propositions, f)
print(f"Number of factors={len(factors)}.")
print(f"Vocabulary size={len(vocabulary)}.")
print(f"Number of propositions={len(propositions)}.")
# factors = pickle.load(open("factors.pkl", "rb"))
# vocabulary = pickle.load(open("vocabulary.pkl", "rb"))
# propositions = pickle.load(open("propositions.pkl", "rb"))


print("Building schemata...")
schemata = build_schemata(factors, ops, vocabulary, propositions)
with open("schemata.pkl", "wb") as f:
    pickle.dump(schemata, f)
for schema in schemata:
    print(schema)
print(f"Number of action schemas={len(schemata)}.")
