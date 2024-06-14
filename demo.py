import logging
import pickle
import argparse

from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.transform_observation import TransformObservation

from environment import MNIST8Tile, ObjectCentricEnv
from collect import collect
from partition import partition_to_subgoal
from factorise import factors_from_partitions, add_factors_to_partitions
from build_vocab import build_vocabulary, build_schemata


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt="%H:%M:%S", force=True)
logger = logging.getLogger("main")


def main(n_samples: int, object_factored: bool):
    # initialize the environment
    if object_factored:
        env = ObjectCentricEnv(MNIST8Tile(random=True, max_steps=1))
    else:
        env = FlattenObservation(MNIST8Tile(random=False, max_steps=1))
        env = TransformObservation(env, lambda x: x / 255)

    # collect data
    logger.info("Collecting data...")
    dataset = collect(n_samples, env, None)
    logger.info("Data collected.")

    # partition the data s.t. subgoal property is satisfied
    logger.info("Partitioning data...")
    partitions = partition_to_subgoal(dataset)
    logger.info(f"Number of partitions={len(partitions)}.")
    with open("partitions.pkl", "wb") as f:
        pickle.dump(partitions, f)

    # learn factors
    logger.info("Learning factors...")
    factors = factors_from_partitions(partitions, threshold=0.9)
    add_factors_to_partitions(partitions, factors, threshold=0.9)

    with open("factors.pkl", "wb") as f:
        pickle.dump(factors, f)
    logger.info(f"Number of factors={len(factors)}.")

    logger.info("Building vocabulary...")
    vocabulary, pre_props, eff_props = build_vocabulary(partitions, factors)
    with open("vocabulary.pkl", "wb") as f:
        pickle.dump(vocabulary, f)
    with open("pre_props.pkl", "wb") as f:
        pickle.dump(pre_props, f)
    with open("eff_props.pkl", "wb") as f:
        pickle.dump(eff_props, f)
    logger.info(f"Vocabulary size={len(vocabulary)}.")
    logger.info(f"Number of precondition propositions={len(pre_props)}.")
    logger.info(f"Number of effect propositions={len(eff_props)}.")

    logger.info("Building schemata...")
    schemata = build_schemata(vocabulary, pre_props, eff_props)
    with open("schemata.pkl", "wb") as f:
        pickle.dump(schemata, f)
    for schema in schemata:
        logger.info(schema)
    logger.info(f"Number of action schemas={len(schemata)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--object_factored", action="store_true")
    args = parser.parse_args()
    main(args.n_samples, args.object_factored)
