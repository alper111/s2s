import logging
import pickle
import argparse

import gym
from gym.wrappers import FlattenObservation
import yaml

from environments.minecraft import Minecraft
from environments.sokoban import MNISTSokoban
from environments.collect import collect
from s2s.partition import partition_to_subgoal
from s2s.factorise import factors_from_partitions
from s2s.vocabulary import build_vocabulary, build_schemata


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt="%H:%M:%S", force=True)
logger = logging.getLogger("main")


def main(env: gym.Env, n_samples: int, object_factored: bool):
    if env == "sokoban":
        env = MNISTSokoban(size=(4, 4), max_crates=1, max_steps=10, object_centric=object_factored,
                           rand_digits=False, rand_agent=False, rand_x=False, render_mode="rgb_array")
        if not object_factored:
            env = FlattenObservation(env)
    elif env == "minecraft":
        world_config = yaml.safe_load(open("data/Build_Wall_Easy.yaml", "r"))
        env = Minecraft(world_config)
    env.reset()

    # collect data
    logger.info("Collecting data...")
    dataset = collect(n_samples, env, None)
    logger.info("Data collected.")
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    # partition the data s.t. subgoal property is satisfied
    logger.info("Partitioning data...")
    partitions = partition_to_subgoal(dataset)
    logger.info(f"Number of partitions={len(partitions)}.")
    with open("partitions.pkl", "wb") as f:
        pickle.dump(partitions, f)

    # learn factors
    logger.info("Learning factors...")
    factors = factors_from_partitions(partitions, threshold=0.9)
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
    parser.add_argument("--env", type=str)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--object_factored", action="store_true")
    args = parser.parse_args()
    main(args.env, args.n_samples, args.object_factored)
