import os
import argparse

import yaml

from environments.sokoban import Sokoban
from environments.npuzzle import NPuzzle
from environments.minecraft import Minecraft
from environments.collect import collect_raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect environment transitions")
    parser.add_argument("--env", type=str, help="The environment name")
    parser.add_argument("--n-samples", type=int, help="Number of samples")
    parser.add_argument("--max-steps", type=int, help="Maximum number of steps per episode")
    args = parser.parse_args()

    if args.env == "npuzzle":
        env = NPuzzle(random=True, max_steps=args.max_steps, object_centric=True)
    elif args.env == "sokoban":
        env = Sokoban(size=(4, 4), object_centric=True, max_crates=1, max_steps=args.max_steps)
    elif args.env == "minecraft":
        world_config = yaml.safe_load(open("data/Build_Wall_Easy.yaml", "r"))
        env = Minecraft(world_config, max_steps=args.max_steps)
    else:
        raise ValueError

    save_folder = os.path.join("data", args.env)
    collect_raw(args.n_samples, env, save_folder=save_folder)
