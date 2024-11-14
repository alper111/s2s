import os
import argparse

import yaml

from environments.sokoban import Sokoban
from environments.npuzzle import NPuzzle
from environments.minecraft import Minecraft
from environments.monty import MontezumaEnv
from environments.monty_skills import Plans, SkillController
from environments.collect import collect_raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect environment transitions")
    parser.add_argument("--env", type=str, help="The environment name")
    parser.add_argument("--n-samples", type=int, help="Number of samples")
    parser.add_argument("--max-steps", type=int, help="Maximum number of steps per episode")
    parser.add_argument("--yaml", type=str, help="Path to the Minecraft environment yaml file")
    parser.add_argument("--extension", default="", type=str, help="Extension to append to the save files")
    args = parser.parse_args()

    if args.env == "npuzzle":
        env = NPuzzle(random=True, max_steps=args.max_steps, object_centric=True)
        options = None
    elif args.env == "sokoban":
        env = Sokoban(map_file="map1.txt", object_centric=True, max_steps=args.max_steps)
        options = None
    elif args.env == "minecraft":
        assert args.yaml is not None, "Please provide the yaml file for the Minecraft environment"
        world_config = yaml.safe_load(open(args.yaml, "r"))
        env = Minecraft(world_config, max_steps=args.max_steps)
        options = None
    elif args.env == "monty":
        env = MontezumaEnv(seed=0, single_life=False, single_screen=False, render_mode="rgb_array")
        options = SkillController(initial_plan=Plans.GetFullRunPlan(), eps=0.0)
    else:
        raise ValueError

    save_folder = os.path.join("data", f"{args.env}{args.extension}")
    collect_raw(args.n_samples, env, options=options, save_folder=save_folder)
