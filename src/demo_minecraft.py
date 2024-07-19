from copy import deepcopy
import pickle

import yaml
import numpy as np

from environments.minecraft import Minecraft


if __name__ == "__main__":
    world_config = yaml.safe_load(open("data/Build_Wall_Easy.yaml", "r"))
    env = Minecraft(world_config)

    state = []
    action = []
    next_state = []

    for epi in range(20):
        obs = env.reset()
        done = False
        it = 0
        while not done:
            actions = env.available_actions()
            a = env.sample_action()
            old_obs = deepcopy(obs)

            obs, rew, done, info = env.step(a)
            if not info["action_success"]:
                continue

            state.append(old_obs)
            action.append(a)
            next_state.append(deepcopy(obs))

            it += 1
            if it == 200:
                break

        np.save("out/state.npy", np.stack(state))
        pickle.dump(action, open("out/action.pkl", "wb"))
        np.save("out/next_state.npy", np.stack(next_state))
    env.close()
