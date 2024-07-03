from copy import deepcopy

import yaml
import numpy as np

from environments.minecraft import Minecraft


if __name__ == "__main__":
    world_config = yaml.safe_load(open("data/Build_Wall_Easy.yaml", "r"))
    env = Minecraft(world_config)

    n_obj = len(world_config['blocks'])
    state = []
    action = []
    next_state = []

    for epi in range(100):
        env.reset()
        done = False
        obs = env.prev_obs
        it = 0
        while not done:
            actions = env.available_actions()
            a = actions[np.random.randint(0, len(actions))]
            state.append(deepcopy(obs))
            action.append(a)

            obs, rew, done, info = env.step(*a)
            next_state.append(deepcopy(obs))

            it += 1

    state = np.stack(state)
    action = np.array(action)
    next_state = np.stack(next_state)
    np.save("out/state.npy", state)
    np.save("out/action.npy", action)
    np.save("out/next_state.npy", next_state)
    env._env.close()
