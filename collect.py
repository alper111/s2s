from typing import Callable

import gymnasium as gym
import numpy as np

from structs import S2SDataset
from environment import S2SEnv, ObjectCentricEnv


def collect(n: int, env: S2SEnv | ObjectCentricEnv, options: dict[str, Callable] | None = None) -> S2SDataset:
    if isinstance(env.observation_space, gym.spaces.Box):
        shape = (n,) + env.observation_space.shape
    elif isinstance(env.observation_space, gym.spaces.Sequence):
        assert isinstance(env.observation_space.feature_space, gym.spaces.Box)
        shape = (n, env.max_objects) + env.observation_space.feature_space.shape
    state_arr = np.zeros(shape, dtype=np.float32)
    action_arr = np.zeros(n, dtype=np.int64)
    reward_arr = np.zeros(n, dtype=np.float32)
    next_state_arr = np.zeros(shape, dtype=np.float32)
    mask_arr = np.zeros(shape, dtype=bool)

    i = 0
    while i < n:
        state, _ = env.reset()
        done = False

        while not done and i < n:
            if options is None:
                action = env.action_space.sample()
            else:
                # TODO:
                # select a random option o with I_o > 0
                # execute it in a loop till it terminates
                pass
            next_state, reward, term, trun, _ = env.step(action)
            opt_mask = env.get_mask(state, next_state)

            state_arr[i] = state
            action_arr[i] = action
            reward_arr[i] = reward
            next_state_arr[i] = next_state
            mask_arr[i] = opt_mask

            done = term or trun
            state = next_state
            i += 1

    return S2SDataset(state_arr, action_arr, reward_arr, next_state_arr, mask_arr)
