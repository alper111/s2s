from typing import Callable

import gymnasium as gym
import numpy as np

from structs import S2SDataset
from environment import ObjectCentricEnv


def collect(n: int, env: gym.Env | ObjectCentricEnv, options: dict[str, Callable] | None = None) -> S2SDataset:
    if isinstance(env.observation_space, gym.spaces.Box):
        obs_shape = (n,) + env.observation_space.shape
        act_shape = (n,)
    elif isinstance(env.observation_space, gym.spaces.Sequence):
        assert isinstance(env.observation_space.feature_space, gym.spaces.Box)
        assert isinstance(env, ObjectCentricEnv)
        obs_shape = (n, env.max_objects) + env.observation_space.feature_space.shape
        act_shape = (n, 2)

    state_arr = np.zeros(obs_shape, dtype=np.float32)
    action_arr = np.zeros(act_shape, dtype=int)
    reward_arr = np.zeros(n, dtype=np.float32)
    next_state_arr = np.zeros(obs_shape, dtype=np.float32)
    mask_arr = np.zeros(obs_shape, dtype=bool)

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
            if isinstance(env, ObjectCentricEnv):
                next_state, reward, term, trun, info = env.step(state, action)
                action = np.array([action, info["acted_object"]])
            else:
                next_state, reward, term, trun, _ = env.step(action)
            opt_mask = state != next_state

            state_arr[i] = state
            action_arr[i] = action
            reward_arr[i] = reward
            next_state_arr[i] = next_state
            mask_arr[i] = opt_mask

            done = term or trun
            state = next_state
            i += 1

    return S2SDataset(state_arr, action_arr, reward_arr, next_state_arr, mask_arr)
