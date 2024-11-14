from typing import Callable, Optional
from copy import deepcopy
import os
import pickle

import numpy as np
import gym

from s2s.structs import S2SDataset


def collect(n: int, env: gym.Env, options: Optional[dict[str, Callable]] = None) -> S2SDataset:
    """
    Collects n samples from the environment.

    Parameters
    ----------
    n: int
        Number of samples to collect.
    env: BaseEnv
        Environment to collect samples from.
    options: dict[str, Callable], default=None
        Options to execute in the environment.

    Returns
    -------
    S2SDataset
        Dataset of <state, action, reward, next_state, mask> tuples.
    """

    obs_shape = (n,) + env.observation_space.shape
    act_shape = (n,) + env.action_space.shape

    state_arr = np.zeros(obs_shape, dtype=np.float32)
    action_arr = np.zeros(act_shape, dtype=int)
    reward_arr = np.zeros(n, dtype=np.float32)
    next_state_arr = np.zeros(obs_shape, dtype=np.float32)
    mask_arr = np.zeros(obs_shape, dtype=bool)

    i = 0
    while i < n:
        state = env.reset()
        done = False

        while not done and i < n:
            if options is None:
                action = env.sample_action()
            else:
                # TODO:
                # select a random option o with I_o > 0
                # execute it in a loop till it terminates
                raise NotImplementedError

            next_state, reward, done, _ = env.step(action)
            opt_mask = env.get_delta_mask(state, next_state)

            state_arr[i] = state
            action_arr[i] = action
            reward_arr[i] = reward
            next_state_arr[i] = next_state
            mask_arr[i] = opt_mask

            state = next_state
            i += 1

    return S2SDataset(state_arr, action_arr, reward_arr, next_state_arr, mask_arr)


def collect_raw(n: int, env: gym.Env,
                options: Optional[Callable] = None,
                save_folder: str = "out") -> None:
    """
    Collects n samples from the environment and saves them to disk.

    Parameters
    ----------
    n: int
        Number of samples to collect.
    env: BaseEnv
        Environment to collect samples from.
    options: Callable, default=None
        A callable options object to execute in the environment.
    save_folder: str, default="out"
        Folder to save the dataset.
    """

    state = []
    priv_state = []
    action = []
    reward = []
    next_state = []
    priv_next_state = []

    i = 0
    while i < n:
        obs = env.reset()
        options.reset()
        info = env.info
        done = False
        old_obs = deepcopy(obs)
        old_info = deepcopy(info)

        while (not done) and (i < n):
            if options is None:
                a = env.sample_action()
                o = a
                o_n = 1
                op_done = True
            else:
                option_obs = env.option_obs
                a, o, o_n, op_done = options(option_obs)
                # print(option_obs['level'], option_obs['screen'], option_obs['score'])
                # print(o, o_n, op_done, options.getValidSkills(option_obs))

            obs, rew, done, info = env.step(a)
            if not info.pop("action_success"):
                continue

            if op_done:
                # print(f"Remaining options {len(options.initial_plan)}")
                state.append(old_obs)
                priv_state.append(old_info)
                action.append(o)
                reward.append(rew)
                next_state.append(deepcopy(obs))
                priv_next_state.append(deepcopy(info))
                old_obs = deepcopy(obs)
                old_info = deepcopy(info)
                i += 1

                if i % (n//100) == 0:
                    print(f"Collected {(100*(i/n)):.1f}%")

    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "state.npy"), np.stack(state))
    np.save(os.path.join(save_folder, "priv_state.npy"), np.stack(priv_state))
    pickle.dump(action, open(os.path.join(save_folder, "action.pkl"), "wb"))
    np.save(os.path.join(save_folder, "reward.npy"), np.array(reward))
    np.save(os.path.join(save_folder, "next_state.npy"), np.stack(next_state))
    np.save(os.path.join(save_folder, "priv_next_state.npy"), np.stack(priv_next_state))
    env.close()
