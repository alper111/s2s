import os

import gym
import numpy as np
import torch
import torchvision
import pygame
from scipy.spatial.distance import cdist

from s2s.structs import UnorderedDataset


class NPuzzle(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, permutation=None, random=False, render_mode=None,
                 stochastic=False, max_steps=200, object_centric: bool = False):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)

        self._data = dataset.data.numpy()
        _labels = dataset.targets.numpy()
        self._labels = {i: np.where(_labels == i)[0] for i in range(1, 9)}

        self._digit_idx = None
        self._directions = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

        self._t = None
        self._size = 3
        self._random = random
        self._stochastic = stochastic
        self._permutation = permutation
        self._object_centric = object_centric
        self._num_tile = self._size ** 2
        self._max_steps = max_steps

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self._feat_dim = 28*28
        self._last_obs = None
        self.observation_space = None
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(4), gym.spaces.Discrete(8)])
        self.reset()

    @property
    def observation(self) -> np.ndarray:
        obs = self._get_obs()
        if self.object_centric:
            obs_dict = {}
            objs = {i: obj for i, obj in enumerate(obs)}
            obs_dict["objects"] = objs
            obs_dict["dimensions"] = {"objects": 786}
            return obs_dict
        return obs

    @property
    def reward(self) -> float:
        return self._get_reward()

    @property
    def done(self) -> bool:
        return (self.reward > 1 - 1e-6) or (self._t >= self._max_steps)

    @property
    def info(self) -> dict:
        return self._get_info()

    @property
    def object_centric(self) -> bool:
        return self._object_centric

    @property
    def stochastic(self) -> bool:
        return self._stochastic

    def reset(self, permutation=None) -> np.ndarray:
        self._last_obs = None
        if permutation is None:
            perm = np.random.permutation(self._num_tile)
        else:
            perm = permutation

        self._init_digits()

        self._permutation = perm.reshape(self._size, self._size)
        self._t = 0

        if self.object_centric:
            self.observation_space = gym.spaces.Box(low=0, high=2, shape=(self._size**2, 28*28+2))
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=((self._size*28, self._size*28)))

        obs = self.observation
        self._last_obs = obs

        if self.render_mode == "human":
            self._render_frame()

        return obs

    def step(self, action):
        self._t += 1
        dir, index = action
        x, y = index // self._size, index % self._size
        dx, dy = self._directions[dir]

        # if hit the limits
        if x+dx < 0 or x+dx >= self._size or y+dy < 0 or y+dy >= self._size:
            info = self.info
            info["action_success"] = False
            return self.observation, self.reward, self.done, self.info

        # if the empty tile is not in the direction of the action
        if self._permutation[x+dx, y+dy] != 0:
            info = self.info
            info["action_success"] = False
            return self.observation, self.reward, self.done, self.info

        self._permutation[x+dx, y+dy] = self._permutation[x, y]
        self._permutation[x, y] = 0

        obs = self.observation
        info = self.info
        reward = self.reward
        done = self.done
        self._last_obs = obs
        info["action_success"] = True

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, done, info

    def sample_action(self) -> int:
        actions = []
        e_idx = self._permutation.reshape(-1).tolist().index(0)
        ex, ey = e_idx // self._size, e_idx % self._size
        if ex != 0:
            actions.append((3, (ex-1)*self._size+ey))
        if ex != self._size-1:
            actions.append((1, (ex+1)*self._size+ey))
        if ey != 0:
            actions.append((0, ex*self._size+ey-1))
        if ey != self._size-1:
            actions.append((2, ex*self._size+ey+1))

        return actions[np.random.randint(len(actions))]

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _init_digits(self):
        self._digit_idx = np.zeros(self._num_tile, dtype=np.int64)
        for i in range(1, self._num_tile):
            labels = self._labels[i]

            if self._random:
                self._digit_idx[i] = labels[np.random.randint(len(labels))]
            else:
                self._digit_idx[i] = labels[0]

    def _get_obs(self):
        obs = self._render_frame()
        if self.object_centric:
            if self._last_obs is not None:
                last_obs = np.stack(list(self._last_obs["objects"].values()))
                indices = self._match_indices(last_obs, obs)
                obs = np.stack([obs[i] for i in indices])
        return obs

    def _get_info(self):
        return {"permutation": self._permutation}

    def _get_reward(self):
        return float(np.sum(self._permutation.reshape(-1) == np.arange(self._num_tile))
                     / self._num_tile)

    def _render_frame(self):
        objects = self._render_tiles()
        canvas = pygame.Surface((self._size*28, self._size*28))
        canvas.fill((0, 0, 0))
        for obj in objects:
            tile, i, j = obj
            canvas.blit(tile, (j*28, i*28))

        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self._size*28, self._size*28))

            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        if self.object_centric:
            entities = []
            for obj in objects:
                tile, i, j = obj
                arr = np.transpose(pygame.surfarray.array3d(tile)[:, :, 0], (1, 0)) / 255.0
                x = np.concatenate([arr.reshape(-1), [i], [j]])
                entities.append(x)
            entities = np.stack(entities)
            return entities
        else:
            return np.transpose(pygame.surfarray.array3d(canvas)[:, :, 0], (1, 0)) / 255.0

    def _render_tiles(self):
        objects = []
        for i in range(self._size):
            for j in range(self._size):
                digit = self._permutation[i][j]
                if digit != 0:
                    tile = self._draw_digit(digit)
                    objects.append((tile, i, j))
        return objects

    def _draw_digit(self, digit: int) -> np.ndarray:
        if self.stochastic:
            self._init_digits()
        digit_idx = self._digit_idx[digit]
        digit = self._data[digit_idx]
        digit = np.stack([digit]*3, axis=-1)
        digit = pygame.surfarray.make_surface(np.transpose(digit, (1, 0, 2)))
        return digit

    def _match_indices(self, state, next_state):
        # TODO
        # normally, we need to figure out which entity maps to which one
        # e.g., maybe with something like the Hungarian algorithm
        feat, _ = state[:, :self._feat_dim], state[:, self._feat_dim:]
        feat_n, _ = next_state[:, :self._feat_dim], next_state[:, self._feat_dim:]
        indices = np.argmin(cdist(feat, feat_n), axis=-1)
        return indices

    @staticmethod
    def get_delta_mask(state: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        return state != next_state


class NPuzzleDataset(UnorderedDataset):
    @staticmethod
    def _actions_to_label(action, key_order):
        a = torch.zeros(len(key_order)+1, dtype=torch.long)
        a[0] = action
        return a
