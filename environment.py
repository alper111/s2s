import numpy as np
import torchvision
import gymnasium as gym
import pygame
from scipy.spatial.distance import cdist


class S2SEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.max_objects = 1

    def get_mask(self, state, next_state):
        raise NotImplementedError


class MNIST8Tile(S2SEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, permutation=None, random=False, render_mode=None, max_steps=200):
        dataset = torchvision.datasets.MNIST(root="data", train=True, download=True)

        self._data = dataset.data.numpy()
        _labels = dataset.targets.numpy()
        self._labels = {i: np.where(_labels == i)[0] for i in range(10)}

        self.index = None
        self.location = None
        self.t = None
        self.size = 3
        self.random = random
        self.permutation = permutation
        self.num_tile = self.size ** 2
        self.num_class = 10
        self.max_steps = max_steps

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(28*self.size, 28*self.size), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.reward_range = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None, permutation=None):
        super().reset(seed=seed)

        if permutation is None:
            perm = np.random.permutation(self.num_tile)
        else:
            perm = permutation

        self.index = np.zeros(self.num_tile, dtype=np.int64)
        for i in range(self.num_tile):
            digit = perm[i]
            labels = self._labels[digit]

            if self.random:
                self.index[i] = labels[np.random.randint(len(labels))]
            else:
                self.index[i] = labels[0]

            if digit == 0:
                self.location = [i // self.size, i % self.size]

        self.index = self.index.reshape(self.size, self.size)
        self.permutation = perm.reshape(self.size, self.size)
        self.t = 0

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action):
        assert self.index is not None, "You must call reset() before calling step()"
        assert self.location is not None, "You must call reset() before calling step()"
        assert self.permutation is not None, "You must call reset() before calling step()"
        assert self.t is not None, "You must call reset() before calling step()"

        row, col = self.location
        blank_idx = self.index[row, col]

        if action == 0:
            if self.location[1] != self.size-1:
                piece_idx = self.index[row, col+1]
                self.index[row, col] = piece_idx
                self.index[row, col+1] = blank_idx
                self.location[1] = col + 1

                self.permutation[row, col] = self.permutation[row, col+1]
                self.permutation[row, col+1] = 0
        elif action == 1:
            if self.location[0] != 0:
                piece_idx = self.index[row-1, col]
                self.index[row, col] = piece_idx
                self.index[row-1, col] = blank_idx
                self.location[0] = row - 1

                self.permutation[row, col] = self.permutation[row-1, col]
                self.permutation[row-1, col] = 0
        elif action == 2:
            if self.location[1] != 0:
                piece_idx = self.index[row, col-1]
                self.index[row, col] = piece_idx
                self.index[row, col-1] = blank_idx
                self.location[1] = col - 1

                self.permutation[row, col] = self.permutation[row, col-1]
                self.permutation[row, col-1] = 0
        elif action == 3:
            if self.location[0] != self.size-1:
                piece_idx = self.index[row+1, col]
                self.index[row, col] = piece_idx
                self.index[row+1, col] = blank_idx
                self.location[0] = row + 1

                self.permutation[row, col] = self.permutation[row+1, col]
                self.permutation[row+1, col] = 0

        self.t += 1

        obs = self._get_obs()
        info = self._get_info()
        reward = float(np.sum(self.permutation.reshape(-1) == np.arange(self.num_tile))
                       / self.num_tile)
        terminated = (reward > 1 - 1e-6)
        truncated = (self.t >= self.max_steps)

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def get_mask(self, state, next_state):
        return state != next_state

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        assert self.index is not None, "You must call reset() before calling _get_obs()"

        canvas = np.zeros((self.size*28, self.size*28))
        for i in range(self.size):
            for j in range(self.size):
                digit = self._data[self.index[i, j]]
                canvas[i*28:(i+1)*28, j*28:(j+1)*28] = digit.copy()
        return canvas

    def _get_info(self):
        return {"location": self.location, "permutation": self.permutation}

    def _render_frame(self):
        assert self.index is not None, "You must call reset() before calling _render_frame()"

        canvas = pygame.Surface((self.size*28, self.size*28))
        canvas.fill((0, 0, 0))

        for i in range(self.size):
            for j in range(self.size):
                digit = self._data[self.index[i, j]]
                digit = np.stack([digit]*3, axis=-1)
                digit = pygame.surfarray.make_surface(np.transpose(digit, (1, 0, 2)))
                canvas.blit(digit, (j*28, i*28))

        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.size*28, self.size*28))

            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))


class ObjectCentricEnv(gym.ObservationWrapper):
    # TODO: the segmentation module will be here.

    def __init__(self, env: S2SEnv):
        super().__init__(env)
        self.observation_space = gym.spaces.Sequence(
            gym.spaces.Tuple(
                (gym.spaces.Box(low=0, high=1, shape=(28*28,), dtype=np.float32),
                 gym.spaces.Box(low=0, high=2, shape=(2,), dtype=np.int64))
            ), stack=True
        )
        self.max_objects = 9

    def observation(self, obs):
        obj_feats = []
        obj_locs = []
        for i in range(3):
            for j in range(3):
                obj_feats.append(obs[i*28:(i+1)*28, j*28:(j+1)*28].flatten().astype(np.float32) / 255.0)
                obj_locs.append([i, j])
        obj_feats = np.stack(obj_feats)
        obj_locs = np.stack(obj_locs)
        return obj_feats, obj_locs

    def get_mask(self, state, next_state):
        # TODO
        # normally, we need to figure out which entity maps to which one
        # e.g., maybe with something like the Hungarian algorithm
        feat, loc = state[:, :784], state[:, 784:]
        feat_n, loc_n = next_state[:, :784], next_state[:, 784:]
        indices = np.argmin(cdist(feat, feat_n), axis=-1)
        next_loc = loc_n[indices]
        mask = np.zeros_like(state)
        mask[:, 784:] = loc != next_loc
        return mask
