import numpy as np
import torchvision
import gymnasium as gym
import pygame
from scipy.spatial.distance import cdist


class MNISTSokoban(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, map_file: str = None, size: tuple[int, int] = None, object_centric: bool = False,
                 max_crates: int = 5, max_steps=200, render_mode: str = None, rand_digits: bool = False,
                 rand_agent: bool = False, rand_x: bool = False):
        assert map_file is not None or size is not None, "Either map_file or size must be provided"

        self._map_file = map_file
        self._size = size
        self._max_crates = max_crates
        self._max_steps = max_steps
        self.object_centric = object_centric
        self.max_objects = max_crates*2 + 1
        self.render_mode = render_mode
        self.rand_digits = rand_digits
        self.rand_agent = rand_agent
        self.rand_x = rand_x

        self._shape = None
        self._window = None
        self._clock = None
        self._map = None
        self._digit_idx = None
        self._agent_loc = None
        self._delta = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
        self._t = 0

        dataset = torchvision.datasets.MNIST(root="data", train=True, download=True)

        self._data = dataset.data.numpy()
        _labels = dataset.targets.numpy()
        self._labels = {i: np.where(_labels == i)[0] for i in range(10)}

        self.action_space = gym.spaces.Discrete(4)
        self._feat_dim = 32*32
        self._last_obs = None
        self.reset()

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._init_agent_mark()
        self._init_x_mark()
        self._init_digits()

        if self._map_file is not None:
            self._map = self.read_map(self._map_file)
        else:
            self._map = self.generate_map(self._size, max_crates=self._max_crates)
        self._shape = (len(self._map), len(self._map[0]))
        shape = (self._shape[0]*32, self._shape[1]*32)
        if self.object_centric:
            self.observation_space = gym.spaces.Sequence(
                gym.spaces.Box(low=0, high=max(self._shape)-1, shape=(32*32+2,), dtype=np.float32)
            )
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        ax, ay = -1, -1
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                if self._map[i][j][1] == "@":
                    ax, ay = i, j
                    break
        self._agent_loc = np.array([ax, ay])
        self._t = 0

        self._last_obs = None
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._map is not None, "You must call reset() before calling step()"

        pos = self._agent_loc
        next_pos = pos + self._delta[action]
        curr_bg, curr_tile = self._map[pos[0]][pos[1]]
        if next_pos[0] < 0 or next_pos[0] >= self._shape[0] or next_pos[1] < 0 or next_pos[1] >= self._shape[1]:
            self._t += 1
            obs = self._get_obs()
            info = self._get_info()
            reward = self._get_reward()
            terminated = reward > 1 - 1e-6
            truncated = (self._t >= self._max_steps)
            return obs, reward, terminated, truncated, info

        next_bg, next_tile = self._map[next_pos[0]][next_pos[1]]

        # the next tile is empty
        if next_tile == " ":
            self._map[pos[0]][pos[1]] = (curr_bg, " ")
            self._map[next_pos[0]][next_pos[1]] = (next_bg, "@")
            self._agent_loc = next_pos
        # the next tile is a wall
        elif next_tile == "#":
            pass
        # the next tile contains a crate
        else:
            # check whether the crate can be pushed
            further_pos = next_pos + self._delta[action]
            if further_pos[0] >= 0 and further_pos[0] < self._shape[0] and \
               further_pos[1] >= 0 and further_pos[1] < self._shape[1]:
                further_bg, further_tile = self._map[further_pos[0]][further_pos[1]]
                if further_tile == " ":
                    self._map[pos[0]][pos[1]] = (curr_bg, " ")
                    self._map[next_pos[0]][next_pos[1]] = (next_bg, "@")
                    self._map[further_pos[0]][further_pos[1]] = (further_bg, next_tile)
                    self._agent_loc = next_pos

        self._t += 1

        obs = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()
        terminated = reward > 1 - 1e-6
        truncated = (self._t >= self._max_steps)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _init_x_mark(self):
        self._x_corners = [
            np.random.randint(2, 9),
            np.random.randint(2, 9),
            np.random.randint(24, 31),
            np.random.randint(24, 31),
            np.random.randint(24, 31),
            np.random.randint(2, 9),
            np.random.randint(2, 9),
            np.random.randint(24, 31)
        ]

    def _init_agent_mark(self):
        # random points for drawing the cross
        self._a_corners = [
            np.random.randint(13, 20),
            np.random.randint(2, 9),
            np.random.randint(2, 9),
            np.random.randint(24, 31),
            np.random.randint(24, 31),
            np.random.randint(24, 31)
        ]

    def _init_digits(self):
        self._digit_idx = np.zeros(10, dtype=np.int64)
        for i in self._labels:
            self._digit_idx[i] = np.random.choice(self._labels[i])

    def _render_frame(self):
        objects = self._render_tiles()
        canvas = pygame.Surface((self._shape[1]*32, self._shape[0]*32))
        canvas.fill((0, 0, 0))
        for obj in objects:
            bg, fg, i, j = obj
            if bg is not None:
                canvas.blit(bg, (j*32, i*32))
            if fg is not None:
                canvas.blit(fg, (j*32, i*32))

        if self.render_mode == "human":
            if self._window is None:
                pygame.init()
                pygame.display.init()
                self._window = pygame.display.set_mode((self._shape[1]*32, self._shape[0]*32))

            if self._clock is None:
                self._clock = pygame.time.Clock()

            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])

        if self.object_centric:
            entities = []
            for obj in objects:
                bg, fg, i, j = obj
                if bg is not None:
                    arr = np.transpose(pygame.surfarray.array3d(bg)[:, :, 0], (1, 0))
                    entities.append((arr, i, j))
                if fg is not None:
                    arr = np.transpose(pygame.surfarray.array3d(fg)[:, :, 0], (1, 0))
                    entities.append((arr, i, j))
            return entities
        else:
            return np.transpose(pygame.surfarray.array3d(canvas)[:, :, 0], (1, 0))

    def _render_tiles(self) -> list[tuple[pygame.Surface | None, pygame.Surface | None, int, int]]:
        objects = []
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                bg_tile = None
                fg_tile = None

                bg, fg = self._map[i][j]
                if bg == "0":
                    bg_tile = self._draw_digit(0)

                if fg == "#":
                    color = (80, 80, 80)
                    fg_tile = pygame.Surface((32, 32))
                    fg_tile.fill(color)
                elif fg == "@":
                    fg_tile = self._draw_agent()
                elif fg != " ":
                    digit = int(self._map[i][j][1])
                    fg_tile = self._draw_digit(digit)
                    if bg == "0":
                        cross = self._draw_cross()
                        fg_tile.blit(cross, (0, 0))

                objects.append((bg_tile, fg_tile, i, j))
        return objects

    def _get_obs(self) -> np.ndarray:
        obs = self._render_frame()
        obs = np.stack([np.concatenate([o_i[0].reshape(-1), [o_i[1]], [o_i[2]]], axis=-1) for o_i in obs])
        if self.object_centric and self._last_obs is not None:
            obs_imgs = np.stack([x[:-2] for x in obs])
            last_obs_imgs = np.stack([x[:-2] for x in self._last_obs])
            indices = self._match_indices(last_obs_imgs, obs_imgs)
            obs = np.stack([obs[i] for i in indices])
        self._last_obs = obs
        return obs

    def _get_info(self) -> dict:
        return {"map": self._map}

    def _get_reward(self) -> float:
        n_crossed = 0
        n_total = 0
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                bg, fg = self._map[i][j]
                if bg == "0":
                    n_total += 1
                    if (fg != "#" and fg != " " and fg != "@"):
                        n_crossed += 1
        return n_crossed / n_total

    def _draw_digit(self, digit: int) -> np.ndarray:
        if self.rand_digits:
            digit_idx = np.random.choice(self._labels[digit])
        else:
            digit_idx = self._digit_idx[digit]
        digit = self._data[digit_idx]
        digit = np.stack([digit]*3, axis=-1)
        digit = pygame.surfarray.make_surface(np.transpose(digit, (1, 0, 2)))
        digit = pygame.transform.scale(digit, (32, 32))
        return digit

    def _draw_cross(self) -> pygame.Surface:
        if self.rand_x:
            self._init_x_mark()
        color = (255, 255, 255)
        width = 4
        return self._draw_lines([self._x_corners[:4], self._x_corners[4:]], color, width)

    def _draw_agent(self) -> pygame.Surface:
        if self.rand_agent:
            self._init_agent_mark()
        color = (255, 255, 255)
        width = 4
        return self._draw_lines([[self._a_corners[0], self._a_corners[1], self._a_corners[2], self._a_corners[3]],
                                 [self._a_corners[0], self._a_corners[1], self._a_corners[4], self._a_corners[5]],
                                 [self._a_corners[2], self._a_corners[3], self._a_corners[4], self._a_corners[5]]],
                                color, width)

    def _draw_lines(self, lines: list[list[int]], color: tuple[int, int, int], width: int) -> pygame.Surface:
        canvas = pygame.Surface((32, 32))
        for line in lines:
            pygame.draw.line(canvas, color, line[:2], line[2:], width)
            pygame.draw.circle(canvas, color, line[:2], width//2)
            pygame.draw.circle(canvas, color, line[2:], width//2)
        return canvas

    def _match_indices(self, state, next_state) -> np.ndarray:
        # TODO
        # normally, we need to figure out which entity maps to which one
        # e.g., maybe with something like the Hungarian algorithm
        feat, _ = state[:, :self._feat_dim], state[:, self._feat_dim:]
        feat_n, _ = next_state[:, :self._feat_dim], next_state[:, self._feat_dim:]
        dists = cdist(feat, feat_n)
        indices = np.argmin(dists, axis=-1)
        return indices

    @property
    def map(self) -> np.ndarray:
        return self._map

    @staticmethod
    def read_map(map_file: str) -> list[list[str]]:
        with open(map_file, "r") as f:
            lines = f.readlines()
        _map = []
        for line in lines:
            row = []
            for x in line.strip():
                if x == "0":
                    row.append(("0", " "))
                else:
                    row.append((" ", x))
            _map.append(row)
        return _map

    @staticmethod
    def generate_map(size: tuple[int, int] = (10, 10), max_crates: int = 5) -> list[list[str]]:
        ni, nj = size
        assert ni >= 3 and nj >= 3, "The size of the map must be at least 3x3"
        total_middle_tiles = (ni-2)*(nj-2)
        assert (2*max_crates+1) <= total_middle_tiles, \
            "The number of crates (together with their goals) must be less than the total non-edge empty tiles"

        _map = [[(" ", " ") for _ in range(nj)] for _ in range(ni)]

        n = np.random.randint(1, max_crates+1)
        digits = np.random.randint(1, 10, n)
        locations = np.random.permutation((ni-2)*(nj-2))[:(2*n+1)]
        for i, x_i in enumerate(digits):
            di, dj = locations[i] // (nj-2) + 1, locations[i] % (nj-2) + 1
            _map[di][dj] = (" ", str(x_i))
            di, dj = locations[i+n] // (nj-2) + 1, locations[i+n] % (nj-2) + 1
            _map[di][dj] = ("0", " ")
        ax, ay = locations[-1] // (nj-2) + 1, locations[-1] % (nj-2) + 1
        _map[ax][ay] = (" ", "@")
        return _map


class MNIST8Tile(gym.Env):
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

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Sequence(
            gym.spaces.Box(low=0, high=2, shape=(28*28+2,), dtype=np.float32)
        )
        self.max_objects = 9
        self.feat_dim = 784

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        state = self.observation(state)
        return state, info

    def step(self, state, action):
        next_state, reward, term, trun, info = self.env.step(action)
        info["acted_object"] = 3*info["location"][0] + info["location"][1]
        next_state = self.observation(next_state)
        indices = self._match_indices(state, next_state)
        next_state = next_state[indices]
        return next_state, reward, term, trun, info

    def observation(self, obs):
        obj_feats = []
        obj_locs = []
        for i in range(3):
            for j in range(3):
                obj_feats.append(obs[i*28:(i+1)*28, j*28:(j+1)*28].flatten().astype(np.float32) / 255.0)
                obj_locs.append([i, j])
        obj_feats = np.stack(obj_feats)
        obj_locs = np.stack(obj_locs)
        new_obs = np.concatenate([obj_feats, obj_locs], axis=-1)
        return new_obs

    def _match_indices(self, state, next_state):
        # TODO
        # normally, we need to figure out which entity maps to which one
        # e.g., maybe with something like the Hungarian algorithm
        feat, _ = state[:, :self.feat_dim], state[:, self.feat_dim:]
        feat_n, _ = next_state[:, :self.feat_dim], next_state[:, self.feat_dim:]
        indices = np.argmin(cdist(feat, feat_n), axis=-1)
        return indices
