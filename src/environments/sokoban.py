from typing import Optional

import gym.spaces
import numpy as np
import torchvision
import gym
import pygame
from scipy.spatial.distance import cdist
import os


class MNISTSokoban(gym.Env):
    metadata = {"render_modes": ["human", "array"], "render_fps": 5}

    def __init__(self, map_file: str = None, size: tuple[int, int] = None, object_centric: bool = False,
                 max_crates: int = 5, max_steps=200, render_mode: str = None, rand_digits: bool = False,
                 rand_agent: bool = False, rand_x: bool = False):
        assert map_file is not None or size is not None, "Either map_file or size must be provided"

        self._map_file = map_file
        self._size = size
        self._max_crates = max_crates
        self._max_steps = max_steps
        self._object_centric = object_centric
        self._max_objects = max_crates*2 + 1
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
        self._delta = np.array([[0, 1], [-1, 0], [0, -1], [1, 0], [0, 0]])
        self._t = 0

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)

        self._data = dataset.data.numpy()
        _labels = dataset.targets.numpy()
        self._labels = {i: np.where(_labels == i)[0] for i in range(10)}

        self._feat_dim = 32*32
        self._last_obs = None
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = None
        self.reset()

    @property
    def observation(self) -> np.ndarray:
        return self._get_obs()

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

    def reset(self) -> np.ndarray:
        self._init_agent_mark()
        self._init_x_mark()
        self._init_digits()

        if self._map_file is not None:
            self._map = self.read_map(self._map_file)
        else:
            self._map = self.generate_map(self._size, max_crates=self._max_crates)
        self._shape = (len(self._map), len(self._map[0]))

        if self.object_centric:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self._max_objects, 32*32+2,))
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(32*self._shape[0], 32*self._shape[1]))

        ax, ay = -1, -1
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                if self._map[i][j][1] == "@":
                    ax, ay = i, j
                    break
        self._agent_loc = np.array([ax, ay])
        self._t = 0

        obs = self.observation
        self._last_obs = obs

        if self.render_mode == "human":
            self._render_frame()

        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._map is not None, "You must call reset() before calling step()"

        pos = self._agent_loc
        next_pos = pos + self._delta[action]
        curr_bg, curr_tile = self._map[pos[0]][pos[1]]
        if next_pos[0] < 0 or next_pos[0] >= self._shape[0] or next_pos[1] < 0 or next_pos[1] >= self._shape[1]:
            self._t += 1
            obs = self.observation
            info = self.info
            reward = self.reward
            done = self.done
            return obs, reward, done, info

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

        obs = self.observation
        info = self.info
        reward = self.reward
        done = self.done
        self._last_obs = obs

        return obs, reward, done, info

    def sample_action(self) -> np.ndarray:
        return np.random.randint(0, 4)

    def render(self):
        if self.render_mode == "array":
            return self._render_frame()

    def close(self):
        if self._window is not None:
            pygame.quit()
            self._window = None
            self._clock = None

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
                    arr = np.transpose(pygame.surfarray.array3d(bg)[:, :, 0], (1, 0)) / 255.0
                    x = np.concatenate([arr.reshape(-1), [i], [j]])
                    entities.append(x)
                if fg is not None:
                    arr = np.transpose(pygame.surfarray.array3d(fg)[:, :, 0], (1, 0)) / 255.0
                    x = np.concatenate([arr.reshape(-1), [i], [j]])
                    entities.append(x)
            entities = np.stack(entities)
            return entities
        else:
            return np.transpose(pygame.surfarray.array3d(canvas)[:, :, 0], (1, 0)) / 255.0

    def _render_tiles(self) -> list[tuple[Optional[pygame.Surface], Optional[pygame.Surface], int, int]]:
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
        if self.object_centric:
            if self._last_obs is not None:
                obs_imgs = np.stack([x[:-2] for x in obs])
                last_obs_imgs = np.stack([x[:-2] for x in self._last_obs])
                indices = self._match_indices(last_obs_imgs, obs_imgs)
                obs = np.stack([obs[i] for i in indices])
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

    @staticmethod
    def get_delta_mask(state: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        return state != next_state
