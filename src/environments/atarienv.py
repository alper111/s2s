"""
Original author: Cam Allen
Modified from https://github.com/camall3n/pix2sym/tree/dev
"""
from enum import IntEnum
import pickle

import gym
import numpy as np
import torchvision.transforms

convertToTensor = torchvision.transforms.ToTensor()
convertToPIL = torchvision.transforms.ToPILImage()


class actions(IntEnum):
    INVALID = -1
    NOOP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    UP_RIGHT = 6
    UP_LEFT = 7
    DOWN_RIGHT = 8
    DOWN_LEFT = 9
    UP_FIRE = 10
    RIGHT_FIRE = 11
    LEFT_FIRE = 12
    DOWN_FIRE = 13
    UP_RIGHT_FIRE = 14
    UP_LEFT_FIRE = 15
    DOWN_RIGHT_FIRE = 16
    DOWN_LEFT_FIRE = 17


class AtariEnv:
    """Wrapped Gym Atari environment
    """
    def __init__(self, name='', seed=None, render_mode="rgb_array"):
        envs = {
            'breakout': 'Breakout',
            'freeway': 'Freeway',
            'montezuma': 'MontezumaRevenge',
            'pacman': 'MsPacman',
            'pong': 'Pong',
        }
        assert name in envs.keys()
        self.env = gym.make('{}NoFrameskip-v4'.format(envs[name]), render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.n_actions = self.env.action_space.n
        self.reset(seed=seed)

    def __getattr__(self, name):
        # NOTE: Forwarding to underlying env's attributes as this is not a proper wrapper
        # not using self.env because self.* internally calls __getattr__, results in recursion loop.
        internal_env = self.__class__.__getattribute__(self, 'env')
        return internal_env.__getattribute__(name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, seed=None):
        self.env.reset()
        self._previous_frame = None
        if seed is not None:
            self.env.seed(seed)
        self.timestep = 0

    def close(self, *args, **kwargs):
        return self.env.close(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.timestep += 1
        ob, reward, done, info = self.env.step(*args, **kwargs)
        return ob, reward, done, info

    def render(self, mode='human', **kwargs):
        """Draw the current frame and display it on-screen
        """
        return self.env.render(mode=mode, **kwargs)

    def getGrayscaleFrame(self):
        img_gray = np.empty([210, 160, 1], dtype=np.uint8)
        self.env.env.ale.getScreenGrayscale(img_gray)
        return img_gray

    def getRGBFrame(self):
        img_rgb = np.empty([210, 160, 3], dtype=np.uint8)
        self.env.env.ale.getScreenRGB(img_rgb)
        return img_rgb

    def getFrame(self):
        """Get the current frame as an RGB array
        """
        return self.getRGBFrame()

    def getRAM(self):
        return self.env.env.ale.getRAM()

    def parseRAM(self, ram):
        raise NotImplementedError

    def getState(self):
        ram = self.getRAM()
        state = self.parseRAM(ram)
        return state

    def printRam(self):
        # prints out the current state of the ram
        ram = self.env.env.ale.getRAM()
        print(ram)

    def setRAM(self, ramIndex, value):
        # set the given index into ram to the given value.
        state = self.env.env.ale.cloneState()
        coded = self.env.env.ale.encodeState(state)
        # the ram section of the underlying state appears to be an array of
        # integers being used to represent an array of bytes. This means that
        # only every 4th element actually does anything: the rest are
        # meaningless 0's
        arrIndex = ramIndex * 4
        # the ram portion of the underlying state is offest 155 elements from
        # the beginning.
        arrIndex += 155
        coded[arrIndex] = value
        state2 = self.env.env.ale.decodeState(coded)
        self.env.env.ale.restoreState(state2)

    def save(self, filePath):
        state = self.env.env.ale.cloneState()
        coded = self.env.env.ale.encodeState(state)
        pickle.dump(coded, open(filePath, "wb"))

    def load(self, filePath):
        coded = pickle.load(open(filePath, "rb"))
        state = self.env.env.ale.decodeState(coded)
        self.env.env.ale.restoreState(state)


def bcd2int(bcd_string):
    # Convert a binary coded decimal string to an integer
    nibbles = [bcd_string[i:i+4] for i in range(0, len(bcd_string), 4)]
    digits = [format(int(nib, 2), '01d') for nib in nibbles]
    return int(''.join(digits), 10)


def _getIndex(address):
    assert isinstance(address, str) and len(address) == 2
    row, col = tuple(address)
    row = int(row, 16) - 8
    col = int(col, 16)
    return row*16+col


def getByte(ram, address):
    # Return the byte at the specified emulator RAM location
    idx = _getIndex(address)
    return ram[idx]


def getByteRange(ram, start, end):
    # Return the bytes in the emulator RAM from start through end-1
    idx1 = _getIndex(start)
    idx2 = _getIndex(end)
    return ram[idx1:idx2]


def setByte(env, address, value):
    # set the given address in the rame of the given environment
    idx = _getIndex(address)
    env.setRAM(idx, value)
