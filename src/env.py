import numpy as np
import gym


class NavDenseEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()
