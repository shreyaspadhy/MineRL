import gym
import minerl
import numpy as np
from env_wrappers import *


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct={}):
        assert False, "Just use a normal dict man"
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def make_minerl_env(env_name, args):
    """
    Create an environment with some standard wrappers.
    """
    env = gym.make(env_name)
    # NOTE: wrapping order matters!
    env = ResetTrimInfoWrapper(env)

    if args['frame_skip'] is not None:
        env = FrameSkip(env, skip=args['frame_skip'])
    if args['gray_scale']:
        env = GrayScaleWrapper(env, dict_space_key='pov')
    if env_name.startswith('MineRLNavigate'):
        env = PoVWithCompassAngleWrapper(env)
    else:
        env = ObtainPoVWrapper(env)

    # if args['frame_stack'] is not None and args['frame_stack'] > 0:
    #     env = FrameStack(env, args['frame_stack'], channel_order='hwc')

    # wrap env: action...
    env = CombineActionWrapper(env)
    env = SerialDiscreteCombineActionWrapper(env)

    return env
