import copy
import os
import pdb
from collections import OrderedDict

import gym
import minerl
import numpy as np
import ray
from gym.spaces import Box, Discrete
from gym.wrappers import Monitor
from gym.wrappers.monitoring.stats_recorder import StatsRecorder
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from env_wrappers import (ResetTrimInfoWrapper, FrameSkip,
                          GrayScaleWrapper, PoVWithCompassAngleWrapper,
                          ObtainPoVWrapper, CombineActionWrapper,
                          erialDiscreteCombineActionWrapper)


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


env_name = 'MineRLNavigateDense-v0'

args = {}
args['frame_skip'] = 4
args['gray_scale'] = False
args['frame_stack'] = 4
args['disable_action_prior'] = True
args['logging_level'] = 20
args['monitor'] = True
args['seed'] = 42
args['outdir'] = './output'

register_env(env_name,
             lambda config: make_minerl_env(env_name, args))

os.makedirs(args['outdir'], exist_ok=True)

train_seed = args['seed']  # noqa: never used in this script
test_seed = 2 ** 31 - 1 - args['seed']

ray.init()
trainer = ppo.PPOTrainer(
    env=env_name,
    config={
        "env_config": {
            'args': args,
            'test': False,
        },  # config to pass to env class
        "model": {
            "dim": 64,
            "conv_filters": [[12, 64, 1]],
        },
        "num_workers": 0,
        "num_cpus_per_worker": 0,
    })

while True:
    print(trainer.train())
