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
from ray.rllib.agents import dqn
from ray.tune.registry import register_env
from utils import make_minerl_env


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


ray.init()
# TODO: --replay-start-size 5000 (replay_starts)
# minibatch_size 32
#  --frame-stack 4 --frame-skip 4
# --batch-accumulator mean
# --prioritized = prioritized_replay_alpha, prioritized_replay_beta
# TODO: clip_delta = use_huber???
# TODO: def soft_copy_param(target_link, source_link, tau)??
# target_update_method and tau

trainer = dqn.DQNTrainer(
    env=env_name,
    config={
        "noisy": True,
        "buffer_size": 11,
        "target_network_update_freq": 10000,
        "n_step": 10,
        "lr": 0.0000625,
        "adam_epsilon": 0.00015,
        "prioritized_replay_alpha": 0.6,
        "num_atoms": 51,
        "gamma": 0.99,
        "train_batch_size": 11,
        "sample_batch_size": 11,
        "evaluation_config": {
            "exploration_fraction": 0,
            "exploration_final_eps": 0,
        },
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
    import pdb
    pdb.set_trace()  # noqa
