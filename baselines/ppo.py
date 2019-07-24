import os
import gym
import ray
from ray.rllib.agents import ppo
from gym.spaces import Discrete, Box
from collections import OrderedDict
import pdb

import numpy as np
import copy
from gym.wrappers import Monitor
from gym.wrappers.monitoring.stats_recorder import StatsRecorder
import numpy as np
import minerl


class MineRLEnv(gym.Env):
    # TODO: Implement
    def __init__(self, env_config):
        super().__init__()
        assert 'args' in env_config
        args = env_config['args']
        assert 'env_name' in args

        import gym
        import minerl
        self.env = gym.make(args['env_name'])

        # Mimic POVWithCompass

        self._compass_angle_scale = 180

        pov_space = self.env.observation_space.spaces['pov']
        compass_angle_space = self.env.observation_space.spaces['compassAngle']

        # pdb.set_trace()
        low = self.observation({'pov': pov_space.low, 'compassAngle': compass_angle_space.low})
        high = self.observation({'pov': pov_space.high, 'compassAngle': compass_angle_space.high})

        self.observation_space = gym.spaces.Box(low=low, high=high)

        # Mimic CombineActionSpace
        self.wrapping_action_space = self.env.action_space

        def combine_exclusive_actions(keys):
            """
            Dict({'forward': Discrete(2), 'back': Discrete(2)})
            =>
            new_actions: [{'forward':0, 'back':0}, {'forward':1, 'back':0}, {'forward':0, 'back':1}]
            """
            new_key = '_'.join(keys)
            valid_action_keys = [k for k in keys if k in self.wrapping_action_space.spaces]
            noop = {a: 0 for a in valid_action_keys}
            new_actions = [noop]

            for key in valid_action_keys:
                space = self.wrapping_action_space.spaces[key]
                for i in range(1, space.n):
                    op = copy.deepcopy(noop)
                    op[key] = i
                    new_actions.append(op)
            return new_key, new_actions

        self._maps = {}
        for keys in (
                ('forward', 'back'), ('left', 'right'), ('sneak', 'sprint'),
                ('attack', 'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt')):
            new_key, new_actions = combine_exclusive_actions(keys)
            self._maps[new_key] = new_actions

        self.combined_action_space = gym.spaces.Dict({
            'forward_back':
                gym.spaces.Discrete(len(self._maps['forward_back'])),
            'left_right':
                gym.spaces.Discrete(len(self._maps['left_right'])),
            'jump':
                self.wrapping_action_space.spaces['jump'],
            'sneak_sprint':
                gym.spaces.Discrete(len(self._maps['sneak_sprint'])),
            'camera':
                self.wrapping_action_space.spaces['camera'],
            'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                gym.spaces.Discrete(len(self._maps['attack_place_equip_craft_nearbyCraft_nearbySmelt']))
        })

        # Serial DiscreteCombine
        self.wrapping_action_space = self.combined_action_space

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        # get each discrete action
        self._actions = [self.noop]
        for key in self.noop:
            if key == 'camera':
                # action candidate : {[0, -10], [0, 10]}
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, -10], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, 10], dtype=np.float32)
                self._actions.append(op)
            else:
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)

        n = len(self._actions)
        self.serial_action_space = gym.spaces.Discrete(n)

        self.action_space = self.serial_action_space

    @property
    def spec(self):
        return self.env.spec

    # Replicates ResetTrimInfoWrapper
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(self.action_combined(self.action_serial(action)))
        return self.observation(obs), rew, done, info

    def observation(self, observation):
        pov = observation['pov']
        compass_scaled = observation['compassAngle'] / self._compass_angle_scale
        compass_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=pov.dtype) * compass_scaled
        return np.concatenate([pov, compass_channel], axis=-1)

    def action_combined(self, action):
        if not self.combined_action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.combined_action_space))

        original_space_action = OrderedDict()
        for k, v in action.items():
            if k in self._maps:
                a = self._maps[k][v]
                original_space_action.update(a)
            else:
                original_space_action[k] = v

        return original_space_action

    def action_serial(self, action):
        if not self.serial_action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.serial_action_space))

        original_space_action = self._actions[action]

        return original_space_action


if __name__ == '__main__':
    args = {}
    args['logging_level'] = 20
    args['monitor'] = True
    args['seed'] = 42
    args['outdir'] = './output'
    args['env_name'] = 'MineRLNavigateDense-v0'

    os.makedirs(args['outdir'], exist_ok=True)

    train_seed = args['seed']  # noqa: never used in this script
    test_seed = 2 ** 31 - 1 - args['seed']

    ray.init()
    trainer = ppo.PPOTrainer(
        env=MineRLEnv,
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
