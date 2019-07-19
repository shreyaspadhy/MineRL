import os
import gym
import minerl
import ray
from ray.rllib.agents import ppo
from gym.spaces import Discrete, Box


ray.init()


class MineRLEnv(gym.Env):
    # TODO: Implement
    def __init__(self, env_config):
        super().__init__()
        assert 'args' in env_config
        args = env_config['args']
        assert 'env_name' in args
        raise NotImplementedError()
        self.action_space = None
        self.observation_space = None

    def reset(self):
        return obs

    def step(self, action):
        return (obs, rew, done, info)


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


    trainer = ppo.PPOTrainer(
        env=WrappedEnv,
        config={
            "env_config": {
                'args': args,
                'test': False,
            },  # config to pass to env class
        })

    while True:
        print(trainer.train())
