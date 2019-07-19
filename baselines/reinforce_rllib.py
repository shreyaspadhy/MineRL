import gym
import ray
from ray.rllib.agents import ppo

# Template of custom environment


class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = <gym.Space >
        self.observation_space = <gym.Space >

    def reset(self):
        return < obs >

    def step(self, action):
        return < obs > , < reward: float > , < done: bool > , < info: dict >


# Wrapping the MineRL Environment
class MineRLEnv(gym.Env):
    def __init__(self, env_config):
        """MineRL Environment for use

        env_config is a dict containing all parameters to pass thru trainer
        """
        # Is there a way to access the obs and action space without making the
        # env, needing MC to start up and take hella time
        self.environ = gym.make(env_config.name)

        self.action_space = self.environ.action_space
        self.observation_space = self.environ.observation_space

    def reset(self):
        return self.environ.reset()

    def step(self, action):
        obs, rew, done, info = self.environ.step(action)

        return obs, rew, done, info


ray.init()
trainer = ppo.PPOTrainer(env=MyEnv, config={
    "env_config": {},  # config to pass to env class
})

while True:
    print(trainer.train())
