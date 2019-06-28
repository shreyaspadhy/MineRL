import gym
import minerl
import logging

logging.basicConfig(level=logging.DEBUG) # First time setup will take forever, DEBUG lets you keep track of wtf is happening

env = gym.make('MineRLNavigateDense-v0')

obs, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
