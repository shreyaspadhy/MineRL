import numpy as np
import gym
import minerl
import logging
import ray
import torch
import ray.rllib.agents.ppo as ppo

# Boilerplate
logging.basicConfig(level=logging.DEBUG) # First time setup will take forever, DEBUG lets you keep track of wtf is happening
ray.init()

if __name__ == '__main__':
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    trainer = ppo.PPOTrainer(config=config, env="MineRLNavigateDense-v0")
    import pdb; pdb.set_trace()
    for i in range(2):
       # Perform one iteration of training the policy with PPO
       result = trainer.train()
       print(pretty_print(result))

       if i % 100 == 0:
           checkpoint = trainer.save()
           print("checkpoint saved at", checkpoint)
