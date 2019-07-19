'''
Just a bunch of commands that you should run manually
Don't have time to automate this.
'''

# 1: Install JDK-8
# 2: Install mineRL
pip3 install - -upgrade minerl

# 3: Installing RLLib
pip install tensorflow == 2.0.0 - beta1  # or tensorflow-gpu
pip install ray  # This works by itself
pip install lz4  # RLLib PPO needs this
# pip install ray[rllib]
# pip install ray[debug]

git clone https: // github.com / ray - project / ray
cd ray / python / ray / rllib
