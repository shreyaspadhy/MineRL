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

<<<<<<< HEAD
git clone https://github.com/ray-project/ray
cd ray/python/ray/rllib


# 4: Download dataset
Inside a Python terminal in MineRL repo
import minerl
DATA_DIR = 'data/'
minerl.data.download(directory=DATA_DIR)
=======
git clone https: // github.com / ray - project / ray
cd ray / python / ray / rllib
>>>>>>> b08cc0d1d68fc448eefe35e2298032adf175574d
