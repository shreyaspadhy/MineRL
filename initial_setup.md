# Instructions
Just a bunch of commands that you should run manually
Don't have time to automate this.

### 1: Install JDK-8 (MAC)
```bash
brew tap AdoptOpenJDK/openjdk
brew cask install adoptopenjdk8
export JAVA_HOME=`/usr/libexec/java_home -v 1.8
```

### 1: Install JDK-8 (Ubuntu)
```bash
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-8-jdk
export JAVA_HOME=`/usr/libexec/java_home -v 1.8`
```


### 2: Install mineRL
```bash
pip3 install --upgrade minerl
```

### 3: Installing RLLib
```bash
pip install tensorflow ==2.0.0-beta1  # or tensorflow-gpu
pip install ray  # This works by itself
pip install lz4  # RLLib PPO needs this
# pip install ray[rllib]
# pip install ray[debug]
```

### 3 (opt) : Compile RLLib from source
```bash
git clone https://github.com/ray-project/ray
cd ray/python/ray/rllib
```


### 4: Download dataset
Inside a Python terminal in MineRL repo
```python
DATA_DIR = 'data/'
minerl.data.download(directory=DATA_DIR)
```
