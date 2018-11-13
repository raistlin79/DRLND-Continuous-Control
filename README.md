# DRLND - Continuous-Control
## Second project in the Deep Reinforcement Learning Nanodegree. 

For this project, you will work with the Reacher environment.

![Reacher_Env.](reacher.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

## Installation
To set up your python environment to run the code in this repository, follow the instructions below.
Create (and activate) a new environment with Python 3.6.
 Linux or Mac:
```
    conda create --name drlnd python=3.6
    source activate drlnd
```
  Windows:
```
    conda create --name drlnd python=3.6 
    activate drlnd
```
Clone the following repository , and navigate to the python/ folder. Then, install several dependencies.
```
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
```
Create an IPython kernel for the drlnd environment.
```
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
More informnation can be found here:
https://github.com/udacity/deep-reinforcement-learning#dependencies.

The environment can be downloaded using the following links:

    Linux:            https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
    Mac OSX:          https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip
    Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip
    Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip


The environment is build using Unity ML-agent. More details about these environments and how to get started can be found here:
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md

To run the environment please use the following information:

    Mac: "path/to/Reacher.app"
    Windows (x86): "path/to/Reacher_Windows_x86/Reacher.exe"
    Windows (x86_64): "path/to/Reacher_Windows_x86_64/Reacher.exe"
    Linux (x86): "path/to/Reacher_Linux/Reacher.x86"
    Linux (x86_64): "path/to/Reacher_Linux/Reacher.x86_64"
    Linux (x86, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86"
    Linux (x86_64, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86_64"
    
 For a Windows x64 system putting Reacher folder right into you Jupyter notebook folder, this should be the resulting statement. <br>
 ```env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')```
