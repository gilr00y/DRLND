# Project 1: Navigation

### Introduction

This project implements the first project for the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). 

It makes use of the [Unity ML Agents](https://github.com/Unity-Technologies/ml-agents) to set up, run, and learn to navigate the Banana environment.

The Banana environment is a 3D "box" in which bananas randomly appear.  These bananas are either yellow or blue, and the goal of the player (or agent) is to collect yellow bananas while avoiding blue bananas. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

Here is a gif of the agent on a training run:

![banana_train](/Users/gilr00y/Dropbox/Udacity/DRLND/banana_train.gif)



#### State Space

The environment state is described to our agent by a 37-dimensional numpy array.  An example starting state looks like:

```python
 0.         1.         0.         0.0748472  0.         1.
 0.         0.         0.25755    1.         0.         0.
 0.         0.74177343 0.         1.         0.         0.
 0.25854847 0.         0.         1.         0.         0.09355672
 0.         1.         0.         0.         0.31969345 0.
 0.
```

#### Action Space

The environment accepts 4 different actions, indicated as discrete integers in the range [0,3] representing direction of movement:

- `0` forward
- `1` backward
- `2` left.
- `3` right.

#### Solving the environment

The Banana environment is episodic. It is considered "solved" when the agent scores an average of >13 points over a single 100-episode block.

### Getting Started

These instructions assume a recent version of macOS, and was tested on High Sierra (v10.13.2).

1. Ensure Python 3.6 is installed. This can be done by running `python --version` from the command line (or occasionally `python3 --version`). If not installed, it can be retrieved [here](https://www.python.org/downloads/mac-osx/).
2. Ensure "Banana.app" (included in the repo) opens correctly.  Double-clicking in "Finder" should yield the visual of a blank environment.![Screenshot 2018-09-09 16.02.44](/Users/gilr00y/Dropbox/Udacity/DRLND/Screenshot 2018-09-09 16.02.44.png)
3. Install the python runtime dependencies listed in `requirements.txt` by running `pip install -r requirements.txt` from the top level of this repo.

### Running the DQN

This model can be run entirely from the command line. From the top level directory of this repo, execute:

 `python main.py`

If the program is running correctly, the Unity environment should be running, and the console will be displaying an output that looks similar to the following:

```Unity Academy name: Academy
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :

Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , ,
Number of agents: 1
Number of actions: 4
States look like: [1.         0.         0.         0.         0.84408134 0.
 0.         1.         0.         0.0748472  0.         1.
 0.         0.         0.25755    1.         0.         0.
 0.         0.74177343 0.         1.         0.         0.
 0.25854847 0.         0.         1.         0.         0.09355672
 0.         1.         0.         0.         0.31969345 0.
 0.        ]
States have length: 37
Episode 100	Average Score: 0.848
Episode 200	Average Score: 4.78
Episode 300	Average Score: 8.16
Episode 400	Average Score: 10.29
Episode 463	Average Score: 13.01
Environment solved in 363 episodes!	Average Score: 13.01
```

