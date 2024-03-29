# Project 3: Collaboration and Competition

_Udacity Deep Reinforcement Learning Nanodegree_

![Smart agents](img/smart_agents.gif)

### Project details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
* The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Prerequisites

* [Miniconda](https://conda.io/miniconda.html)

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

    conda create --name drlnd python=3.6
    source activate drlnd
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .

### How to Run

Clone this repo, then run `jupyter notebook` to access the notebook.

### Notebook

[Link](Tennis.ipynb)

### Report

[Link](Report.md)
