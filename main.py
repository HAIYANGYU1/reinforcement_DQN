import math

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from copy import deepcopy
import torch
import torch.nn as nn

import torch.nn.functional as F
from model import DQN,TemporalDifference
import torch.optim as optim

from shpere_env import SphereEnv
from rich.console import Console

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"

)

console = Console()
console.print("hello agent")

training_episodes = 10_000
env = SphereEnv([0,0],[2,5])

agent = TemporalDifference(env, alpha=0.001, gamma=0.99, epsilon=0.3, lambd=1)
agent.train(num_episodes=training_episodes)