
import random

from copy import deepcopy

import numpy as np
from tqdm import tqdm

from rich.console import Console
from rich.markdown import Markdown
from rich.columns import Columns
from rich.panel import Panel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, deque, Counter

console = Console()

console = Console()


def get_state(step,state,action,reward):
    """Extract text from user dict."""
    return f"[b]{step}[/b]\n[yellow]{state}\n[blue]{action}\n[green]{reward}"

class AgentGPT:
    pass

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

class TemporalDifference:
    def __init__(self, Env, alpha=0.001, gamma=0.9, epsilon=0.1, lambd=0.9, batch_size=32):
        self.Env = Env
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.name = f"TemporalDifference(α = {self.alpha}, ɣ = {self.gamma}, ε = {self.epsilon}, λ = {self.lambd})"
        
        self.state_dim = self.Env._get_state_dim()      
        self.action_dim = self.Env._get_action_dim() 

        console.print(f"action dim:{self.action_dim}",style="white on blue")   
        console.print(f"action dim:{self.state_dim}",style="white on blue")   

        self.Q_main = DQN(self.state_dim, self.action_dim)
        self.Q_target = deepcopy(self.Q_main)
        self.optimizer = optim.Adam(self.Q_main.parameters(), lr=self.alpha)

    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            # 随机生成 [0,x] [y,0]
            axis = random.choice([0,1])
            value = round(random.uniform(-2, 2), 2)
            vector = [0, 0]
            vector[axis] = value
            return  np.array(vector)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                # [0,5,0]
                vector = self.Q_main(state_tensor).numpy()[0]
                return vector



    def reset(self):
        self.Q_main = DQN(self.state_dim, self.action_dim)
        self.Q_target = deepcopy(self.Q_main)
        self.optimizer = optim.Adam(self.Q_main.parameters(), lr=self.alpha)

    def _soft_update_Qtarget(self, tau=0.01):
        with torch.no_grad():
            for target_param, param in zip(self.Q_target.parameters(), self.Q_main.parameters()):
                target_param += tau * (param - target_param)

    def _update_Qmain_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update_Qtarget()


    def reset_episode(self):
        state = self.Env.reset()
        done = False
        step = 0
        episode_return = 0
        trace_dict = {} 
    
        return state, done, step, episode_return, trace_dict
    
    # Train the agent
    def train(self, num_episodes, on_policy=True):
        console.print( Markdown("#train"),style="white on blue")
        
        memory = deque(maxlen=10_000)      
        step_limit = 10                 
        
        for episode in tqdm(range(num_episodes), desc="Episodes", position=0, leave=True):
            state, done, step, episode_return, trace_dict = self.reset_episode()
            action = self.epsilon_greedy_policy(state)
            console.print(f"action:{action}",style="white on blue")
            while not done and step < step_limit:
                reward, next_state, done = self.Env.transition(state, action)
                console.print(Panel(get_state(episode,state,action,reward)))
                next_action = self.epsilon_greedy_policy(next_state)

                console.print("next state and next action")
                console.print(Panel(get_state(episode,next_state,next_action,"next")))
                
                trace_key = (state[0],state[1],action[0],action[1])
                if trace_key not in trace_dict:
                    trace_dict[trace_key] = 0
                trace_dict[trace_key] += 1
                trace = list(trace_dict.values())

                memory.append((episode, state, action, reward, next_state, next_action, done, trace))
                # console.print(Panel(get_state(episode,state,action,reward)))

                trace_dict[trace_key] = (self.gamma**step) * (self.lambd**step)

                state, action = next_state, next_action
                episode_return += reward
                step += 1
                
                if step >= step_limit and not done:
                    print('Episode reset, agent stuck')
                    state, done, step, episode_return, trace_dict = self.reset_episode()
                    action = self.epsilon_greedy_policy(state)
                    memory = [tup for tup in memory if tup[0] != episode] # remove the bad episode from memory

                if len(memory) >= self.batch_size:
                    batch = random.choices(memory, k = self.batch_size)
                    self.replay(batch, on_policy)

    def replay(self, batch, on_policy):
        episodes, states, actions, rewards, next_states, next_actions, dones, traces = zip(*batch)
        states = torch.tensor(states).to(torch.float32)
        actions = torch.tensor(actions).to(torch.int64)
        rewards = torch.tensor(rewards).to(torch.float32)
        next_states = torch.tensor(next_states).to(torch.float32)
        next_actions = torch.tensor(next_actions).to(torch.int64)
        dones = torch.tensor(dones).to(torch.int16)

        if on_policy==True:
            next_q = self.Q_target(next_states)
            next_q = next_q.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
        else:
            next_q = self.Q_target(next_states).max(1)[0]

        targets = rewards + (self.gamma * next_q * (1 - dones))

        current_q = self.Q_main(states)                                     # q values across all possible actions
        current_q = current_q.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # Pick the q value for the corresponding action taken

        traces = [torch.tensor(trace).to(torch.float32) for trace in traces]
        current_q = torch.cat([torch.mul(trace, q) for trace, q in zip(traces, current_q)])
        targets = torch.cat([torch.mul(trace, target) for trace, target in zip(traces, targets)])


        loss = nn.MSELoss()(current_q, targets)

        self._update_Qmain_weights(loss)