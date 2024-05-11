from common import *
import torch
from torch import optim
from pathlib import Path
import numpy as np
from dataset import *
from torch import distributions as pyd
import torch.nn as nn
import os
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Callable
from snake.snake_env import snake_head_relative
from common.policies import CNN

class DAGGER:
    def __init__(self, env: gym.Env, policy, expert, obs_space, action_space, expert_obs_space, expert_action_space, lr=0.0003):
        self.policy = policy
        self.env = env

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.expert = expert
        self.expert_obs_dim = expert_obs_space.n
        self.expert_action_dim = expert_action_space.n
        
        self.loss = nn.CrossEntropyLoss()


    def expert_policy(self, obs):
        p = self.expert(obs)
        action = np.random.choice(self.expert_action_dim, p=p)
        return action

    def rollout(self, env: gym.Env, num_steps: int):
        states = []
        expert_actions = []
        obs, _ = env.reset()
        novice_obs = torch.from_numpy(obs.novice)
        expert_obs = torch.from_numpy(obs.expert)
        
        for _ in range(num_steps):
            logits = self.policy(novice_obs)

            action = self.sample_from_logits(logits)
            expert_action = self.expert_policy(expert_obs)
            
            states.append(novice_obs)
            expert_actions.append(torch.tensor(expert_action))
            
            obs, _, done, _ = env.step(action)
            if done:
                obs, _ = env.reset()
            novice_obs = torch.from_numpy(obs.novice)
            expert_obs = torch.from_numpy(obs.expert)
        return ExpertData(torch.stack(states, dim=0), torch.stack(expert_actions, dim=0))
    
    def get_logits(self, states):
        return self.policy(states)

    def sample_from_logits(self, logits):
        logits = logits.detach().numpy()
        a = np.random.choice(a=len(logits), p=utils.compute_softmax(logits, axis=-1).flatten())
        return a

    def learn(self, expert_states, expert_actions):
        logits = torch.nn.functional.softmax(self.policy(expert_states), dim=1)
        loss = self.loss(logits, expert_actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

def train(env: Callable[[], gym.Env], policy, expert, num_epochs: int, num_rollout_steps:int, supervision_steps):
    # expert dataset loading  
    expert_dataset = ExpertDataset(ExpertData(torch.tensor([]), torch.tensor([], dtype=int)))
    
    # Create env
    env = env()
    
    # policy initialization
    learner = DAGGER(env, )
    epoch_losses = []
    
    for _ in tqdm(range(num_epochs)):
        new_data = learner.rollout(env, num_rollout_steps)
        
        expert_dataset.add_data(new_data)

        dataloader = get_dataloader(expert_dataset, args)

        # Supervised learning step
        supervision_loss = []
        for _ in tqdm(range(supervision_steps)):
            loss = 0.0
            for batch in dataloader:
                states, actions = batch[0], batch[1]
                loss += learner.learn(states, actions)
            supervision_loss.append(loss)
        epoch_losses.append(torch.mean(supervision_loss))

    # plotting
    # epochs = np.arange(1, args.dagger_epochs + 1)
    
    # plot_losses(epochs, epoch_losses, args.env)

    # saving policy
    dagger_path = os.path.join(args.policy_save_dir, 'dagger')
    os.makedirs(dagger_path, exist_ok=True)
    
    policy_save_path = os.path.join(dagger_path, f'{args.env}.pt')

    learner.save(policy_save_path)


if __name__ == '__main__':
    env = snake_head_relative()
    policy = CNN(env.observation_space)
    expert = PPO.load(f'./expert_policies/{env.spec.id}_policy.pt', env=env)
    num_epochs = 1e4
    num_rollout_steps = 2048
    supervision_steps = 100

    train(env, policy, expert, num_epochs, num_epochs, num_rollout_steps, supervision_steps)
