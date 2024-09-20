import torch
from torch import optim
from pathlib import Path
import numpy as np
from dagger.dataset import *
from torch import distributions as pyd
import torch.nn as nn
import os
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from typing import Callable
from snake.snake_env import snake_head_relative
from common.policies import ActionCNN
from common.wrappers import ImitationWrapper
from pathlib import Path
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

file_path = Path(__file__).resolve().parent

class DAGGER:
    def __init__(self, env: gym.Env, policy, expert, lr=0.0003):
        self.policy = policy
        self.env = env

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.expert: PPO = expert
        self.expert_action_dim = env.action_space.n
        
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.functional.softmax

    def expert_policy(self, obs):
        a, _ = self.expert.predict(obs, deterministic=True)
        return a
    
    def rollout(self, env: gym.Env, num_steps: int):
        states = []
        expert_actions = []
        obs, _ = env.reset()
        novice_obs = torch.from_numpy(obs["novice"]).float()
        expert_obs = torch.from_numpy(obs["expert"])
        
        for _ in range(num_steps):
            # print(novice_obs[None])
            logits = self.policy(novice_obs[None]).squeeze()

            action = self.sample_from_logits(logits)
            expert_action = self.expert_policy(expert_obs)
            
            states.append(novice_obs)
            expert_actions.append(torch.tensor(expert_action))
            
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            novice_obs = torch.from_numpy(obs["novice"]).float()
            expert_obs = torch.from_numpy(obs["expert"])
        return ExpertData(torch.stack(states, dim=0), torch.stack(expert_actions, dim=0))
    
    def get_logits(self, states):
        return self.policy(states)

    def sample_from_logits(self, logits):
        p = self.softmax(logits, dim=-1)
        a = p.multinomial(1)[0]
        # logits = logits.detach().numpy()
        # a = np.random.choice(a=len(logits), p=utils.compute_softmax(logits, axis=-1).flatten())
        return a.item()

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

def train(env: Callable[[], gym.Env], policy, expert, num_epochs: int = 1000, 
          num_rollout_steps: int = 8192, supervision_steps: int = 20, num_dataset_samples: int = 100_000, batch_size: int = 64, save_dir=file_path):
    # expert dataset loading
    expert_dataset = ExpertDataset(ExpertData(torch.tensor([]), torch.tensor([], dtype=int)))
    
    # Create env
    env = env()
    
    # policy initialization
    learner = DAGGER(env, policy, expert)
    epoch_losses = []
    
    for _ in (bar := tqdm(range(num_epochs))):
        new_data = learner.rollout(env, num_rollout_steps)
        expert_dataset.add_data(new_data)

        if len(expert_dataset) > 1_000_000:
            expert_dataset = expert_dataset[len(expert_dataset)-800_000:]
        dataloader = get_dataloader(expert_dataset, num_dataset_samples, batch_size)

        # Supervised learning step
        supervision_loss = []
        for _ in tqdm(range(supervision_steps)):
            loss = 0.0
            for batch in dataloader:
                states, actions = batch[0], batch[1]
                loss += learner.learn(states, actions)
            supervision_loss.append(loss.detach().numpy())
        epoch_losses.append(np.mean(supervision_loss))
        bar.set_description(desc=f"{epoch_losses[-1]}")

    # plotting
    epochs = np.arange(0, num_epochs)
    plot_losses(epochs, epoch_losses)

    # saving policy
    policy_save_path = os.path.join(save_dir, 'imitation.pt')
    learner.save(policy_save_path)

def plot_losses(epochs, losses):
    plt.plot(epochs, losses)
    
    plt.title(f'DAGGER losses')
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    plt.show()
    plot_dir = './plots'

    # os.makedirs(plot_dir, exist_ok=True)
    
    # plt.savefig(os.path.join(plot_dir, f'dagger_losses.png'))

if __name__ == '__main__':
    from head_relative.head_relative_policy import get_policy
    
    save_dir = os.path.join(file_path, 'models')

    env = ImitationWrapper(snake_head_relative())
    policy = ActionCNN(env.observation_space)
    expert = get_policy()
    num_epochs = 1e4
    num_rollout_steps = 2048
    supervision_steps = 100

    train(env, policy, expert, num_epochs, num_rollout_steps, supervision_steps, 
          save_dir)
