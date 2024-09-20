import torch
from torch import nn
import matplotlib.pyplot as plt
import os
import os.path as osp

DISCRETE_ENVS = ['CartPole-v0', 'MountainCar-v0']

# Neural networks for use in BC + DAGGER
class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscretePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, states):
        '''Returns action distribution for all states s in our batch.
        
        :param states: torch.Tensor, size (B, state_dim)
        
        :return logits: action logits, size (B, action_dim)
        '''
        logits = self.net(states)
        return logits.float()

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GaussianPolicy, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2 * action_dim)
        )
        
    def forward(self, states):
        '''Returns mean and standard deviation of Gaussian action distribution for all states s in our batch.
        
        :param states: torch.Tensor, size (B, state_dim)
        
        :return mean & std, size (B, action_dim) each
        '''
        mean, std = self.net(states).chunk(2, 1)
        std = nn.functional.softplus(std) + 0.1
        
        return mean, std

# Plotting utils
def plot_losses(epochs, losses):
    plt.plot(epochs, losses)
    
    plt.title(f'DAGGER losses')
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    plot_dir = './plots'

    os.makedirs(plot_dir, exist_ok=True)
    
    plt.savefig(osp.join(plot_dir, f'dagger_losses.png'))
    