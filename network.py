import torch
from torch import nn
from snake.utils import make_network
import numpy as np

class QNetwork(nn.Module):
    def __init__(self,
                gamma,
                state_dim,
                action_dim,
                hidden_sizes=[10, 10]):
        super().__init__()
        self.gamma = gamma
        # neural net architecture
        self.network = make_network(state_dim, action_dim, hidden_sizes)
    
    def forward(self, states):
        '''Returns the Q values for each action at each state.'''
        qs = self.network(states)
        return qs

    def get_max_q(self, states):
        qs = torch.max(self.forward(states), 1).values
        return qs
    
    def get_action(self, state, eps):
        qs = self.forward(state)
        explore = np.random.binomial(1, eps)
        if explore:
            return np.random.randint(qs.size(dim=0))
        return int(torch.argmax(qs))
    
    @torch.no_grad()
    def get_targets(self, rewards, next_states, dones):
        # Get the next Q function targets, as given by the Bellman optimality equation for Q functions.
        return rewards + ((self.gamma * self.get_max_q(next_states)) * (1 - dones))