from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch
from torch import nn

class CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
class MLP(BaseFeaturesExtractor):

def make_action_network(base):
    class ActionNetwork(base):
        def __init__(self, observation_space: spaces.Box, action_dim, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            self.action_layer = nn.Sequential(nn.Linear(features_dim, action_dim))
        
        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.action_layer(super().forward(observations))
    return ActionNetwork

ActionCNN = make_action_network(CNN)
ActionMLP = make_action_network(MLP)