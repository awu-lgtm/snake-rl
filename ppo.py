from gymnasium.wrappers import FlattenObservation, FrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from snake.snake import SnakeEnv
from tqdm import tqdm
from gymnasium import spaces
import torch
from torch import nn
from common.wrappers import ExtractStateWrapper

class CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 238):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0),
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

if __name__ == "__main__":
    # snake_env = lambda : FlattenObservation(SnakeEnv(w = 10, h = 9, food_count = 5, head_relative_action=True))
    policy_kwargs = dict(
        features_extractor_class=CNN,
        features_extractor_kwargs=dict(),
    )
    
    num_cpu = 32
    w,h = 9,10
    snake_env = lambda render_mode = None : ExtractStateWrapper(SnakeEnv(w=w, h=h, food_count=1, head_relative_action=True, head_relative_state=False, as_image=True, render_mode=render_mode, truncation_lim=(w*h)**2))
    env = make_vec_env(env_id=snake_env, n_envs=num_cpu)

    eval_callback = EvalCallback(eval_env=snake_env(), eval_freq=1_000_000//num_cpu, n_eval_episodes=10, best_model_save_path='./best_models/')
    checkpoint_callback = CheckpointCallback(save_freq=10_000_000//num_cpu, save_path='./model_checkpoints/', save_replay_buffer=True, save_vecnormalize=True)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tracking/", device="auto")
    print(model.policy.state_dict)
    model.learn(total_timesteps=1_000_000_000, callback=[eval_callback, checkpoint_callback])