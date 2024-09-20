from common.wrappers import ImitationWrapper
from gymnasium.wrappers import TransformObservation
from snake.snake_env import snake_all
import os
from pathlib import Path
from common.policies import ActionCNN
from dagger.dagger import train

file_path = Path(__file__).resolve().parent

if __name__ == '__main__':
    from head_relative.head_relative_policy import get_policy

    save_dir = os.path.join(file_path, 'dagger', 'models')

    env = lambda: ImitationWrapper(snake_all(), "state", "head-relative")
    policy = ActionCNN(env().observation_space["novice"], 3)
    expert = get_policy()
    num_epochs = 1e4
    num_rollout_steps = 200
    supervision_steps = 20

    train(env=env, policy=policy, expert=expert, save_dir=save_dir)
