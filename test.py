import torch
import numpy as np
import argparse
import sys
from snake import SnakeEnv
from gymnasium.wrappers import FlattenObservation, FrameStack
from tqdm import tqdm


def test_agent(agent_path, env):
    lengths = []
    rewards = []
    episode_length = 1000
    for _ in tqdm(range(episode_length)):
        terminated, truncated = False, False
        ob, _ = env.reset()
        agent = torch.load(agent_path)
        length = 0
        reward = 0

        while not terminated and not truncated:
            # if 'google.cloud' not in sys.modules:  # env.render() will not work in colab
            #     env.render()
            qs = agent(torch.from_numpy(ob).float())
            a = qs.argmax().numpy()

            next_ob, r, terminated, truncated, _ = env.step(a)
            ob = next_ob
            length += 1
            reward += r

        # env.close()
        lengths.append(length)
        rewards.append(reward)
    env.close()
    print(f'average episode length: {np.mean(lengths)}')
    print(f'average reward incurred: {np.mean(rewards)}')


def get_args():
    parser = argparse.ArgumentParser(description='test-function')
    parser.add_argument('--env', default='Snake', help='name of environment')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    env = FlattenObservation(FrameStack(FlattenObservation(SnakeEnv(w=20, h=20, food_count=5, render_mode=None)), num_stack=3))
    env.metadata["render_fps"] = 32
    agent_path = f'./trained_agent_Snake_final.pt'
    test_agent(agent_path, env)
