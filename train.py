import torch
from torch import optim
from torch.nn import functional as F
from network import QNetwork
from tqdm import tqdm
from buffer import ReplayBuffer
from itertools import count
import argparse
import numpy as np
import utils
from snake import SnakeEnv
from gymnasium.wrappers import FlattenObservation, FrameStack
import matplotlib.pyplot as plt

def experiment(args):
    # environment setup
    render_mode = None
    if args.render:
        render_mode = "human"
    env = FlattenObservation(SnakeEnv(w = 20, h = 20, food_count = 1, render_mode=render_mode))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n # here it is discrete, so we have n here as opposed to the dimension of the action
    
    # network setup
    network = QNetwork(args.gamma, state_dim, action_dim, args.hidden_sizes)
    
    # optimizer setup
    # if args.env == 'CartPole-v0':
    #     optimizer = optim.RMSprop(network.parameters(), lr=args.lr)
    # else:
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    
    # target setup (if wanted)
    if args.target:
        target_network = QNetwork(args.gamma, state_dim, action_dim, args.hidden_sizes)
        target_network.load_state_dict(network.state_dict())
        target_network.eval()
    
    # buffer setup
    buffer = ReplayBuffer(maxsize=args.max_size)
    
    # training
    scores = []
    max_scores = []
    mean_scores = []
    window = 100
    eps = args.eps
    for i in (bar := tqdm(range(args.num_episodes))):
        # initial observation, cast into a torch tensor
        ob, _ = env.reset(seed=0)
        ob = torch.tensor(ob).float()
        
        with torch.no_grad():
            eps = utils.get_eps(eps, 0.9999)
        for t in count():
            with torch.no_grad():
                # Collect the action from the policy.
                action = network.get_action(ob, eps)
            
            # Step the environment, convert everything to torch tensors
            n_ob, rew, terminated, truncated, info = env.step(action)
            
            action = torch.tensor(action)
            n_ob = torch.from_numpy(n_ob).float()
            rew = torch.tensor([rew])
            
            # Add new experience to replay buffer.
            buffer.add_experience(ob, action, rew, n_ob, terminated)
            
            ob = n_ob
            
            if len(buffer) >= args.batch_size:
                if t % args.learning_freq == 0:
                    # Sample batch from replay buffer and optimize model via gradient descent.
                    # HINTS:
                    #   If we're using a target Q network (see get_args() for details), make sure to get the targets with the target network.
                    #   Make sure to 'zero_grad' the optimizer before performing gradient descent.
                    
                    # SAMPLE FR OM BUFFER HERE
                    optimizer.zero_grad()
                    states, actions, rewards, next_states, dones = buffer.sample(args.batch_size)
                    
                    # COMPUTE Q VALUES HERE 
                    actions = actions.view(-1, 1)
                    qs = network.forward(states)
                    qs = qs.gather(1, actions)
                    qs = qs.view(-1)
                    # qs = network.get_max_q(states)

                    # COMPUTE TARGET Q VALUES FROM BATCH HERE
                    if args.target:
                        targets = target_network.get_targets(rewards, next_states, dones)
                    else:
                        targets = network.get_targets(rewards, next_states, dones)
                    
                    # GRADIENT DESCENT HERE
                    loss = torch.nn.functional.mse_loss(qs, targets)
                    loss.backward()
                    optimizer.step()
            
            if terminated or truncated:
                # we are done, so we break out of the for loop here
                # feel free to log anything you want here during training.
                score = info["score"]
                scores.append(scores)
                max_scores.append(max(score, max_scores[-1]))
                mean_scores.append(np.mean(mean_scores[-window:]))
                bar.set_description(f"score: {scores[-1]}, max: {max_scores[-1]}, mean: {mean_scores[-1]}")
                if args.plot:
                    utils.plot(scores, max_scores, mean_scores)
                break
        
        # Update target based on args.target_update_freq.
        # See 'utils.py' and argparse for more information.
        if args.target and i % args.target_update_freq == 0:
            utils.update_target(network, target_network, args.tau)
    
        # save agent
        if i % 1000 == 0:
            save_path = f"{args.save_path}_Snake.pt"
            torch.save(network, save_path)
        
        if i % 10000 == 0:
            save_path = f"{args.save_path}_Snake_{i}.pt"
            torch.save(network, save_path)
    save_path = f"{args.save_path}_Snake_final.pt"
    torch.save(network, save_path)

def get_args():
    parser = argparse.ArgumentParser(description='Q-Learning')
    
    # Environment args
    parser.add_argument('--gamma', type=float, default=0.999, help='discount factor')
    parser.add_argument('--eps', type=float, default=0.999, help='epsilon parameter')
    
    # Network args
    parser.add_argument('--hidden_sizes', default=[128, 128], nargs='+', type=int, help='hidden sizes of Q network')
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate for Q function optimizer')
    parser.add_argument('--target', action='store_true', help='if we want to use a target network')
    parser.add_argument('--target_update_freq', type=int, default=500, help='how often we update the target network')
    parser.add_argument('--tau', type=float, default=0.5, help='target update parameter')
    
    # Replay buffer args
    parser.add_argument('--max_size', type=int, default=100_000, help='max buffer size')
    
    # Training/saving args
    parser.add_argument('--num_episodes', type=int, default=100_000, help='number of episodes to run during training')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--save_path', default='./trained_agent', help='agent save path')
    parser.add_argument('--learning_freq', type=int, default=2, help='how often to update the network after collecting experience')

    # display
    parser.add_argument('render', action='store_true', help='renders game during training')
    parser.add_argument('plot', action='store_true', help='plots scores during training')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    args = get_args()
    if args.plot:
        plt.ion()
    experiment(args)
    