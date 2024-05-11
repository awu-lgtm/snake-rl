import torch
import torch.nn as nn
from collections import namedtuple
from enum import IntEnum
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from IPython import display

Point = namedtuple('Point', ['x', 'y'])
class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @classmethod
    def is_opposite(self, d1, d2):
        return (d1 == Direction.RIGHT and d2 == Direction.LEFT) \
                or (d1 == Direction.LEFT and d2 == Direction.RIGHT) \
                or (d1 == Direction.UP and d2 == Direction.DOWN) \
                or (d1 == Direction.DOWN and d2 == Direction.UP)
    @classmethod
    def turn_right(self, d):
        return (d + 1)%4

    @classmethod
    def turn_left(self, d):
        return (d + 3)%4 

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)

def make_network(state_dim, action_dim, hidden_sizes):
    '''Initializes Q network.'''
    layers = []
    layers.append(nn.Linear(state_dim, hidden_sizes[0]))
    for i in range(1, len(hidden_sizes)):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
    
    layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_sizes[-1], action_dim))
    
    network = nn.Sequential(*layers).apply(initialize_weights)
    return network

@torch.no_grad()
def update_target(net, target_net, tau):
    # Update the target parameters using a soft update given by the parameter tau.
    # We want the following update to happen:
    #    θ_target = τ * θ_current + (1 - τ) * θ_target
    for n,t in zip(net.parameters(), target_net.parameters()):
        t.data = (1 - tau) * t.data + tau * n.data

def get_eps(eps_param, t):
    eps = eps_param * t
    if eps <= 0.001:
        return 0.001
    return eps

def l1_norm(p1: Point, p2: Point):
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    return abs(x1 - x2) + abs(y1 - y2)

def dirs_to_point(start: Point, end: Point):
    x1, y1 = start.x, start.y
    x2, y2 = end.x, end.y
    
    dirs = np.zeros(4, dtype=int)
    if y2 < y1:
        dirs[Direction.UP] = 1
    if y2 > y1:
        dirs[Direction.DOWN] = 1
    if x2 > x1:
        dirs[Direction.RIGHT] = 1
    if x2 < x1:
        dirs[Direction.LEFT] = 1
    return dirs

def move_in_dir(p:Point, dir: Direction):
    x,y = p.x, p.y
    match dir:
        case Direction.RIGHT:
            x += 1
        case Direction.LEFT:
            x -= 1
        case Direction.UP:
            y -= 1
        case Direction.DOWN:
            y += 1
    return Point(x, y)

def map_dirs(p: Point, fun: Callable[[Point], int], dir=None):
    if dir is not None:
        dirs = np.empty(3, dtype=np.int8)
        dirs[0] = fun(move_in_dir(p, dir))
        dirs[1] = fun(move_in_dir(p, Direction.turn_left(dir)))
        dirs[2] = fun(move_in_dir(p, Direction.turn_right(dir)))
    else:
        x,y = p.x, p.y
        dirs = np.empty(4, dtype=np.int8)
        dirs[Direction.UP] = fun(Point(x, y - 1))
        dirs[Direction.RIGHT] = fun(Point(x + 1, y))
        dirs[Direction.DOWN] = fun(Point(x, y + 1))
        dirs[Direction.LEFT] = fun(Point(x - 1, y))
    return dirs

def plot(scores, max_scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(max_scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(max_scores)-1, max_scores[-1], str(max_scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)