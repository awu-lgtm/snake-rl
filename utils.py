import torch
import torch.nn as nn
from collections import namedtuple
from enum import IntEnum
import numpy as np
from typing import Callable

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
    eps = eps_param ** t
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
    if y2 > y1:
        dirs[Direction.UP] = 1
    if y2 < y1:
        dirs[Direction.DOWN] = 1
    if x2 > x1:
        dirs[Direction.RIGHT] = 1
    if x1 < x2:
        dirs[Direction.LEFT] = 1
    return dirs

def map_dirs(p: Point, fun: Callable[[Point], int]):
    x,y = p.x, p.y
    dirs = np.empty(4, dtype=int)
    dirs[Direction.UP] = fun(Point(x, y + 1))
    dirs[Direction.RIGHT] = fun(Point(x + 1, y))
    dirs[Direction.DOWN] = fun(Point(x, y - 1))
    dirs[Direction.LEFT] = fun(Point(x - 1, y))
    return dirs