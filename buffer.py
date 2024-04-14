import torch
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer(object):
    '''Replay buffer that stores online (s, a, r, s', d) transitions for training.'''
    def __init__(self, maxsize=100000):
        # Initialize the buffer using the given parameters.
        # HINT: Once the buffer is full, when adding new experience we should not care about very old data.
        self.q = deque(maxlen=maxsize)
    
    def __len__(self):
        return len(self.q)
        
    
    def add_experience(self, state, action, reward, next_state, done):
        # Add (s, a, r, s', d) to the buffer.
        # HINT: See the transition data type defined at the top of the file for use here.
        self.q.append(Transition(state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        # Sample 'batch_size' transitions from the buffer.
        # Return a tuple of torch tensors representing the states, actions, rewards, next states, and terminal signals.
        # HINT: Make sure the done signals are floats when you return them.
        sample: list[Transition] = random.sample(self.q, batch_size)
        ds = sample[0].state.size(dim=0)
        states, actions, rewards, next_states, dones = torch.zeros([batch_size, ds]), torch.zeros(batch_size, dtype=torch.int64), torch.zeros(batch_size), torch.zeros([batch_size,ds]), torch.zeros(batch_size)
        for i, t in enumerate(sample):
            states[i] = t.state
            actions[i] = t.action
            rewards[i] = t.reward
            next_states[i] = t.next_state
            dones[i] = float(t.done)
        return states, actions, rewards, next_states, dones