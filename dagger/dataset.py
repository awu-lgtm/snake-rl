import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from pathlib import Path
import os
from collections import deque

file_path = Path(__file__).resolve().parent

ExpertData = namedtuple('ExpertData', ('states', 'actions'))

class ExpertDataset(Dataset):
    def __init__(self, expert_data):
        self.states = expert_data.states
        self.actions = expert_data.actions
        
    def __len__(self):
        return self.states.size(0)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        
        return state, action
    
    def add_data(self, data):
        self.states = torch.cat([self.states, data.states], dim=0)
        self.actions = torch.cat([self.actions, data.actions], dim=0)

def get_dataloader(dataset, num_dataset_samples, batch_size):
    small_dset = dataset[:num_dataset_samples]
    small_states, small_actions = small_dset
    small_dset = ExpertDataset(ExpertData(small_states, small_actions))
    return DataLoader(small_dset, batch_size=batch_size, shuffle=True)

def make_dataset(policy, env, n_trajs):
    states = []
    actions = []
    for _ in range(n_trajs):
        traj_states = []
        traj_actions = []
        ob = env.reset()
        done = False
        
        while not done:
            action, _ = policy.predict(ob, deterministic=True)
            n_ob, _, done, _ = env.step(action)
            
            traj_states.append(torch.from_numpy(ob))
            traj_actions.append(torch.tensor(action))
            
            ob = n_ob
        
        states += traj_states
        actions += traj_actions
    
    state_tensor = torch.stack(states)
    action_tensor = torch.stack(actions)
    
    print(state_tensor.shape, action_tensor.shape)
    
    dataset = ExpertDataset(ExpertData(state_tensor, action_tensor))
    
    data_path = os.path.join(file_path, "data", "dataset.pt")
    torch.save(dataset, data_path)

def load_saved_dataset(batch_size):
    data_path = os.path.join(file_path, "data", "dataset.pt")
    dl = DataLoader(dataset=data_path, batch_size=batch_size, shuffle=True)
    return dl
    
if __name__ == '__main__':
    from snake.snake_env import snake_head_relative
    from head_relative.head_relative_policy import get_policy
    
    n_trajs = 1e4
    policy = get_policy()
    make_dataset(policy, snake_head_relative(), n_trajs)