from gymnasium import ObservationWrapper
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiBinary, flatten_space, flatten
import numpy as np

class ImitationWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env, novice, expert):
        super().__init__(env)
        h,w = env.observation_space[novice].shape[0], env.observation_space[novice].shape[1]
        self.observation_space = Dict({"novice": Box(low=0, high=1, shape=(1,h,w), dtype=np.float32), "expert": flatten_space(env.observation_space[expert])})
        self.novice = novice
        self.expert = expert
        self.h = h
        self.w = w

    def observation(self, observation):
        observation = {"novice": observation[self.novice].reshape((1,self.h, self.w))/255, "expert": flatten(self.env.observation_space[self.expert], observation[self.expert])}
        return observation
    
class ExtractStateWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space["state"]

    def observation(self, observation):
        observation = observation["state"]
        return observation