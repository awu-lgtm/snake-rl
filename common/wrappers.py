from gymnasium import ObservationWrapper
import gymnasium as gym
from gymnasium.spaces import Dict, Box

class ImitationWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env, novice, expert):
        super().__init__(self, env)
        self.observation_space = Dict({"novice": env.observation_space[novice], "expert": env.observation_space[expert]})
        self.novice = novice
        self.expert = expert

    def observation(self, observation):
        observation = {"novice": observation[self.novice], "expert": observation[self.expert]}
        return observation