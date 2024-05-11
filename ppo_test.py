from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation
from snake import SnakeEnv

w,h = 10,9
env = FlattenObservation(SnakeEnv(w=w, h=h, food_count=5, head_relative_action=True, render_mode="human", truncation_lim=(w*h)**2))
model = PPO.load("./best_models/best_model.zip")

obs, _ = env.reset()
term, trunc = False, False
while not term and not trunc:
    action, _ = model.predict(obs)
    obs, rewards, term, trunc, _ = env.step(action)