from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
file_path = Path(__file__).resolve().parent

from snake import SnakeEnv

w,h = 9, 10
env = FlattenObservation(SnakeEnv(w=w, h=h, food_count=1, absolute_state=False, head_relative_state=True, head_relative_action=True, render_mode=None, truncation_lim=(w*h)**2))
models = [f"{file_path}/best_models/best_model.zip", f"{file_path}/model_checkpoints/rl_model_10000000_steps.zip", f"{file_path}/model_checkpoints/rl_model_20000000_steps.zip", f"{file_path}/model_checkpoints/rl_model_30000000_steps.zip", f"{file_path}/model_checkpoints/rl_model_40000000_steps.zip", f"{file_path}/model_checkpoints/rl_model_50000000_steps.zip", f"{file_path}/model_checkpoints/rl_model_60000000_steps.zip"]

for model in models:
    model = PPO.load(model)
    
    rew = 0
    ep_len = 0
    for i in range(1000):
        obs, _ = env.reset()
        term, trunc = False, False
        while not term and not trunc:
            action, _ = model.predict(obs)
            obs, reward, term, trunc, _ = env.step(action)
            rew += reward
            ep_len += 1
    print(rew/(i+1))
    print(ep_len/(i+1))