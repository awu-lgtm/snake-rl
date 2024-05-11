from gymnasium.wrappers import FlattenObservation, FrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
import optuna
from snake.snake_env import snake_head_relative
from pathlib import Path

file_path = Path(__file__).resolve().parent

def get_hp(trial: optuna.trial.Trial):
    trial.set_user_attr("total_timesteps", 1_000_000_00)
    trial.set_user_attr("state", "head_relative_state")
    trial.set_user_attr("policy", "MlpPolicy")
    trial.set_user_attr("w", 9)
    trial.set_user_attr("h", 10)

    trial.suggest_float("lr", 1e-5, 1e-3)
    trial.suggest_float("gamma", 0.9, 0.9999)
    trial.suggest_float("gae_lambda", 0.92, 0.98)
    trial.suggest_categorical("clipping_range", [0.1, 0.2, 0.3])


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1_000_000_000,
}
run = wandb.init(
    project="snake",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
)

if __name__ == "__main__":
    num_cpu = 32

    snake_env = lambda: FlattenObservation(snake_head_relative())
    env = make_vec_env(env_id=snake_env, n_envs=num_cpu)

    eval_callback = EvalCallback(eval_env=Monitor(snake_env()), eval_freq=1_000_000//num_cpu, n_eval_episodes=10, best_model_save_path=f"{file_path}/best_models/")
    checkpoint_callback = CheckpointCallback(save_freq=10_000_000//num_cpu, save_path=f"{file_path}/model_checkpoints/", save_replay_buffer=True, save_vecnormalize=True)
    wandb_callback = WandbCallback(gradient_save_freq=100, verbose=1, log="all")
    model = PPO(config["policy_type"], env, verbose=1, device="auto", tensorboard_log=f"{file_path}/log/{run.id}")
    model.learn(total_timesteps=config["total_timesteps"], callback=[eval_callback, checkpoint_callback, wandb_callback])