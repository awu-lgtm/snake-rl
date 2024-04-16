from gym.envs.registration import register

register(
    id='snake-game-v0',
    entry_point='env:snake',
    max_episode_steps=300,
)