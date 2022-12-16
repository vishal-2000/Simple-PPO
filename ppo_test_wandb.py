import gym
import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from pushGymEnv import pushGymEnv


def make_env():
    env = pushGymEnv()
    env = gym.wrappers.Monitor(env, f"videos")      # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env) # record stats such as returns
    return env

config = {
    "policy": 'MlpPolicy',
    "total_timesteps": 20
}

wandb.init(
    config=config,
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    project="pushGymEnv",
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=True,
)

env = DummyVecEnv([make_env])
model = PPO(config['policy'], env, verbose=1, tensorboard_log=f"runs/ppo")
model.learn(total_timesteps=config['total_timesteps'])
wandb.finish()