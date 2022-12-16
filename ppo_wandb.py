'''PPO with very sparse reward

Reward Specification:
1. Edge reached - (r=1, done=True)
2. Object fell on ground - (r=-0.01, done=True)
3. Other cases - (r=-0.01, done=False)
'''


# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


from stable_baselines3 import PPO 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecVideoRecorder

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import dummy_vec_env, vec_video_recorder

import wandb
from wandb.integration.sb3 import WandbCallback

from pushGymEnv import pushGymEnv

def main():

    model_name = 'test_model0'

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 25,
        "env_name": "PyBullet_Custom"
    }

    run = wandb.init(
        project='sb3',
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    def make_env():
        env = pushGymEnv() # pushGymEnv()
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env]) # dummy_vec_env([make_env])

    # env = VecVideoRecorder(env, f"./videos/{run.id}", record_video_trigger=lambda x: x % 5)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)# , clip_obs=10.)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)

    model = PPO(config['policy_type'], env, verbose=1, tensorboard_log=f"./runs/{run.id}")
    model.learn(
        total_timesteps=config['total_timesteps'], 
        progress_bar=True,
        callback=WandbCallback(
            gradient_save_freq=1,
            model_save_path=f"./models/{run.id}",
            verbose=2
        )
    ) # Total number of env steps = 25000
    print("----------------------------Training Complete----------------------------")
    run.finish()
    # model.save(os.path.join(currentdir, 'Results/New_Rewards', model_name))
    # stats_path = os.path.join(currentdir, 'Results/New_Rewards', model_name+"vec_normalize.pkl")
    # env.save(stats_path)

if __name__=="__main__":
    main()

# env = make_vec_env("CartPole-v1", n_envs=1)

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo_cartpole")

# del model

# model = PPO.load("ppo_cartpole")
# obs = env.reset()

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

