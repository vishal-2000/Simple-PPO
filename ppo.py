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
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.env_util import make_vec_env

from pushGymEnv import pushGymEnv

def main():

    model_name = 'test_model0'

    env = make_vec_env(pushGymEnv, n_envs=1) # pushGymEnv()
    env = VecNormalize(env, norm_obs=True, norm_reward=True)# , clip_obs=10.)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10, progress_bar=True) # Total number of env steps = 25000
    print("----------------------------Training Complete----------------------------")
    model.save(os.path.join(currentdir, 'Results/New_Rewards', model_name))
    stats_path = os.path.join(currentdir, 'Results/New_Rewards', model_name+"vec_normalize.pkl")
    env.save(stats_path)

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

