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

import argparse
import json

def main(gamma1, gamma2, beta2, beta3, beta4, timesteps, model_name):

    model_name = model_name

    env_args = {
        'gamma1': gamma1,
        'gamma2': gamma2,
        'beta2': beta2,
        'beta3': beta3,
        'beta4': beta4
    }

    total_timesteps = timesteps

    model_args = {
        "model_name": model_name,
        "env_config": env_args,
        "total_timesteps": total_timesteps,
        "policy_type": 'MlpPolicy',
        "Model": "PPO"
    }

    print(model_args)

    env = make_vec_env(pushGymEnv, n_envs=1, env_kwargs=env_args) # pushGymEnv()
    env = VecNormalize(env, norm_obs=True, norm_reward=True)# , clip_obs=10.)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)

    model = PPO(model_args['policy_type'], env, verbose=1)
    model.learn(total_timesteps=model_args['total_timesteps'], progress_bar=True) # Total number of env steps = 25000
    print("----------------------------Training Complete----------------------------")

    new_dir = os.path.join(currentdir, f'Results/{model_name}')

    os.makedirs(new_dir)

    with open(os.path.join(new_dir, 'info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_args, f, ensure_ascii=False, indent=4)

    model.save(os.path.join(new_dir, model_name)) # currentdir, 'Results/New_Rewards'
    stats_path = os.path.join(new_dir, model_name+"vec_normalize.pkl")
    env.save(stats_path)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Inputs to Reward Hyperparams")
    parser.add_argument('--gamma1', type=float, default=1, help='Value of gamma1')
    parser.add_argument('--gamma2', type=float, default=1, help="Value of gamma2")
    parser.add_argument('--beta2', type=float, default=1, help="Value of beta2")
    parser.add_argument('--beta3', type=float, default=1, help="Value of beta3")
    parser.add_argument('--beta4', type=float, default=1, help="Value of beta4")
    parser.add_argument('--timesteps', type=int, default=20, help="Number of timesteps to run the algo")
    parser.add_argument('--model_name', type=str, default="model0", help="Name of the model, must be unique")

    args = parser.parse_args()

    main(args.gamma1, args.gamma2, args.beta2, args.beta3, args.beta4, args.timesteps, args.model_name)

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

