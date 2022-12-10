# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


from stable_baselines3 import PPO 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.env_util import make_vec_env

from pushGymEnv import pushGymEnv

def main():
    model_name = 'test_model5'

    env = pushGymEnv(renders=True)

    model = PPO.load(os.path.join(currentdir, 'Results', model_name), env=env)

    print(evaluate_policy(model=model, env=env, n_eval_episodes=10, render=True, return_episode_rewards=True))

    

if __name__=='__main__':
    main()