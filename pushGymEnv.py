'''Gym type environment for push manipulation problem 

Author: Vishal Reddy Mandadi
'''

import os
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import numpy as np
import cv2
import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data
from pkg_resources import parse_version

from Environments.environment_sim import Environment
from Environments.single_object_case import TestCase1
from Environments.utils import get_true_heightmap

from Config.constants import (
    WORKSPACE_LIMITS, 
    TARGET_LOWER, # Target object
    TARGET_UPPER, # Target object
    orange_lower, # Bottom object
    orange_upper, # Bottom object 
    MIN_GRASP_THRESHOLDS
)

from Utils.actionUtils import get_push_end
from Utils.rewardUtils import get_max_extent_of_target_from_bottom, get_max_extent_of_target_from_bottom2, get_state_reward, get_min_distance_from_edges

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class pushGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def  __init__(self,
                    gamma1=1,
                    gamma2=1,
                    beta2=1,
                    beta3=1,
                    beta4=1,
                    maxActions=30, #15, # max num. of actions per episode
                    guiOn=False,
                    renders=False) -> None:

        print("init")
        self._observation = []
        self._action = None
        self.envStepCounter = 0
        self.terminateCurrentEpisode = False
        self.maxActions = maxActions
        self._renders = renders
        self.env = None
        self._p = None

        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4

        self.prevState = {
            'target_pose': np.zeros(shape=(4), dtype=float), # x, y, z, yaw
            'bottom_pose': np.zeros(shape=(4), dtype=float) # x, y, z, yaw
        }

        
        if self._renders or guiOn:
            self.env = Environment(gui=True)
            self._p = self.env.client_id
        else:
            self.env = Environment(gui=False)
            self._p = self.env.client_id

        self.seed()

        observationDim = 14 # if end-effector pose is not considered ([ox, oy, oz, o_yaw, ol, ob, ow, bx, by, bz, b_yaw, bl, bb, bw])
        actionDim = 5 # [px, py, pz, push_direction, push_distance]

        observation_high = np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 
                                    1, 1, 1, 2*np.pi, 1, 1, 1], dtype=float)
        observation_low = np.array([-1, -1, -1, -2*np.pi, 0, 0, 0, 
                                    -1, -1, -1, -2*np.pi, 0, 0, 0], dtype=float)

        action_high = np.array([WORKSPACE_LIMITS[0][1], WORKSPACE_LIMITS[1][1], WORKSPACE_LIMITS[2][1], 2*np.pi, 100.0], dtype=np.float32) # np.array([1, 1, 1, 2*np.pi, 1], dtype=float) # push distance in mm
        action_low = np.array([WORKSPACE_LIMITS[0][0], WORKSPACE_LIMITS[1][0], WORKSPACE_LIMITS[2][0], 0, 5.0], dtype=np.float32) # push distance in mm

        self.observation_space = spaces.Box(observation_low, observation_high, dtype=float)
        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)

        self.body_ids = None # self.body_ids[0] - Base object, self.body_ids[1] - top object

        # super().__init__()


    def reset(self):
        '''Reset env and create the test case
        '''
        self.env.reset() # Reset environment

        self.envStepCounter = 0
        self.terminateCurrentEpisode = False

        self.prevState['target_pose'] = self.prevState['target_pose']*0
        self.prevState['bottom_pose'] = self.prevState['bottom_pose']*0

        self.testcase = TestCase1(self.env) #, self._p)
        success = False
        while not success:
            self.body_ids, success = self.testcase.sample_test_case(bottom_obj='default') #'random')

        self._observation = self.getFullObservation()
        return self._observation
    
    def __del__(self):
        self._p = 0
        self.env = None
        self.testcase = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getFullObservation(self):
        '''Gets full observation
        '''
        bottomPos, bottomOrn = self._p.getBasePositionAndOrientation(self.body_ids[0])
        euler_orn = p.getEulerFromQuaternion(bottomOrn)
        bottomYaw = euler_orn[2]

        targetPos, targetOrn = self._p.getBasePositionAndOrientation(self.body_ids[1])
        euler_orn = p.getEulerFromQuaternion(targetOrn)
        targetYaw = euler_orn[2]

        obs = np.array( [
                targetPos[0], targetPos[1], targetPos[2], targetYaw,
                self.testcase.current_target_size[0], self.testcase.current_target_size[1], self.testcase.current_target_size[2],
                bottomPos[0], bottomPos[1], bottomPos[2], bottomYaw,  
                self.testcase.current_bottom_size[0], self.testcase.current_bottom_size[1], self.testcase.current_bottom_size[2]  
            ], dtype=float)
        return obs

    def step(self, action):
        '''Action Space description

        action = [x, y, z, theta, d] 
        (x, y, z) -> Push start position
        (theta) -> Push Direction
        (d) -> Push distance in mm

        Push end_pos = [push_start[0] + d*cos(theta), push_start[1] + d*sin(theta), push_start[2]]
        '''
        # print("Current State-Action: -----------------------------------------------------")
        # Get push parameters
        # print("Action: {}".format(action))
        action[4] = action[4]/1000
        self._action = action
        push_start = np.array([action[0], action[1], action[2]], dtype=float)
        push_end = get_push_end(push_start, action[3], action[4])
        # Perform push
        success = self.env.push(push_start, push_end)
        if success==False:
            pass
            # print("Action failed to complete")
        # Get observatino after push
        # print("PreviousObservation: {}".format(self._observation))

        self._observation = self.getFullObservation()

        # print("Action: {}\nNextObservation: {}".format(action, self._observation))

        self.envStepCounter += 1
        reward = self._reward()
        done = self._termination(reward=reward)

        self.prevState['target_pose'] = self._observation[0:4] # x, y, z, yaw of target
        self.prevState['bottom_pose'] = self._observation[7:11] # x, y, z, yaw of base

        return self._observation, reward, done, {}

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])

        for config in self.env.agent_cams:
            color_img, _, _ = self.env.render_camera(config)
        rgb_array = np.array(color_img)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
        # (_, _, px, _, _) = self._p.

    def getSimpleReward(self):
        '''Simple and sparse grasp reward
        '''
        bottomPos, bottomOrn = self._p.getBasePositionAndOrientation(self.body_ids[0])
        targetPos, targetOrn = self._p.getBasePositionAndOrientation(self.body_ids[1])


        
        max_extents = get_max_extent_of_target_from_bottom2(bottomPos=bottomPos, bottomOrn=bottomOrn, current_bottom_obj_size=self.testcase.current_bottom_size,
                                                            targetPos=targetPos, targetOrn=targetOrn, current_target_obj_size=self.testcase.current_target_size, is_viz=False)

        # cmap, hmap, _ = get_true_heightmap(self.env)
        # temp = cv2.cvtColor(cmap, cv2.COLOR_RGB2HSV)
        # target_mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
        # bottom_mask = cv2.inRange(temp, orange_lower, orange_upper)
        # bottomPos, bottomOrn = self._p.getBasePositionAndOrientation(self.body_ids[0])
        # targetPos, targetOrn = self._p.getBasePositionAndOrientation(self.body_ids[1])
        # max_extents = get_max_extent_of_target_from_bottom(bottomPos, bottomOrn, target_mask, bottom_mask, self.body_ids[0], self.testcase.current_bottom_size, is_viz=False)
        reward = get_state_reward(bottomPos=bottomPos, 
                                targetPos=targetPos, bottomSize=self.testcase.current_bottom_size, 
                                targetSize=self.testcase.current_target_size, max_extents=max_extents, 
                                MIN_GRASP_EXTENT_THRESH=MIN_GRASP_THRESHOLDS)

    def get_object_interaction_reward(self):
        '''Calculates and returns the Object Interaction Reward for the current (state, action, prev_state) pair
        '''
        targetPos, targetOrn = self._p.getBasePositionAndOrientation(self.body_ids[1])
        bottomPos, bottomOrn = self._p.getBasePositionAndOrientation(self.body_ids[0])
        targetYaw = p.getEulerFromQuaternion(targetOrn)[2]
        bottomYaw = p.getEulerFromQuaternion(bottomOrn)[2] 

        relativeTargetPos = np.array(targetPos) - np.array(bottomPos)
        relativeTargetYaw = targetYaw - bottomYaw

        prevRelativeTargetPos = self.prevState['target_pose'][0:3] - self.prevState['bottom_pose'][0:3]
        prevRelativeTargetYaw = self.prevState['target_pose'][3] - self.prevState['bottom_pose'][3]

        cartesian_diff = np.linalg.norm(relativeTargetPos - prevRelativeTargetPos)
        yaw_diff = abs(relativeTargetYaw - prevRelativeTargetYaw)

        return cartesian_diff, yaw_diff

    def get_goal_distance_reward(self):
        '''Calculates the reward based on the difference in the distance between 
            the (current-object and the edge) and the (previous-pose and the edge)
        '''
        goal_dist_rew = 0
        bottomPos, bottomOrn = self._p.getBasePositionAndOrientation(self.body_ids[0])
        targetPos, targetOrn = self._p.getBasePositionAndOrientation(self.body_ids[1])
        
        max_extents = get_max_extent_of_target_from_bottom2(bottomPos=bottomPos, bottomOrn=bottomOrn, current_bottom_obj_size=self.testcase.current_bottom_size,
                                                            targetPos=targetPos, targetOrn=targetOrn, current_target_obj_size=self.testcase.current_target_size, is_viz=False)

        # print("Bottom yaw: {}, and target yaw: {}".format(self.prevState['bottom_pose'][3][0], self.prevState['target_pose'][3][0]))
        prev_max_extents = get_min_distance_from_edges(bottomPos=self.prevState['bottom_pose'][0:3], bottomYaw=self.prevState['bottom_pose'][3], current_bottom_obj_size=self.testcase.current_bottom_size,
                                                            targetPos=self.prevState['target_pose'][0:3], targetYaw=self.prevState['target_pose'][3], current_target_obj_size=self.testcase.current_target_size)

        goal_dist_rew = np.max(max_extents) - np.max(prev_max_extents)

        return goal_dist_rew

    def get_basic_grasp_reward(self):
        '''Checks if the object is graspable or not (based on whether it is projecting out of the surface) and returns 
            the appropriate reward

            This is the simplest grasp reward
        '''
        bottomPos, bottomOrn = self._p.getBasePositionAndOrientation(self.body_ids[0])
        targetPos, targetOrn = self._p.getBasePositionAndOrientation(self.body_ids[1])
        
        max_extents = get_max_extent_of_target_from_bottom2(bottomPos=bottomPos, bottomOrn=bottomOrn, current_bottom_obj_size=self.testcase.current_bottom_size,
                                                            targetPos=targetPos, targetOrn=targetOrn, current_target_obj_size=self.testcase.current_target_size, is_viz=False)

        if (max_extents[0] > MIN_GRASP_THRESHOLDS[0]) or (max_extents[1] > MIN_GRASP_THRESHOLDS[1]): # Object is now graspable
            # print("Max extents: {}, MIN GRASP EXTENT: {}".format(max_extents, MIN_GRASP_THRESHOLDS))
            return 1.0
            self.terminateCurrentEpisode = True
        else:
            return 0.0

    def get_ee_obj_distance_reward(self):
        '''A reward signal that encourages to predict the ee_pose close to the object's COM

        Concentric regional in nature
        '''
        EOR = 0
        ee_obj_dist = np.linalg.norm(self.prevState['target_pose'][0:3] - self._action[0:3])

        threshold = 0.5 * np.max(np.array([
            self.testcase.current_target_size[0],
            self.testcase.current_target_size[1],
            self.testcase.current_target_size[2]
        ], dtype=float))

        if ee_obj_dist > threshold:
            EOR = -1 + np.exp(1/(0.8 + ee_obj_dist)) # Adding 0.5 to prevent the reward from overshadowing other rewards
        else:
            EOR = -1 + np.exp(1/(0.8 + threshold))

        return EOR
        


    def _reward(self):
        '''Return the reward obtained after performing the current action
        '''
        
        oir1, oir2 = self.get_object_interaction_reward()
        gdr = self.get_goal_distance_reward()
        er = self.get_basic_grasp_reward() # Called the edge reward (er)
        eor = self.get_ee_obj_distance_reward()

        reward = self.gamma1*oir1 + self.gamma2*oir2 + self.beta2*gdr + self.beta3*er + self.beta4*eor

        # print("Component Rewards:\nOIR1: {}\tOIR2: {}\tGDR: {}\tER: {}\tAggregate: {}".format(oir1, oir2, gdr, er, reward))

        return reward

    def _termination(self, reward):
        '''Check if this is the termination state and return the bool value accordingly
        '''
        targetPos, _ = self._p.getBasePositionAndOrientation(self.body_ids[1])
        bottomPos, _ = self._p.getBasePositionAndOrientation(self.body_ids[0])
        done = False
        if self.terminateCurrentEpisode==True: # Reached the edge
            done=True
        if targetPos[2] < (bottomPos[2] + self.testcase.current_bottom_size[2]/2 + self.testcase.current_target_size[2]/2 - 0.005): # Object fell on the ground
            done=True
        if self.envStepCounter >= self.maxActions: # Takes more than max actions
            done=True

        return done


    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step


if __name__=='__main__':
    # env = pushGymEnv(renders=True)
    # env.reset()
    # env2 = pushGymEnv(renders=False)
    # env2.reset()
    # import time
    # time.sleep(10)
    pass
