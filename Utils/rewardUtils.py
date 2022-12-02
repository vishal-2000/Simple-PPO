'''Reward system design and helper functions

Author: Vishal Reddy Mandadi
'''

import numpy as np
import cv2
import pybullet as p
from Config.constants import (
    PIXEL_SIZE, 
    WORKSPACE_LIMITS
    # MIN_GRASP_THRESHOLDS
)

def get_max_extent_of_target_from_bottom(bottomPos, bottomOrn, target_mask, bottom_mask, bottom_obj_body_id, current_bottom_obj_size, is_viz=False):
    '''Calculates in pixels, the max outward extent of the target object from the bottom object
    '''
    # bottomPos, bottomOrn = client_id.getBasePositionAndOrientation(bottom_obj_body_id)
    euler_orn = p.getEulerFromQuaternion(bottomOrn)
    # print(bottomPos, bottomOrn, euler_orn)
    for i in range(0, 224):
        for j in range(0, 224):
            if j==112:
                continue
            tan_val = np.tan(euler_orn[2])
            val1 = np.sqrt(((j-112) - (tan_val)*(i-112))**2)
            val2 = np.sqrt(((j-112)*(tan_val) + (i-112))**2)
            if val1 <= 1:
                bottom_mask[i, j] = 255
            if val2 <= 1:
                bottom_mask[i, j] = 255    
    if is_viz:
        cv2.imshow("bottom mask with it's principle axes", bottom_mask) # Bottom object is being masked properly - checked
        cv2.waitKey(0)

    # Find the corners for the target object
    dst = cv2.cornerHarris(target_mask, 2, 9, 0.04) # Corner detection done perfectly!!
    temp = np.zeros(shape=(224, 224))
    temp[dst>0.01*dst.max()] = 255
    if is_viz==True:
        cv2.imshow("Corners in the target", temp) # Corner detection done perfectly!! - Checked
        cv2.waitKey(0)
    corners = np.where(temp==255)

    ######## Start here
        ## Code up the max extent calculation and return the extent ##
    orn_vec = np.array([np.cos(euler_orn[2]), np.sin(euler_orn[2])])
    orn_perpendicular = np.array([-np.sin(euler_orn[2]), np.cos(euler_orn[2])])
    max_extents = np.ones(shape=(2,))*(-10.0)
    for corner in zip(corners[0], corners[1]):
        corner_in_real = np.array([
            corner[0]*PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
            corner[1]*PIXEL_SIZE + WORKSPACE_LIMITS[1][0]
        ])
        corner_wrt_bottom_pos = corner_in_real - bottomPos[0:2]
        ext1 = abs(np.dot(orn_vec, corner_wrt_bottom_pos)) - current_bottom_obj_size[0]/2 # length component
        ext2 = abs(np.dot(orn_perpendicular, corner_wrt_bottom_pos)) - current_bottom_obj_size[1]/2 # breadth/width component
        if ext1 >= max_extents[0]:
            max_extents[0] = ext1
        if ext2 >= max_extents[1]:
            max_extents[1] = ext2
    # current_bottom_obj_size[0]
    return max_extents
        # max_extents[0] = np.max(np.dot(orn_vec, ))
    ######## Ends here
    # return 0


def get_state_reward(bottomPos, 
                    targetPos, bottomSize, 
                    targetSize, max_extents, 
                    MIN_GRASP_EXTENT_THRESH):
    '''
    '''
    if targetPos[2] < bottomPos[2] + bottomSize[2]/2 + targetSize[2]/2 - 0.005: # Object fell down
        # print("Object fell down")
        return -0.01
    elif (max_extents[0] > MIN_GRASP_EXTENT_THRESH[0]) or (max_extents[1] > MIN_GRASP_EXTENT_THRESH[1]): # Object is now graspable
        # print("Max extents: {}, MIN GRASP EXTENT: {}".format(max_extents, MIN_GRASP_EXTENT_THRESH))
        return 1.0
    else: # Nothing much happened
        return -0.01 # for fast achievement of goal