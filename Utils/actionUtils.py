import copy
import numpy as np
# from Config.constants import PUSH_DISTANCE

def get_push_end(push_start, push_dir, push_distance):
    '''Get the end point for push, given the start point and the direction
    '''
    push_vec = np.array([np.cos(push_dir), np.sin(push_dir)])
    desired_push_end = copy.deepcopy(push_start)
    desired_push_end[:2] = push_start[:2] + push_distance*push_vec # orn_vec
    return desired_push_end