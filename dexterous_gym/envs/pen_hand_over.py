import numpy as np
import os
from gym import utils
from dexterous_gym.core.two_hands_env import TwoHandsEnv

dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_XML = dir_path+"/hand/pen_handover.xml"
DEFAULT_RANGE_1 = np.array([[0.96, 1.04], [0.81, 0.89], [0.2, 0.26]])
DEFAULT_RANGE_2 = np.array([[0.96, 1.04], [0.60, 0.68], [0.2, 0.26]])

class PenHandOver(TwoHandsEnv, utils.EzPickle):
    def __init__(self, target_rotation='xyz', reward_type='dense', target_range_1=DEFAULT_RANGE_1,
                 target_range_2=DEFAULT_RANGE_2, break_up_obs=False, distance_threshold=0.01,
                 rotation_threshold=0.1, dist_multiplier=20.0):
        self.dist_multiplier = dist_multiplier
        utils.EzPickle.__init__(self, 'random', target_rotation, reward_type)
        TwoHandsEnv.__init__(self, model_path=MODEL_XML, initial_qpos=None, target_position_range_1=target_range_1,
                             target_position_range_2=target_range_2, reward_type=reward_type, target_rotation=target_rotation,
                             distance_threshold=distance_threshold, moving_hands=False, rotation_threshold=rotation_threshold,
                             break_up_obs=break_up_obs, two_objects=False, randomise_initial_rotation=False)

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, desired_goal).astype(np.float32)
            return (success - 1.0)
        else:
            d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
            dist = self.dist_multiplier*d_pos + d_rot
            return np.exp(-1.0*dist)

    def _viewer_setup(self):
        middle_id = self.sim.model.site_name2id('centre-point')
        lookat = self.sim.data.site_xpos[middle_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.2
        self.viewer.cam.azimuth = 180.0
        self.viewer.cam.elevation = -55.0