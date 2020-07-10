import numpy as np
import os
from gym import utils
from dexterous_gym.core.two_hands_env import TwoHandsEnv

dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_XML = dir_path+"/hand/two_block_catch_underarm.xml"
DEFAULT_RANGE_1 = np.array([[0.97, 1.03], [0.38, 0.45], [0.15, 0.35]])
DEFAULT_RANGE_2 = np.array([[0.97, 1.03], [0.90, 0.97], [0.15, 0.35]])

class TwoBlockCatchUnderarm(TwoHandsEnv, utils.EzPickle):
    def __init__(self, target_rotation='xyz', reward_type='dense', target_range_1=DEFAULT_RANGE_1,
                 target_range_2=DEFAULT_RANGE_2, break_up_obs=False, distance_threshold=0.01,
                 rotation_threshold=0.1, dist_multiplier=50.0):
        self.dist_multiplier = dist_multiplier
        utils.EzPickle.__init__(self, 'random', target_rotation, reward_type)
        TwoHandsEnv.__init__(self, model_path=MODEL_XML, initial_qpos=None, target_rotation=target_rotation,
                             target_position_range_1=target_range_1, target_position_range_2=target_range_2,
                             reward_type=reward_type, distance_threshold=distance_threshold, moving_hands=True,
                             rotation_threshold=rotation_threshold, break_up_obs=break_up_obs, two_objects=True)

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, desired_goal).astype(np.float32)
            return (success - 1.0)
        else:
            (d_pos_1, d_rot_1), (d_pos_2, d_rot_2) = self._goal_distance(achieved_goal, desired_goal)
            dist_1 = self.dist_multiplier*d_pos_1 + d_rot_1
            dist_2 = self.dist_multiplier*d_pos_2 + d_rot_2
            return np.exp(-0.2*dist_1) + np.exp(-0.2*dist_2)

    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('robot0:palm')
        middle_id = self.sim.model.site_name2id('centre-point')
        # lookat = self.sim.data.body_xpos[body_id]
        lookat = self.sim.data.site_xpos[middle_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = 180.0
        self.viewer.cam.elevation = -55.0
