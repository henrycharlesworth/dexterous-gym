import numpy as np
import os
from gym import utils
from dexterous_gym.core.two_hands_env import TwoHandsEnv

dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_XML = dir_path+"/hand/egg_catch_overarm.xml"
DEFAULT_RANGE_1 = np.array([[0.9, 1.1], [-0.05, 0.05], [0.4, 0.5]])
DEFAULT_RANGE_2 = np.array([[0.9, 1.1], [1.25, 1.35], [0.4, 0.5]])

init_qpos = np.array([
    -1.69092e-16, 1.28969e-13, -0.0983158, 0.00671436, 8.81387e-05, -6.18241e-07, 0.00953895, 0.00197672, 0.000177327,
    0.00967983, 0.00965064, 0.0203301, 0.000178503, 0.0096801, 0.00965076, 0.0203529, 0.000178503, 0.00968019, 0.00965077,
    0.0203259, 0.00969491, 0.000176805, 0.00967987, 0.00965127, 0.021756, -0.000176283, 0.00969183, 0.000157871, 0.00720181,
    -0.00959593, 1.52536e-12, 2.07967e-12, -0.0983158, -0.438026, 6.86275e-06, 3.1689e-06, -0.111061, -0.688919, -0.00154806,
    0.0115532, 0.746703, 0.994549, -0.0971119, 0.201292, 0.777997, 0.885088, -0.0269409, 0.0686252, 0.903907, 1.11351,
    0.00921815, -0.00116265, 0.00948305, 1.32187, 1.56106, -0.322045, 0.199435, 0.0400855, -0.51436, -1.41563, 1, -0.2,
    0.40267, 1, 0, 0, 0, 1, 0.7, -24370.4, 1, 0, 0, 0
])

class EggCatchOverarm(TwoHandsEnv, utils.EzPickle):
    def __init__(self, target_rotation='xyz', reward_type='dense', target_range_1=DEFAULT_RANGE_1,
                 target_range_2=DEFAULT_RANGE_2, break_up_obs=False, distance_threshold=0.01,
                 rotation_threshold=0.1, dist_multiplier=50.0):
        self.dist_multiplier = dist_multiplier #scale of pos distance vs rot distance for reward.
        utils.EzPickle.__init__(self, 'random', target_rotation, reward_type)
        TwoHandsEnv.__init__(self, model_path=MODEL_XML, initial_qpos=None, target_rotation=target_rotation,
                             target_position_range_1=target_range_1, target_position_range_2=target_range_2,
                             reward_type=reward_type, distance_threshold=distance_threshold, moving_hands=True,
                             rotation_threshold=rotation_threshold, break_up_obs=break_up_obs, two_objects=False)

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, desired_goal).astype(np.float32)
            return (success - 1.0)
        else:
            d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
            dist = self.dist_multiplier*d_pos + d_rot
            return np.exp(-0.2*dist)

    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('robot0:palm')
        middle_id = self.sim.model.site_name2id('centre-point')
        # lookat = self.sim.data.body_xpos[body_id]
        lookat = self.sim.data.site_xpos[middle_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = 180.0
        self.viewer.cam.elevation = -15.0

    def _reset_sim(self):
        for i in range(len(init_qpos)):
            self.initial_state.qpos[i] = init_qpos[i]
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True