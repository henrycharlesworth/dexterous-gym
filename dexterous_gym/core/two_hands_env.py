import numpy as np
from gym.envs.robotics import rotations
from gym import error

from dexterous_gym.core.two_hand_robot_env import RobotEnv
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat

class TwoHandsEnv(RobotEnv):
    def __init__(self, model_path, initial_qpos, target_rotation, target_position_range_1, target_position_range_2,
                 reward_type="dense", n_substeps=20, distance_threshold=0.01, rotation_threshold=0.1,
                 ignore_z_target_rotation=False, break_up_obs=False, moving_hands=False, two_objects=False,
                 randomise_initial_rotation=True):
        """
        :param model_path (string): path to environment XML file
        :param initial_qpos (dict): define initial configuration
        :param target_rotation (string):
                - "ignore": target rotation ignored.
                - "xyz": fully randomised target rotation
                - "z": randomised target rotation around Z axis
        :param target_position_range_1 (np.array (3,2)): valid target positions for hand 1
        :param target_position_range_2 (np.array (3,2)): valid target positions for hand 2
        :param reward_type (string): "dense" or "sparse"
        :param n_substeps (int): number of simulation steps per call to step
        :param distance_threshold (float, metres): threshold at which position is "achieved"
        :param rotation_threshold (float, radians): threshold at which rotation is "achieved"
        :param ignore_z_target_rotation (bool): whether or not the Z axis of the target rotation is ignored
        :param break_up_obs (bool): if true, obs returns a dictionary with different entries for each hand,
            {"hand_1", "hand_2", "object", "achieved_goal", "desired_goal"} as opposed to {"observation", "achieved_goal", "desired_goal"}
        :param moving_hands (bool): whether or not the hands are able to move and rotate
        :param two_objects (bool): whether or not there are one or two objects
        """
        self.target_position_1 = target_position_range_1
        self.target_position_2 = target_position_range_2
        self.reward_type = reward_type
        self.target_rotation = target_rotation
        self.ignore_z_target_rotation = ignore_z_target_rotation
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.break_up_obs = break_up_obs
        self.moving_hands = moving_hands
        self.two_objects = two_objects
        self.randomise_initial_rotation = randomise_initial_rotation

        if moving_hands:
            self.n_actions = 52
        else:
            self.n_actions = 40

        assert self.target_rotation in ['ignore', 'xyz', 'z']
        assert self.reward_type in ['dense', 'sparse']
        initial_qpos = initial_qpos or {}

        super(TwoHandsEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=self.n_actions, initial_qpos=initial_qpos
        )

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        actuation_centre = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        self.sim.data.ctrl[:] = actuation_centre + action*actuation_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def _get_achieved_goal(self):
        object_1_qpos = self.sim.data.get_joint_qpos('object:joint')
        assert object_1_qpos.shape == (7,)
        if self.two_objects:
            object_2_qpos = self.sim.data.get_joint_qpos('object_2:joint')
            assert object_2_qpos.shape == (7,)
            return {"object_1": object_1_qpos.copy(), "object_2": object_2_qpos.copy()}
        return object_1_qpos

    def _goal_distance_single(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7

        d_rot = np.zeros_like(goal_b[..., 0])
        d_pos = np.linalg.norm(goal_a[..., :3] - goal_b[..., :3], axis=-1)

        if self.target_rotation != 'ignore':
            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]
            if self.ignore_z_target_rotation:
                euler_a = rotations.quat2euler(quat_a)
                euler_b = rotations.quat2euler(quat_b)
                euler_a[2] = euler_b[2]
                quat_a = rotations.euler2quat(euler_a)
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
            d_rot = angle_diff
        return d_pos, d_rot

    def _goal_distance(self, goal_a, goal_b):
        if self.two_objects:
            d_pos_1, d_rot_1 = self._goal_distance_single(goal_a["object_1"], goal_b["object_1"])
            d_pos_2, d_rot_2 = self._goal_distance_single(goal_a["object_2"], goal_b["object_2"])
            return (d_pos_1, d_rot_1), (d_pos_2, d_rot_2)
        else:
            return self._goal_distance_single(goal_a, goal_b)

    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError #each environment will implement this individually.

    def _is_success(self, achieved_goal, desired_goal, array=False):
        if self.two_objects:
            (d_pos_1, d_rot_1), (d_pos_2, d_rot_2) = self._goal_distance(achieved_goal, desired_goal)
            a_pos_1 = (d_pos_1 < self.distance_threshold).astype(np.float32)
            a_rot_1 = (d_rot_1 < self.rotation_threshold).astype(np.float32)
            a_1 = a_pos_1 * a_rot_1
            a_pos_2 = (d_pos_2 < self.distance_threshold).astype(np.float32)
            a_rot_2 = (d_rot_2 < self.rotation_threshold).astype(np.float32)
            a_2 = a_pos_2 * a_rot_2
            if array:
                return np.array([a_1, a_2])
            else:
                return a_1*a_2
        else:
            d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
            a_pos = (d_pos < self.distance_threshold).astype(np.float32)
            a_rot = (d_rot < self.rotation_threshold).astype(np.float32)
            return a_pos * a_rot

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        if self.randomise_initial_rotation:
            initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
            initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1., 1., size=3)
            offset_quat = quat_from_angle_and_axis(angle, axis)
            initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            initial_quat /= np.linalg.norm(initial_quat)
            initial_qpos = np.concatenate([initial_pos, initial_quat])
            self.sim.data.set_joint_qpos('object:joint', initial_qpos)
            if self.two_objects:
                initial_qpos_2 = self.sim.data.get_joint_qpos('object_2:joint').copy()
                initial_pos_2, initial_quat_2 = initial_qpos_2[:3], initial_qpos_2[3:]
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1., 1., size=3)
                offset_quat_2 = quat_from_angle_and_axis(angle, axis)
                initial_quat_2= rotations.quat_mul(initial_quat_2, offset_quat_2)
                initial_quat_2 /= np.linalg.norm(initial_quat_2)
                initial_qpos_2 = np.concatenate([initial_pos_2, initial_quat_2])
                self.sim.data.set_joint_qpos('object_2:joint', initial_qpos_2)


        for _ in range(10):
            self._set_action(np.zeros(self.n_actions))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        return True

    def _sample_goal(self, goal_side=None):
        """
        :param goal_side: which hand (1 or 2) is the target near. If None we automatically choose the side furthest away.
        """
        if self.two_objects:
            obj_state = self._get_achieved_goal()
            pos_1 = np.mean(self.target_position_1, axis=-1)
            pos_2 = np.mean(self.target_position_2, axis=-1)
            dist_1 = np.linalg.norm(obj_state["object_1"][:3]-pos_1)
            dist_2 = np.linalg.norm(obj_state["object_1"][:3] - pos_2)
            if dist_1 > dist_2:
                target_pos_1 = self.np_random.uniform(self.target_position_1[:, 0], self.target_position_1[:, 1])
                target_pos_2 = self.np_random.uniform(self.target_position_2[:, 0], self.target_position_2[:, 1])
            else:
                target_pos_1 = self.np_random.uniform(self.target_position_2[:, 0], self.target_position_2[:, 1])
                target_pos_2 = self.np_random.uniform(self.target_position_1[:, 0], self.target_position_1[:, 1])
        else:
            if goal_side is None:
                obj_state = self._get_achieved_goal()
                pos_1 = np.mean(self.target_position_1, axis=-1)
                pos_2 = np.mean(self.target_position_2, axis=-1)
                dist_1 = np.linalg.norm(obj_state[:3]-pos_1)
                dist_2 = np.linalg.norm(obj_state[:3]-pos_2)
                if dist_1 > dist_2:
                    goal_side = 1
                else:
                    goal_side = 2
            if goal_side == 1:
                target_pos_1 = self.np_random.uniform(self.target_position_1[:, 0], self.target_position_1[:, 1])
            else:
                target_pos_1 = self.np_random.uniform(self.target_position_2[:, 0], self.target_position_2[:, 1])

        if self.target_rotation == 'z':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0.0, 0.0, 1.0])
            target_quat_1 = quat_from_angle_and_axis(angle, axis)
            if self.two_objects:
                angle_2 = self.np_random.uniform(-np.pi, np.pi)
                target_quat_2 = quat_from_angle_and_axis(angle_2, axis)
        elif self.target_rotation == 'xyz':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1.0, 1.0, size=3)
            target_quat_1 = quat_from_angle_and_axis(angle, axis)
            if self.two_objects:
                angle_2 = self.np_random.uniform(-np.pi, np.pi)
                axis_2 = self.np_random.uniform(-1.0, 1.0, size=3)
                target_quat_2 = quat_from_angle_and_axis(angle_2, axis_2)
        elif self.target_rotation == 'ignore':
            target_quat_1 = np.zeros((4,))+0.1
            if self.two_objects:
                target_quat_2 = np.zeros((4,))+0.1
        else:
            raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))
        target_quat_1 /= np.linalg.norm(target_quat_1)
        if self.two_objects:
            target_quat_2 /= np.linalg.norm(target_quat_2)
            return {"object_1": np.concatenate([target_pos_1, target_quat_1]),
                    "object_2": np.concatenate([target_pos_2, target_quat_2])}
        else:
            return np.concatenate([target_pos_1, target_quat_1])

    def _get_obs(self):
        achieved_goal = self._get_achieved_goal()
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            names_1 = [n for n in self.sim.model.joint_names if n.startswith('robot') and not n.endswith('_2')]
            hand_qpos_1 = np.array([self.sim.data.get_joint_qpos(name) for name in names_1])
            hand_qvel_1 = np.array([self.sim.data.get_joint_qvel(name) for name in names_1])
            object_qvel_1 = self.sim.data.get_joint_qvel('object:joint')
            hand_1 = np.concatenate((hand_qpos_1, hand_qvel_1))
            names_2 = [n for n in self.sim.model.joint_names if n.startswith('robot') and n.endswith('_2')]
            hand_qpos_2 = np.array([self.sim.data.get_joint_qpos(name) for name in names_2])
            hand_qvel_2 = np.array([self.sim.data.get_joint_qvel(name) for name in names_2])
            hand_2 = np.concatenate((hand_qpos_2, hand_qvel_2))
            if self.two_objects:
                object_qvel_2 = self.sim.data.get_joint_qvel('object_2:joint')
                obj_1 = np.concatenate((achieved_goal["object_1"], object_qvel_1))
                obj_2 = np.concatenate((achieved_goal["object_2"], object_qvel_2))
                if self.break_up_obs:
                    return {
                        "hand_1": hand_1, "hand_2": hand_2, "obj_1": obj_1, "obj_2": obj_2, "achieved_goal": achieved_goal,
                        "desired_goal": self.goal.copy()
                    }
                else:
                    obs = np.concatenate((hand_1, hand_2, obj_1, obj_2))
                    return {
                        "observation": obs, "achieved_goal": achieved_goal, "desired_goal": self.goal.copy()
                    }
            else:
                obj_1 = np.concatenate((achieved_goal, object_qvel_1))
                if self.break_up_obs:
                    return {
                        "hand_1": hand_1, "hand_2": hand_2, "obj": obj_1, "achieved_goal": achieved_goal, "desired_goal": self.goal.copy()
                    }
                else:
                    return {
                        "observation": np.concatenate((hand_1, hand_2, obj_1)), "achieved_goal": achieved_goal, "desired_goal": self.goal.copy()
                    }
        return np.zeros(0), np.zeros(0)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _viewer_setup(self): #each environment defines this itself.
        raise NotImplementedError

    def _render_callback(self):
        if self.two_objects:
            goal_1 = self.goal["object_1"]
            goal_2 = self.goal["object_2"]
            self.sim.data.set_joint_qpos('target_2:joint', goal_2)
            self.sim.data.set_joint_qvel('target_2:joint', np.zeros((6,)))
            if 'object_hidden_2' in self.sim.model.geom_names:
                hidden_id = self.sim.model.geom_name2id('object_hidden_2')
                self.sim.model.geom_rgba[hidden_id, 3] = 1.0
        else:
            goal_1 = self.goal.copy()

        self.sim.data.set_joint_qpos('target:joint', goal_1)
        self.sim.data.set_joint_qvel('target:joint', np.zeros((6,)))
        if 'object_hidden' in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id('object_hidden')
            self.sim.model.geom_rgba[hidden_id, 3] = 1.0
        self.sim.forward()

    def render(self, mode='human', width=500, height=500):
        return super(TwoHandsEnv, self).render(mode, width, height)