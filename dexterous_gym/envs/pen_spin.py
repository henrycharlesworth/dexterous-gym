import numpy as np
from gym.envs.robotics.hand.manipulate import HandPenEnv

class PenSpin(HandPenEnv):
    def __init__(self, direction=1, alpha=1.0):
        self.direction = direction #-1 or 1
        self.alpha = alpha
        super(PenSpin, self).__init__()
        self.bottom_id = self.sim.model.site_name2id("object:bottom")
        self.top_id = self.sim.model.site_name2id("object:top")
        self._max_episode_steps = 250
        self.observation_space = self.observation_space["observation"]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        done = False
        info = {}
        reward = self.compute_reward()
        return obs["observation"], reward, done, info

    def compute_reward(self):
        bottom_z = self.sim.data.site_xpos[self.bottom_id].ravel()[-1]
        top_z = self.sim.data.site_xpos[self.top_id].ravel()[-1]
        reward_1 = -15 * np.abs(bottom_z - top_z)  # want pen to be in x-y plane
        if top_z < 0.12:
            reward_1 -= 10.0
        if bottom_z < 0.12:
            reward_1 -= 10.0
        qvel = self.sim.data.get_joint_qvel('object:joint')
        reward_2 = self.direction * qvel[3]
        return self.alpha * reward_2 + reward_1

    def reset(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = np.array([10000,10000,10000,0,0,0,0]) #hide offscreen
        obs = self._get_obs()
        return obs