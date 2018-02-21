import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pusher.xml', 5)

    def _step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = -90
        self.viewer.cam.distance = 1.8
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0.3
        self.viewer.cam.lookat[2] = 0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.15, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])


class PusherEnv2(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pusher2.xml', 5)
        np.random.seed(self._seed()[0] % (1 << 32))

    def _step(self, a):
        obj_pos = self.get_body_com("object")
        arm2obj = obj_pos - self.get_body_com("tips_arm")
        obj2goal = obj_pos - self.get_body_com("goal")

        reward_arm2obj = - np.linalg.norm(arm2obj)
        reward_obj2goal = - np.linalg.norm(obj2goal)
        reward_ctrl = - np.square(a).sum()
        reward = reward_obj2goal + 0.1 * reward_ctrl + 0.5 * reward_arm2obj

        # Convert control relative object to relative robot
        absolute_control = self._from_relative_control(a)
        self.do_simulation(absolute_control, self.frame_skip)

        obs = self._get_obs()
        done = False
        return obs, reward, done, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = -90
        self.viewer.cam.distance = 1.8
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0.3
        self.viewer.cam.lookat[2] = 0

    def reset_model(self):
        qpos = self.init_qpos
        self.cylinder_pos = np.array([-0.15, 0.00]) + self.np_random.uniform(low=-0.1, high=0.1, size=2)

        qpos[2] = self.np_random.uniform(low=-0.3, high=0.0)
        qpos[-4:-2] = self.cylinder_pos
        self.set_state(qpos, self.init_qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:4],
            self.model.data.qvel.flat[:4],
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])

    def _from_relative_control(self, rel):
        obj = self.get_body_com('object')
        cur = self._get_obs()[:4]
        goal = np.array([obj[0] + rel[0], obj[1] + rel[1], obj[2] + rel[2], rel[3]])
        return goal - cur

    #############################################
    # Convenience functions for post processing #
    #############################################

    @staticmethod
    def goal_from_obs(obs):
        """Returns the goal position encoded in an observation"""
        return obs[-3:-1]

    @staticmethod
    def cylinder_from_obs(obs):
        """Returns the cylinder position encoded in an observation"""
        return obs[-6:-4]

    @staticmethod
    def obs_with_goal(obs, goal):
        """Returns an observation with a new goal set"""
        obs_ = obs.copy()
        obs_[-3:-1] = goal
        return obs_

    @staticmethod
    def reward_from_obs(obs):
        """Returns the reward calculated from an observation"""
        arm2obj = PusherEnv.goal_from_obs(obs) - PusherEnv.cylinder_from_obs(obs)
        obj2goal = PusherEnv.goal_from_obs(obs) - PusherEnv.cylinder_from_obs(obs)
        reward = -float(np.linalg.norm(obj2goal) > 1e-6)
        return reward


if __name__ == '__main__':
    env = PusherEnv2()
    import matplotlib.pyplot as plt
    o = env.reset()
    q = o[:4]
    a = np.array([0.00, 0.00, 0.20, 0.00])
    for _ in range(10):
        o, r, _, _ = env.step(a)
        env.render()
