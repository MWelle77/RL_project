import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])


class ReachHerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def _step(self, a):
        x = self._get_obs()
        self.do_simulation(a, self.frame_skip)
        x_ = self._get_obs()
        done = False
        return x_, self.reward(x, a, x_), done, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    @staticmethod
    def reward(x, a, x_):
        if len(x.shape) == 1:
            fingertip = x[-6:-3]
            target = x[-3:]
            reward_dist = -np.linalg.norm(fingertip - target)
            reward_ctrl = -np.square(a).sum()
            return reward_dist + reward_ctrl
        else:
            fingertip = x[:, -6:-3]
            target = x[:, -3:]
            reward_dist = -np.linalg.norm(fingertip - target, axis=1)
            reward_ctrl = -np.square(a).sum(axis=1)
            return reward_dist + reward_ctrl

    @staticmethod
    def hindsight_obs(x, x_achieved):
        if len(x.shape) == 1:
            target = x_achieved[-3:]
            x_new = x.copy()
            x_new[-3:] = target
            return x_new
        else:
            target = x_achieved[:, -6:-3]
            x_new = x.copy()
            x_new[:, -3:] = target
            return x_new

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip"),
            self.get_body_com("target"),
        ])

class ReachHerBinaryEnv(ReachHerEnv):

    def __init__(self):
        super(ReachHerBinaryEnv, self).__init__()

    @staticmethod
    def reward(x, a, x_):
        eps = 1e-6
        if len(x.shape) == 1:
            fingertip = x_[-6:-3]
            target = x_[-3:]
            reward_dist = -float(np.linalg.norm(fingertip - target) > eps)
            return reward_dist
        else:
            fingertip = x_[:, -6:-3]
            target = x_[:, -3:]
            reward_dist = -(np.linalg.norm(fingertip - target, axis=1) > eps).astype(np.float32)
            return reward_dist


if __name__ == '__main__':
    env = ReachHerBinaryEnv()
    x = env.reset()
    for _ in range(1000):
        _, r, _, _ = env.step(env.action_space.sample())
        print(r)
        env.render()
