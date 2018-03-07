import os
import six

import gym
from gym.utils import seeding

import mujoco_py
from mujoco_py.mjlib import mjlib
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import random


class MujocoSpecial(gym.Env):
    """Special superclass for MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.MjModel(fullpath)
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()

        bounds = self.model.actuator_ctrlrange.copy()
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    @property
    def observation_space(self):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError

    # -----------------------------

    def _reset(self):
        mjlib.mj_resetData(self.model.ptr, self.model.data.ptr)
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer.autoscale()
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model._compute_subtree()  # pylint: disable=W0212
        self.model.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.model.data.ctrl = ctrl
        for _ in range(n_frames):
            self.model.step()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            self._get_viewer().render()
            data, width, height = self._get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().loop_once()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.body_comvels[idx]

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.xmat[idx].reshape((3, 3))

    def state_vector(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])

class YumiEnvSimple(MujocoSpecial, utils.EzPickle):

    def __init__(self, frame_skip=1):

        # Super class
        file_path = os.path.abspath(__file__)
        model_path = os.path.join(os.path.dirname(file_path), 'yumi', 'yumi.xml')
        super(YumiEnvSimple, self).__init__(model_path, frame_skip=1)

        # Child class        
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.low=low
        self.high=high
        utils.EzPickle.__init__(self)
        self.frame_skip = frame_skip

        #set up evil force
        self._adv_bindex = body_index(self.model,'gripper_r_finger_l')
        self.adv_max_force = 0.1
        #high_adv = np.ones(2)*adv_max_force
        #low_adv = -high_adv
        #self.adv_action_space = spaces.Box(low_adv, high_adv)

    @property
    def observation_space(self):
        high = np.inf * np.ones(self._get_obs().shape)
        low = -high
        return gym.spaces.Box(low, high)

    @property
    def action_space(self):
        return gym.spaces.Box(self.low, self.high)

    #evil force
    def _adv_to_xfrc(self, adv_act):
        new_xfrc = self.model.data.xfrc_applied*0.0
        new_xfrc[self._adv_bindex] = np.array([adv_act[0], adv_act[1], adv_act[2], adv_act[3], adv_act[4], adv_act[5]])
        self.model.data.xfrc_applied = new_xfrc

    def _step(self, a):
        #evil forces
        randf =np.random.uniform(low=-self.adv_max_force, high=self.adv_max_force, size=(6,))
        self._adv_to_xfrc(randf)
        #make sure actions are inbound        
        a = np.clip(a, self.low, self.high)        
        self.do_simulation(a, 1)
        ob = self._get_obs()
        reward = self.reward(a)
        done = False
        return ob, reward, done, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 180
        self.viewer.cam.distance = 1.5
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.2

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        # separate shoulder joints to avoid initial collision
        qpos[0] = -1.0
        qpos[9] =  1.0

        qpos[1] = 0.3

        self.set_state(qpos, qvel)
        return self._get_obs()

    def reward(self, a):
        arm = np.concatenate([body_pos(self.model, 'gripper_r_base'),body_quat(self.model, 'gripper_r_base')])
        goal =np.concatenate([body_pos(self.model, 'goal'),body_quat(self.model, 'goal')])
        arm2goal = np.linalg.norm(arm - goal)

        return -arm2goal*100

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            body_pos(self.model, 'gripper_r_base'),
            body_quat(self.model, 'gripper_r_base')
        ])


def body_frame(model, body_name):
    """
    Returns the rotation matrix to convert to the frame of the named body
    """
    ind = body_index(model, body_name)
    b = model.data.xpos[ind]
    q = model.data.xquat[ind]
    qr, qi, qj, qk = q
    s = np.square(q).sum()
    R = np.array([
        [1 - 2 * s * (qj ** 2 + qk ** 2), 2 * s * (qi * qj - qk * qr), 2 * s * (qi * qk + qj * qr)],
        [2 * s * (qi * qj + qk * qr), 1 - 2 * s * (qi ** 2 + qk ** 2), 2 * s * (qj * qk - qi * qr)],
        [2 * s * (qi * qk - qj * qr), 2 * s * (qj * qk + qi * qr), 1 - 2 * s * (qi ** 2 + qj ** 2)]
    ])
    return R


def body_index(model, body_name):
    return model.body_names.index(six.b(body_name))


def body_pos(model, body_name):
    ind = body_index(model, body_name)
    return model.data.xpos[ind]

def body_quat(model,body_name):
    ind = body_index(model, body_name)
    return model.data.xquat[ind]


def rotational_axis(model, body_name, local_axis):
    R = body_frame(model, body_name)
    return R @ local_axis


def reference_vector(model, body_name, eef_name):
    pj = body_pos(model, body_name)
    pe = body_pos(model, eef_name)
    return pe - pj


def jacobian_inv(model):
    J = np.zeros((6, 7))
    for j, body_name in enumerate([
        'yumi_link_1_r',
        'yumi_link_2_r',
        'yumi_link_3_r',
        'yumi_link_4_r',
        'yumi_link_5_r',
        'yumi_link_6_r',
        'yumi_link_7_r',
    ]):
        k = rotational_axis(model, body_name, np.array([[0, 0, 1]]).T)
        r = reference_vector(model, body_name, 'gripper_r_base')
        b = body_pos(model, body_name)
        c = np.cross(k.reshape(1, 3), r.reshape(1, 3)).flatten()
        J[:3, j] = c.flatten()
        J[3:, j] = k.flatten()
    return J.T @ np.linalg.inv(J @ J.T + np.eye(6) * 1e-9)


if __name__ == '__main__':
    import gym
    yumi = gym.make('Yumi-Simple-v0')
    yumi.reset()
    #a = np.array([0.0, 0.0, 0.10])
    for _ in range(1):
        yumi.reset()
        for t in range(2000):
            _, r, _, _ = yumi.step(a)
            yumi.render()
