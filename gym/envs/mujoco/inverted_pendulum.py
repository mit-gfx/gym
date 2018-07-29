import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import IPython

class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    
    def __init__(self):
        self.timestep = 0
        
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        #notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2) and self.timestep < 1000
        notdone = np.isfinite(ob).all() and self.timestep < 999
        done = not notdone
        self.timestep += 1
        return ob, reward, done, {}

    def reset_model(self):
        self.timestep = 0
        qpos = np.array([0.0, np.pi]) + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = np.array([0.0, 0.0]) + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        #qpos = np.array([0.0, np.pi])
        #qvel = np.array([0.0, 0.0])
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
