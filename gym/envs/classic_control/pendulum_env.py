import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import torch
from torch.autograd import Variable
dtype = torch.DoubleTensor
import IPython

class PerfExpPendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
    
        #TODO: load stuff from file here
        self.max_torque=1.0
        self.dt=.05
        self.m = 1.0
        self.l = 1.0
        self.g = -9.81
        self.b = 0.1
        self.target = np.array([1.0, 0.0])
        self.i = 0
        self.max_speed = 1
        
        self.viewer = None

        high = np.array([2 * np.pi, self.max_speed])
        #high = np.array([np.inf, np.inf])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self.seed()
        


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def set_target(self, target):
        target_cast = [float(t) for t in target]
        self.target = np.array(target_cast)

    def step(self,u):
        th, thdot = self.state # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        b = self.b
        self.i += 1
        
        
        #Simulate a step:
        u = np.clip(u, -self.max_torque, self.max_torque) #TODO: make sure u can be a function of a policy directly
        self.last_u = u # for rendering
        newthdot = thdot + dt * (u - thdot * b + m * g * l * np.sin(th)) / (m * l * l)
        newth = th + dt * newthdot
        self.state = np.array([newth, newthdot])
        

        f1 = np.sin(newth) * newthdot * dt / 2.0
        f2 = u * u * dt / 2000.0 #TODO: generalize this
        
        costs = np.array([f1, f2])
        
        if self.i == 2000:
            done = True
            self.i = 0
        else:
            done = False
        
        return self._get_obs(), costs, done, {}
        

    def reset(self):
        high = np.array([np.pi, 1])
        #self.state = self.np_random.uniform(low=-high, high=high)
        self.state = np.array([0.0, 0.0])
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            
        if self.state is None: return None

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] - np.pi/2.0)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

