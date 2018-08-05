'''
a 1D spaceship, or brick
'''

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import IPython

class SpaceshipEnv(gym.Env):
    def __init__(self):
        self.phi = np.array([1.0]) #Just the mass
        self.H = 1000 #max horizon in steps
        self.goal = np.array([1.0, 0.0]) #Need to reach target location 1 on the number line at stationary velocity.       
        
        action_high = np.array([np.finfo(np.float32).max])
        self.action_space = spaces.Box(-action_high, action_high) #1 thruster input
        
        
        obs_high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-obs_high, obs_high) #x and v
        self.dt = 0.05
        
        self.seed(0)
        self.viewer = None
        self.reset()
        self.done_thresh = 0.01
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        #Action should be 1D
        
        #first, get the acceleration:
        #TODO: assuming the action is 1-D now
        a = action / self.phi[0]
        
        #Next, update the velocity:
        self.state[1] += a * self.dt
        
        #Then, update the position:
        self.state[0] += self.state[1] * self.dt
        
        #Did we succeed?
        success = np.linalg.norm(self.state - self.goal) < self.done_thresh
        if success:
            print("Success!!!!")
            
        #Did we fail?
        fail = (not success) and (self.timestep == self.H)
        
        #Are we done?
        done = success or fail
            
        #Accrue rewards:
        #fail_val = -np.finfo(np.float32).max  #TODO: Make this a large but not absurd value that upper bounds R1 and R2
        fail_val = -1000.0
        #if fail:
        if False:
            R1 = fail_val
            R2 = fail_val
        else:
            R1 = -np.linalg.norm(self.state - self.goal)
            #if self.state[0] < self.goal[0]:
            #    R1 = self.state[1]
            #else:
            #    R1 = -self.state[1]
            R2 = -np.inner(action, action)
            
        #reward = np.array([R1, R2])
        reward = np.array([R1])
        
        #Note: reward is now 2D!!
        return np.array(self.state), reward, done, {}
     
    def reset(self, random=False):
        if random:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))
        else:
            self.state = np.zeros([2])
            
        self.timestep = 0 #Reset the timestep
        return np.array(self.state)
        
        
if __name__ == '__main__':
    spaceship = SpaceShip()
    
    for i in range(10000):
        state, reward, done, info = spaceship.step(np.random.rand(1))
        print(reward)
        print(state)
        if i % 1000 == 0:
            spaceship.reset()
        
    
