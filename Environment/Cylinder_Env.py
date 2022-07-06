import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math

import torch
import random
import math
import torch.nn as nn

from CFD import *
from Cylinder_Rotation_Env_utils import *
import time

class Cylinder_Rotation_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, params=None):
        self.set_params(params)

        self.action_space = spaces.Box(
            low= self.params['min_w'],
            high=self.params['max_w'],
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low= self.params['min_velocity'],
            high=self.params['max_velocity'],
            shape=(self.params['dimx'], self.params['dimy']),
            dtype=np.float32
        )
        self.sim = Cylinder_Rotation_Sim(self.params)
        self.sim.generate_grid()
        self.sim.cal_mask()
        self.current_t = 0
        
    def set_params(self, params =None):
        if params is not None:
            self.params = params
        else:
            self.params = {'dtr': 0.1,
                            'T': 1,
                            'rho_0': 1,
                            'mu' : 1/1000,
                            'traj_max_T': 20,
                            'dimx': 100,
                            'dimy': 100,
                            'min_w': -1,
                            'max_w': 1,
                            'min_velocity': -1,
                            'max_velocity': 1,
                            'min_x' : 0, 
                            'max_x' : 2.2, 
                            'min_y' : 0, 
                            'max_y' : 0.41, 
                            'r' : 0.05, 
                            'center':(0.2, 0.2),
                            'U_max': 1.5,                          
                                    }
        
    def step(self, action):
        time0 = time.time()
        self.sim.do_simulation(action)
        print(time.time()- time0)
        reward = self._get_reward()
        print(time.time()- time0)
        ob = self.sim.get_observation('grid')
        print(time.time()- time0)
        # episode_over = self._get_done()
        episode_over = False
        return ob, reward, episode_over, {}

    def reset(self):
        self.sim.set_state_vector(self.sim.init_state_1_vector.vector())
        self.current_t = 0
        return self.sim.get_observation('grid')        

    def _render(self, mode='grid', obj='u', close=False):
        if mode == 'grid':
            if obj  == 'u':
                pass
                
        elif mode == 'node':
            pass
 
    def _get_reward(self):
        C_D, C_L, p_diff = self.sim.postprocess(self.sim.solver.sol)
        return -C_D - C_L  
        
    def get_down(self):
        pass
        # return self.env.getDown()
 