import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class FooEnv(gym.Env):
    def __init__(self):
        
        self.action_space = spaces.Box(
            low= -1 ,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low= -5 ,
            high=5,
            shape=(1,),
            dtype=np.float32)
    
    def step(self, action):
        pass
    def reset(self):
        pass
    def render(self, mode='human', close=False):
        pass