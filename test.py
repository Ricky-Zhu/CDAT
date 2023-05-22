import gym
from modified_envs import *

env = gym.make('HalfCheetah_3leg-v2')
env.reset()
for i in range(1000):
    env.step(env.action_space.sample())
    env.render()