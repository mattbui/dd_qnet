from __future__ import division, print_function

import gym
import gym_gazebo
import numpy as np
import sys
import os
import time
#from ddq_model import Qnet
from laser_model import Qnet
from experience_replay import ExperienceReplay
from utils import Config

env = gym.make('GazeboTurtlebotMazeColor-v0')
qnet = Qnet(env.num_state, env.num_action)

from_pretrain = sys.argv[1]

if(from_pretrain != None):
    qnet.load(from_pretrain)
else:
    print("Missing from pretrain field")

while True:
    state = env.reset()
    total_reward = 0
    num_step = 0
    for i in range(1000):
        num_step += 1
        # get action
        action = qnet.get_actions(state.reshape(1, -1))[0]

        # get after take action
        newstate, reward, done, _ = env.step(action)
        if(newstate == []):
            print("Terminate")
            # state = env.reset()
            break

        total_reward += reward
        state = newstate
        if done:
            break

    print("\nDone episode in {} steps, Total reward: {}".format(num_step, total_reward))