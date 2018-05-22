from __future__ import division, print_function

import gym
import gym_gazebo
import numpy as np
import sys
import os
import time
from ddq_model import Qnet
from experience_replay import ExperienceReplay
from utils import Config

if(len(sys.argv) < 3):
    print("Missing fields")
    print("running command: python test_laser_only.py <from_pretrain_dir> <epsilon>")

else:
    env = gym.make('GazeboTurtlebotMazeColor-v0')
    qnet = Qnet(env.num_state, env.num_action)

    from_pretrain = sys.argv[1]
    epsilon = float(sys.argv[2])
    qnet.load(from_pretrain)

    while True:
        state = env.reset()
        total_reward = 0
        num_step = 0
        num_random_step = 0
        for i in range(1000):
            num_step += 1
            # get action
            if(np.random.rand(1) < epsilon):
                action = np.random.randint(env.num_action)
                num_random_step += 1
            else:
                action = qnet.get_actions(state)[0]

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

        print("\nDone episode in {} steps, {} random steps, Total reward: {}".format(num_step, num_random_step, total_reward))
