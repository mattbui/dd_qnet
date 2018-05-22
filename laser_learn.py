from __future__ import division, print_function

import gym
import gym_gazebo
import numpy as np
import sys
import os
import time
from laser_model import Qnet
from experience_replay import ExperienceReplay
from utils import Config

argv = sys.argv[1:]
config = Config(argv)
env = gym.make('GazeboTurtlebotMazeColor-v0')
replay = ExperienceReplay(config.args.output_dir ,config.args.replay_buffer_size)
qnet = Qnet(env.num_state, env.num_action)

if(config.args.continue_from != None):
    qnet.load(config.args.continue_from)
    replay.load(config.args.continue_from)

elif(config.args.from_pretrain != None):
    qnet.load(config.args.from_pretrain)

epsilon_decay = (config.args.start_epsilon - config.args.end_epsilon)/config.args.annealing_steps

while config.episode <= config.episode:
    state = env.reset()
    replay_ep = ExperienceReplay(config.args.output_dir ,config.args.replay_buffer_size)
    total_reward = 0
    num_random_step = 0
    total_loss = 0
    num_training = 0
    start_step = config.total_step
    
    if(config.total_step >= config.args.num_pretrain_step):
        config.episode += 1
    
    for i in range(config.args.num_training_step):

        # get action
        if(config.total_step < config.args.num_pretrain_step or np.random.rand(1) < config.epsilon):
            action = np.random.randint(env.num_action)
            num_random_step += 1

        else:
            action = qnet.get_actions(state.reshape(1, -1))[0]
    
        # get after take action
        newstate, reward, done, _ = env.step(action)
        if(newstate == []):
            print("Terminate")
            # state = env.reset()
            break
        replay_ep.add(np.reshape(np.array([state, action, reward, done, newstate]), [1, 5]))
        # train
        if config.total_step > config.args.num_pretrain_step:
            if config.epsilon > config.args.end_epsilon:
                config.epsilon -= epsilon_decay

            if config.total_step % config.args.online_update_freq == 0:
                train_batch = replay.sample(config.args.batch_size)
                loss = qnet.learn_on_minibatch(train_batch, config.args.gamma)
                total_loss += loss
                num_training += 1
                sys.stdout.write("\rTrain step at {}th step | loss {} | epsilon {}".format(config.total_step, loss, config.epsilon))
                sys.stdout.flush()
            
            if config.total_step % config.args.target_update_freq == 0:

                # print("Update target net")
                qnet.update_target_model(config.args.tau)
        
        config.total_step += 1
        total_reward += reward
        state = newstate
        if done:
            break

    replay.add(replay_ep.buffer)
    if(num_training == 0):
        num_training = 1
    print("\nDone epoch in {} steps, {} random steps, Total reward: {}, Total step: {}, Average loss: {}".format(config.total_step - start_step, num_random_step, total_reward, config.total_step, total_loss/num_training))

    if(config.episode % config.args.save_model_freq == 0 and config.total_step > config.args.num_pretrain_step):
        qnet.save(config.args.output_dir)
        config.save()
        replay.save()
        print("Save model at {}".format(config.args.output_dir))
