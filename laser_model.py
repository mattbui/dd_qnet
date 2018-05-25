from __future__ import print_function, division

import numpy as np
import keras
from keras.models import Model, load_model
from keras import optimizers
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Conv2D, MaxPool2D, Flatten, Lambda, Concatenate
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import sys
import os
import time
import random

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

class Qnet:

    def __init__(self, numstate, num_actions):
        self.input_size = numstate
        self.output_size = num_actions
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):

        inp = Input(shape=(self.input_size[0], ))
        inner = Dense(300, activation='relu')(inp)
        inner = Dense(300, activation='relu')(inner)
        out = Dense(self.output_size + 1)(inner)

        # out = value + avantage - mean(advantage)
        out = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True, axis=1))(out)

        model = Model(inp, out)
        
        model.summary()

        optimizer = optimizers.Adam(0.0001)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def get_qvalues(self, states):
        predicted = self.model.predict(states)
        return predicted

    def get_target_qvalues(self, states):
        predicted = self.target_model.predict(states)
        return predicted

    def get_actions(self, states):
        qvalues = self.get_qvalues(states)
        actions = np.argmax(qvalues, axis=1)
        return actions

    def update_target_model(self, tau):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i, layer_weights in enumerate(main_weights):
            target_weights[i] *= (1-tau)
            target_weights[i] += tau * layer_weights
        self.target_model.set_weights(target_weights)

    def save(self, path):
        self.target_model.save(path + "/target.h5")
        self.model.save(path + "/main.h5")
    
    def load(self, path):
        self.target_model.load_weights(path + "/target.h5")
        self.model.load_weights(path + "/main.h5")

    def learn_on_minibatch(self, minibatch, gamma):
        states = np.vstack(minibatch[:,0])
        actions = minibatch[:, 1]
        rewards = minibatch[:, 2]
        dones = minibatch[:, 3]
        newstates = np.vstack(minibatch[:, 4])

        actions_newstate = self.get_actions(newstates)
        target_qvalues_newstate = self.get_target_qvalues(newstates)
        double_q = target_qvalues_newstate[range(target_qvalues_newstate.shape[0]), actions_newstate]

        done_multiplier = 1 - dones
        target_q = rewards + gamma * double_q * done_multiplier
        
        qvalues = self.get_qvalues(states)
        for i in range(qvalues.shape[0]):
            qvalues[i, actions[i]] = target_q[i]

        loss = self.model.train_on_batch(states, qvalues)
        return loss