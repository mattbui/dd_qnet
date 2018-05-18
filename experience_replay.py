from __future__ import print_function, division

import numpy as np
import random

class ExperienceReplay:
    """
    Class for storing experience
    1 experience is an array of [state, action, reward, done, newstate]
    """
    def __init__(self, output_dir, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.output_dir = output_dir
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
            
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, min(size, len(self.buffer)))), [size, 5])

    def save(self):
        np.save(self.output_dir + "/experience", self.buffer)

    def load(self, path):
        self.buffer = np.load(path + "/experience.npy").tolist()