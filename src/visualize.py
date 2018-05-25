"""
Plot reward and losses
"""

import numpy as np 
import matplotlib.pyplot as plt 

loss = np.load("output/laser-only/loss.npy")
reward = np.load("output/laser-only/reward.npy")
print(loss)
print(loss.shape, reward.shape)