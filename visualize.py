import numpy as np 
import matplotlib.pyplot as plt 

loss = np.load("laser-only/loss.npy")
reward = np.load("laser-only/reward.npy")
print(loss)
print(loss.shape, reward.shape)