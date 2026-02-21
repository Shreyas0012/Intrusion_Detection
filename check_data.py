import numpy as np
import os
import matplotlib.pyplot as plt

path = "dataset/normal"
files = os.listdir(path)

sample = np.load(os.path.join(path, files[0]))

print("Shape of one sample:", sample.shape)