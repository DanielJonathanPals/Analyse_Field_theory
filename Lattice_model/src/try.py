import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


q = np.reshape(np.arange(0,144,1), (12,12))
length_scale = 4
kernel = np.ones((length_scale, length_scale)) / (length_scale ** 2)
c = convolve2d(q, kernel, mode='valid')
print(c)
print(c[::4,::4])