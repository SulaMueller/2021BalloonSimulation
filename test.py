
import matplotlib.pyplot as plt
import numpy as np
import pylops
import pylops.optimization



t = np.arange(63) * 0.004
h, th, hcenter = pylops.utils.wavelets.ricker(t, f0=20)
print('Done.')