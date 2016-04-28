import numpy as np
import matplotlib.pyplot as plt


dkdt = np.loadtxt(open("data_dkdt.txt"))
time = np.loadtxt(open("data_time.txt"))

#dkdt = np.loadtxt(open("data_dkdt_2.txt"))
#time = np.loadtxt(open("data_time_2.txt"))

plt.plot(-dkdt)
plt.show()
