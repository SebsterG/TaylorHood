import numpy as np
import matplotlib.pyplot as plt


#E_k = np.loadtxt(open("kinetic_reference2016.4.26.18.txt"))
#dt = 20.0/len(E_k)
#dkdt_oasis = -(E_k[1:]-E_k[:-1])/dt
e_k = np.loadtxt(open("results/IPCS/e_k_10.txt"))
dkdt = np.loadtxt(open("results/IPCS/dKdt_10.txt"))
time = np.loadtxt(open("results/IPCS/time.txt"))


#time1 = len(dkdt_oasis)
#time2 = 20.0/len(dkdt)



#dkdt = np.loadtxt(open("data_dkdt_2.txt"))
#time = np.loadtxt(open("data_time_2.txt"))
#plt.plot(E_k, label = "kinetic energy")
plt.figure(1)
plt.plot(dkdt,"r", label = "dkdt")
plt.figure(2)
plt.plot(e_k,"b",label="dkdt_oasis")
plt.show()
