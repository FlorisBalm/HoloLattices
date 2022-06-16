import numpy as np
import scipy as sp
import h5py
import matplotlib.pyplot as plt
import yaml
import pandas as pd

rho_datas = []
constant_dicts = []

with h5py.File("data/OutputDataTest.h5", "r") as infile:
    # You can loop over all the groups
    for group in infile.keys():
        grp = infile[group]
        rho = grp["rho"][:]
        rho_datas.append(rho)
        
        constants = grp["constants"][()]
        constant_dicts.append(yaml.load(constants, Loader=yaml.Loader))

constants_dF = pd.DataFrame(constant_dicts)
rho_array  = np.array(rho_datas)


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
for i in constants_dF.index:    
    ax.plot(rho_array[i], label=f"dataset {i}")
ax.legend()
plt.show()
