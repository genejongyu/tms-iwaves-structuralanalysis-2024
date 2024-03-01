# Plots smoothed population rates related to preferential parameters for Fig 6

from os.path import join
from get_paths import *

import pickle
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import h5py
from colors_paultol import colors

save_flag = 0

# Load simulation results
fname_simulations = "data_TVAT_waveforms_examples.pickle"
with open(join(dir_data, fname_simulations), "rb") as f:
    popsmooths = pickle.load(f)

n_samples = 275
t = 0.025 * np.arange(n_samples) - 1.125

ymax = 2.5
ymin = -1.5

colors_custom = {}
colors_custom["L5"] = colors["waves"]["D"]
colors_custom["L5input"] = colors["waves"]["I1"]
colors_custom["L23"] = colors["waves"]["I2"]
colors_custom["L6input"] = colors["waves"]["I3"]

figs = {}
for parameter in popsmooths:
    figs[parameter] = plt.figure(figsize=(4, 3))
    figs[parameter].subplots_adjust(left=0.1, right=0.85, bottom=0.17, top=0.9)
    ax1 = figs[parameter].add_subplot(111)
    _=plt.title(parameter)
    for ii in range(len(popsmooths[parameter]["nonzero"])):
        _=plt.plot(t, popsmooths[parameter]["nonzero"][ii], color=colors_custom[parameter], linewidth=1, alpha=0.01)

    for ii in range(len(popsmooths[parameter]["zero"])):
        _=plt.plot(t, popsmooths[parameter]["zero"][ii], color="k", linewidth=1, alpha=0.01)

    _=plt.ylabel("Normalized Values", color=colors_custom[parameter], fontsize=16)
    _=plt.yticks([])
    _=plt.ylim(-1.5, 3)
    _=plt.xlim(t[0], t[-1])
    _=plt.grid()
    _=plt.xlabel("Time After Stimulus (ms)", fontsize=16)

if save_flag:
    for parameter in figs:
        figs[parameter].savefig("Fig6_preferentialeffectsize_popsmooth_{}.tiff".format(parameter), dpi=600)

plt.show()