# Plots neuron positions for Fig 3B

from os.path import join
from get_paths import *

import matplotlib.pyplot as plt
from colors_paultol import colors
import pickle
plt.rcParams["mathtext.default"] = "regular"

save_flag = 0

# Load data
fpath = join(dir_data, "data_network_locations.pickle")
with open(fpath, "rb") as f:
    locations = pickle.load(f)

# Create neuron labels
labels = {}
labels["L23"] = "L2/3 IT"
labels["I23"] = "L2/3 BC"
labels["L5"] = "L5 PTN"
labels["I5"] = "L5 BC"
labels["L6"] = "L6 IT"
labels["I6"] = "L6 BC"

# Plot column
fig_column = plt.figure(figsize=(5, 5))
ax_column = fig_column.add_subplot(projection="3d")
ax_column.set_box_aspect([1.0, 1.0, 1.0])
for ind, celltype in enumerate(locations["x"]):
    if celltype.startswith("L"):
        marker = "^"
    else:
        marker = "o"
    if "input" not in celltype:
        _=ax_column.scatter(
            locations["x"][celltype], 
            locations["y"][celltype], 
            -locations["z"][celltype], 
            s=10, 
            color=colors["spikes"][celltype],
            marker=marker,
            label=labels[celltype]
        )

plt.legend(loc=1, markerscale=2, prop={'size': 10})
ax_column.set_xlim(-300, 300)
ax_column.set_ylim(-300, 300)
ax_column.view_init(22, 27)
ax_column.set_xlabel("Mediolateral ($\mu$m)", fontsize=16)
ax_column.set_ylabel("Anterior-Posterior ($\mu$m)", fontsize=16)
ax_column.set_zlabel("\nCortical Depth ($\mu$m)", fontsize=16)

# Plot birds-eye-view of column
fig_column2 = plt.figure(figsize=(5, 5))
ax_column2 = fig_column2.add_subplot(projection="3d")
ax_column2.set_box_aspect([1.0, 1.0, 1.0])
for ind, celltype in enumerate(locations["x"]):
    if celltype.startswith("L"):
        marker = "^"
    else:
        marker = "o"
    if "input" not in celltype:
        _=ax_column2.scatter(
            locations["x"][celltype], 
            locations["y"][celltype], 
            -locations["z"][celltype], 
            s=10, 
            color=colors["spikes"][celltype],
            marker=marker,
        )

ax_column2.set_xlabel("\nMediolateral ($\mu$m)", fontsize=16)
ax_column2.set_ylabel("\nAnterior-Posterior ($\mu$m)", fontsize=16)
ax_column2.view_init(89, 0)
ax_column2.set_xlim(-300, 300)
ax_column2.set_ylim(-300, 300)
ax_column2.set_zticks([])

if save_flag:
    fig_column.savefig("Fig3B_column_3D.tiff", dpi=600)
    fig_column2.savefig("Fig3B_column_birdseye.tiff", dpi=600)

plt.show()