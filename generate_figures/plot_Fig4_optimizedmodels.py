# Plots errors, unified model criteria, smoothed population rates, and spike rasters
# used in Fig 4

from os.path import join
from get_paths import *

import pickle
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import objective
from colors_paultol import *

save_flag = 0

## Load data
# Load simulation results
fname_data = "data_unified_model_search_best.pickle"
with open(join(dir_data, fname_data), "rb") as f:
    data = pickle.load(f)

fname_settings = {}
fname_settings["D+"] = "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle"
fname_settings["D-"] = "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle"
params = {}
for Dtype in fname_settings:
    with open(join(dir_data, fname_settings[Dtype]), "rb") as f:
        params[Dtype] = pickle.load(f)

# Organize spike times for plotting
# Get stimulation times
state = "resting"
trial_interval = params["D+"]["trial_interval"][state]
tres = params["D+"]["tres"]
state = "resting"
stim_times = []
for ii in range(params["D+"]["num_trials"]):
    stim_times.append(params["D+"]["stim_delay"] + ii * trial_interval)

cell_order_spikes = ["I6", "L6", "I5", "L5", "I23", "L23"]
spikes_plot = {}
for Dtype in params:
    spikes_plot[Dtype] = {}
    ID_plot = 0
    for celltype in cell_order_spikes:
        spikes_plot[Dtype][celltype] = [[], []]
        for ID in data["spike_times"]["Unified"][Dtype][celltype]:
            trial_spikes = data["spike_times"]["Unified"][Dtype][celltype][ID]
            for stim_time in stim_times:
                idx_spikes = np.all(
                    [
                        trial_spikes >= stim_time ,
                        trial_spikes < stim_time + trial_interval
                    ],
                    axis=0
                )
                spikes_plot[Dtype][celltype][0] += list(trial_spikes[idx_spikes] - stim_time - 1)
                spikes_plot[Dtype][celltype][1] += [ID_plot] * len(trial_spikes[idx_spikes])
            ID_plot += 1
        spikes_plot[Dtype][celltype] = np.array(spikes_plot[Dtype][celltype])

# ID offsets for ytick labels
ID_offsets = []
cell_ylabels = []
for cell in cell_order_spikes:
    ID_max = spikes_plot[Dtype][cell][1].max() + 1
    ID_min = spikes_plot[Dtype][cell][1].min()
    ID_offsets.append(0.5 * (ID_max + ID_min))
    if cell.startswith("I"):
        layer = cell[1:]
        if layer == "23":
            layer = "2/3"
        next = "L" + cell[1:]
        ID_max2 = spikes_plot[Dtype][next][1].max() + 1
        ID_offsets.append(0.5 * (ID_min + ID_max2))
        cell_ylabels.append("BC")
        cell_ylabels.append("Layer {}\n".format(layer))
    else:
        if cell.startswith("L5"):
            cell_ylabels.append("PTN")
        else:
            cell_ylabels.append("IT")

###############################################################
## Create figure for paper with all plots arranged correctly ##
###############################################################
ind_order = [0, 2, 1]
colors_bry = [cset_high_contrast[ind] for ind in ind_order]

weights = data["weights"]#sorted(list(data["relative_error"][Dtype].keys()))
fontsize_label = 10
fontsize_axis = 8
fontsize_legend = 8
fig_size = (6.5, 6)
x_gridlines = np.arange(0, 8, 2)

fig_paper = plt.figure(figsize=fig_size)
fig_paper.subplots_adjust(left=0.1, right=0.98, bottom=0.08, top=0.97, wspace=0.5, hspace=0.8)
gs = fig_paper.add_gridspec(4, 2)

# Plot best model for D+
ax_Dplus = fig_paper.add_subplot(gs[1, 0])
Dtype = "D+"
_=plt.plot(
    data["popsmooth"][Dtype]["t_exp"],
    data["popsmooth"][Dtype]["filtered_exp"],
    color="k",
    linewidth=2,
    label="Experimental"
)
_=plt.plot(
    data["t_sim"],
    data["popsmooth"][Dtype]["simulation"],
    linewidth=2,
    linestyle="--",
    color=colors_bry[0],
    label="Individual/Unified"
)
_=plt.plot(
    data["t_sim"],
    data["popsmooth"]["Unified"][Dtype],
    linewidth=2,
    linestyle="--",
    color=colors_bry[0],
    label="Unified"
)
_=plt.ylabel("Normalized Value", fontsize=fontsize_label)
_=plt.xlabel("Time after Stimulus (ms)", fontsize=fontsize_label)
_=plt.xlim(-0.5, 7.5)
_=plt.ylim(-1.2, 1.7)
_=plt.xticks(x_gridlines, fontsize=fontsize_axis)
_=plt.yticks(fontsize=fontsize_axis)
_=plt.grid(True)

# Plot best unified model for D-
ax_Dminus = fig_paper.add_subplot(gs[1, 1])
Dtype = "D-"
_=plt.plot(
    data["popsmooth"][Dtype]["t_exp"],
    data["popsmooth"][Dtype]["filtered_exp"],
    color="k",
    linewidth=2,
)
_=plt.plot(
    data["t_sim"],
    data["popsmooth"][Dtype]["simulation"],
    linewidth=2,
    linestyle="--",
    color=colors_bry[1],
    label="Individual"
)
_=plt.plot(
    data["t_sim"],
    data["popsmooth"]["Unified"][Dtype],
    linewidth=2,
    linestyle="--",
    color=colors_bry[2],
    label="Unified"
)
_=plt.ylabel("Normalized Value", fontsize=fontsize_label)
_=plt.xlabel("Time after Stimulus (ms)", fontsize=fontsize_label)
_=plt.xlim(-0.5, 7.5)
_=plt.ylim(-1.2, 1.7)
_=plt.xticks(x_gridlines, fontsize=fontsize_axis)
_=plt.yticks(fontsize=fontsize_axis)
_=plt.grid(True)

# Plot D+ spike raster
ax_spikes_Dplus = fig_paper.add_subplot(gs[2:, 0])
Dtype = "D+"
for celltype in cell_order_spikes:
    _=plt.scatter(
        spikes_plot[Dtype][celltype][0],
        spikes_plot[Dtype][celltype][1],
        0.5,
        color=colors["spikes"][celltype]
    )
    if celltype.startswith("I"):
        line = spikes_plot[Dtype][celltype][1].min()
        _=plt.axhline(line, linewidth=0.5, linestyle="--", color="k", alpha=0.5)

_=plt.ylim(-0.5, 389.5)
_=plt.xlim(-0.5, 7.5)
_=plt.xticks(x_gridlines, fontsize=fontsize_axis)
_=plt.yticks(ID_offsets, cell_ylabels, fontsize=fontsize_label, rotation=90, va="center")
_=plt.xlabel("Time after Stimulus (ms)", fontsize=fontsize_label)
ax_spikes_Dplus.tick_params(axis="y", which="both", length=0)

# Plot D- unified spike raster
ax_spikes_Dminus = fig_paper.add_subplot(gs[2:, 1])
ave_type = "similar"
Dtype = "D-"
weight = 1
for celltype in cell_order_spikes:
    _=plt.scatter(
        spikes_plot[Dtype][celltype][0],
        spikes_plot[Dtype][celltype][1],
        0.5,
        color=colors["spikes"][celltype]
    )
    if celltype.startswith("I"):
        line = spikes_plot[Dtype][celltype][1].min()
        _=plt.axhline(line, linewidth=0.5, linestyle="--", color="k", alpha=0.5)

_=plt.ylim(-0.5, 389.5)
_=plt.xlim(-0.5, 7.5)
_=plt.xticks(x_gridlines, fontsize=fontsize_axis)
_=plt.yticks(ID_offsets, cell_ylabels, fontsize=fontsize_label, rotation=90, va="center")
_=plt.xlabel("Time after Stimulus (ms)", fontsize=fontsize_label)
ax_spikes_Dminus.tick_params(axis="y", which="both", length=0)

# Plot unified model search
ax_unified_cost = fig_paper.add_subplot(gs[0, 1])
_=plt.plot(weights, data["norm_total"], linewidth=2, label="Total Error", color=colors_bry[0])
_=plt.plot(weights, data["norm_diff"], linewidth=2, label="Error Difference", color=colors_bry[1])
_=plt.plot(weights, data["norm_cost"], linewidth=2, label="Cost", color=colors_bry[2])

_=plt.legend(prop={"family": "Georgia", "size": fontsize_legend})
_=plt.grid(True)
_=plt.xlabel("Weight", fontsize=fontsize_label)
_=plt.ylabel("Normalized Value", fontsize=fontsize_label)
_=plt.xticks(fontsize=fontsize_axis)
_=plt.yticks(fontsize=fontsize_axis)

# Plot I-wave errors
ax_spikes_Dplus = fig_paper.add_subplot(gs[0, 0])
objectives = [obj for obj in data["relative_error"][Dtype] if "iwave-" in obj]
objective_labels = ["D-Max", "I2-Max", "I3-Max", "D-Min", "I1-Min", "I2-Min", "I3-Min", "I2-TTP", "I3-TTP", "Ave"]

error_plot = {}
error_plot["D+"] = []
error_plot["D-"] = []
error_plot["Unified"] = []
for obj in objectives:
    error_plot["D+"].append(100 * data["relative_error"]["D+"][obj])
    error_plot["D-"].append(100 * data["relative_error"]["D-"][obj])
    error_plot["Unified"].append(100 * data["relative_error"]["Unified"][obj])

error_plot["D+"].append(np.mean(error_plot["D+"]))
error_plot["D-"].append(np.mean(error_plot["D-"]))
error_plot["Unified"].append(np.mean(error_plot["Unified"]))

objective_labels_r = objective_labels.copy()
objective_labels_r.reverse()
width_bar = 0.25
x_bar = np.arange(len(error_plot[Dtype]))
for ind, Dtype in enumerate(error_plot):
    _=plt.bar(x_bar + ind * width_bar, error_plot[Dtype], width_bar, label=Dtype, color=colors_bry[ind])

_=plt.grid(True, axis="y")
_=plt.xticks(x_bar + width_bar, objective_labels, fontsize=fontsize_axis, rotation=90)
_=plt.yticks(fontsize=fontsize_axis)
_=plt.ylim(0, 50)
_=plt.ylabel("Relative Error (%)", fontsize=fontsize_label)

if save_flag:
    fig_paper.savefig("Fig4_optimizedmodels.tiff", dpi=600)

plt.show()
