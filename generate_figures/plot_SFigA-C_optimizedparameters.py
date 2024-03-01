# Plot parameters for best models for S1 Appendix Fig A-C

from os.path import join
from get_paths import *

import pickle
import matplotlib.pyplot as plt
import numpy as np
from colors_paultol import colors

save_flag = 0

## Load data
# Load simulation results
fname_data = "data_unified_model_search_best.pickle"
with open(join(dir_data, fname_data), "rb") as f:
    data = pickle.load(f)

particle_best = data["particle_best"]

# Load settings files
fname_settings = {}
fname_settings["D+"] = "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle"
fname_settings["D-"] = "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle"
params = {}
for Dtype in fname_settings:
    with open(join(dir_data_original, fname_settings[Dtype]), "rb") as f:
        params[Dtype] = pickle.load(f)

## Compute difference between parameters normalized by the parameter range
param_rel_diff = np.zeros(len(particle_best[Dtype]))
for ii in range(len(particle_best[Dtype])):
    param_diff = np.abs(particle_best["D+"][ii] - particle_best["D-"][ii])
    param_range = params[Dtype]["Bounder"]["hi"][ii] - params[Dtype]["Bounder"]["lo"][ii]
    param_rel_diff[ii] = param_diff / param_range

print("Average Normalized Parameter Difference: {}".format(param_rel_diff.mean()))

## Plot activation parameters
celltypes_activation = ["L23input", "L23", "I23input", "I23", "L5input", "L5", "I5input", "I5", "L6input", "L6", "I6input", "I6"]
bar_labels = ["L2/3 IT AFF", "L2/3 IT", "L2/3 BC AFF",  "L2/3 BC", "L5 PTN AFF", "L5 PTN", "L5 BC AFF", "L5 BC", "L6 IT AFF", "L6 IT", "L6 BC AFF",  "L6 BC"]
celltype_to_label = {}
for ind, celltype in enumerate(celltypes_activation):
    celltype_to_label[celltype] = bar_labels[ind]

activation_values = {}
for Dtype in particle_best:
    activation_values[Dtype] = []
    for celltype in celltypes_activation:
        for ii in range(len(params[Dtype]["parameters"])):
            if params[Dtype]["parameters"][ii][0].startswith("p_active"):
                if params[Dtype]["parameters"][ii][-1] == celltype:
                    activation_values[Dtype].append(100 * particle_best[Dtype][ii])

bar_labels_r = bar_labels.copy()
bar_labels_r.reverse()
width_bar = 1. / 3
x_bar = np.arange(len(activation_values[Dtype]))
fig_activation_h = plt.figure(figsize=(4, 6))
fig_activation_h.subplots_adjust(left=0.3, right=0.95, bottom=0.15)
_=plt.title("Activation Parameters")
for ind, Dtype in enumerate(["D+", "D-"]):
    _=plt.barh(x_bar + ind * width_bar, np.flipud(activation_values[Dtype]), width_bar, label=Dtype, color=colors["popsmooth"][Dtype])

_=plt.grid(True, axis="x")
_=plt.yticks(x_bar, bar_labels_r, fontsize=12)
_=plt.xlim(0, 100)
_=plt.xlabel("Percent Activated (%)", fontsize=12)
_=plt.legend()

## Plot strength parameters
strength_labels = []
strength_inds = []
for ii in range(len(params[Dtype]["parameters"])):
    if params[Dtype]["parameters"][ii][0] == "strength":
        ptype, pre, post, syntype = params[Dtype]["parameters"][ii]
        label = celltype_to_label[pre] + "-" + celltype_to_label[post] + "-" + syntype
        strength_labels.append(label)
        strength_inds.append(ii)

strength_inds = np.array(strength_inds)

strength_labels_r = strength_labels.copy()
strength_labels_r.reverse()
x_bar = np.arange(len(strength_inds))
fig_strength_ave_h = plt.figure(figsize=(4, 6))
fig_strength_ave_h.subplots_adjust(left=0.65, right=0.85, bottom=0.1, top=0.95)
_=plt.title("Strength Parameters")
for ind, Dtype in enumerate(["D+", "D-"]):
    _=plt.barh(x_bar + ind * width_bar, np.flipud(particle_best[Dtype][strength_inds]), width_bar, label=Dtype, color=colors["popsmooth"][Dtype])

_=plt.grid(True, axis="x")
_=plt.yticks(x_bar, strength_labels_r)
_=plt.xlabel("Strength Factor")
_=plt.legend()

## Plot delay parameters
delay_labels = []
delay_inds = []
for ii in range(len(params[Dtype]["parameters"])):
    if params[Dtype]["parameters"][ii][0].startswith("delay"):
        ptype, pre, post = params[Dtype]["parameters"][ii]
        if "input" in pre:
            continue
        label = celltype_to_label[pre] + "-" + celltype_to_label[post]
        delay_labels.append(label)
        delay_inds.append(ii)

delay_inds = np.array(delay_inds)

delay_labels_r = delay_labels.copy()
delay_labels_r.reverse()
x_bar = np.arange(len(delay_inds))
fig_delay_ave_h = plt.figure(figsize=(4, 6))
fig_delay_ave_h.subplots_adjust(left=0.45, right=0.95, bottom=0.1, top=0.95)
_=plt.title("Conduction Velocity Scalar")
for ind, Dtype in enumerate(["D+", "D-"]):
    _=plt.barh(x_bar + ind * width_bar, np.flipud(particle_best[Dtype][delay_inds]), width_bar, label=Dtype, color=colors["popsmooth"][Dtype])

_=plt.axvline(1, linestyle="--", color="k")
_=plt.grid(True, axis="x")
_=plt.yticks(x_bar, delay_labels_r)
_=plt.xlabel("Velocity Factor")
_=plt.legend()

## Plot afferent delay parameters
afferent_delay_labels = []
afferent_delay_inds = []
for ii in range(len(params[Dtype]["parameters"])):
    if params[Dtype]["parameters"][ii][0] == "delay_afferent":
        ptype, pre, post = params[Dtype]["parameters"][ii]
        label = celltype_to_label[pre] + "-" + celltype_to_label[post]
        afferent_delay_labels.append(label)
        afferent_delay_inds.append(ii)

afferent_delay_inds = np.array(afferent_delay_inds)

afferent_delay_labels_r = afferent_delay_labels.copy()
afferent_delay_labels_r.reverse()
x_bar = np.arange(len(afferent_delay_inds))
fig_afferent_delay_ave_h = plt.figure(figsize=(4, 6))
fig_afferent_delay_ave_h.subplots_adjust(left=0.4, right=0.8, bottom=0.1, top=0.95)
_=plt.title("Afferent Activation Delay Parameters")
for ind, Dtype in enumerate(["D+", "D-"]):
    _=plt.barh(x_bar + ind * width_bar, np.flipud(particle_best[Dtype][afferent_delay_inds] - 1), width_bar, label=Dtype, color=colors["popsmooth"][Dtype])

_=plt.grid(True, axis="x")
_=plt.yticks(x_bar, afferent_delay_labels_r)
_=plt.xlabel("Delay (ms)")
_=plt.legend()

## Plot noise synaptic weight parameters
syn_noise_weight_labels = []
syn_noise_weight_inds = []
for ii in range(len(params[Dtype]["parameters"])):
    if params[Dtype]["parameters"][ii][0] == "syn_noise_weight":
        ptype, post = params[Dtype]["parameters"][ii]
        label = celltype_to_label[post]
        syn_noise_weight_labels.append(label)
        syn_noise_weight_inds.append(ii)

syn_noise_weight_inds = np.array(syn_noise_weight_inds)

syn_noise_weight_labels_r = syn_noise_weight_labels.copy()
syn_noise_weight_labels_r.reverse()
x_bar = np.arange(len(syn_noise_weight_inds))
fig_syn_noise_weight_ave_h = plt.figure(figsize=(4, 6))
fig_syn_noise_weight_ave_h.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.95)
_=plt.title("Syn Noise Weight Parameters")
for ind, Dtype in enumerate(["D+", "D-"]):
    _=plt.barh(x_bar + ind * width_bar, np.flipud(particle_best[Dtype][syn_noise_weight_inds]), width_bar, label=Dtype, color=colors["popsmooth"][Dtype])

_=plt.grid(True, axis="x")
_=plt.yticks(x_bar, syn_noise_weight_labels_r)
_=plt.xlabel("Weight (ms)")
_=plt.legend()

## Plot noise firing rate parmameters
syn_noise_FR_labels = []
syn_noise_FR_inds = []
for ii in range(len(params[Dtype]["parameters"])):
    if params[Dtype]["parameters"][ii][0] == "syn_noise_FR":
        ptype, post = params[Dtype]["parameters"][ii]
        label = celltype_to_label[post]
        syn_noise_FR_labels.append(label)
        syn_noise_FR_inds.append(ii)

syn_noise_FR_inds = np.array(syn_noise_FR_inds)

syn_noise_FR_labels_r = syn_noise_FR_labels.copy()
syn_noise_FR_labels_r.reverse()
x_bar = np.arange(len(syn_noise_FR_inds))
fig_syn_noise_FR_ave_h = plt.figure(figsize=(4, 6))
fig_syn_noise_FR_ave_h.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.95)
_=plt.title("Syn Noise Firing Rate Parameters")
for ind, Dtype in enumerate(["D+", "D-"]):
    _=plt.barh(x_bar + ind * width_bar, np.flipud(particle_best[Dtype][syn_noise_FR_inds]), width_bar, label=Dtype, color=colors["popsmooth"][Dtype])

_=plt.grid(True, axis="x")
_=plt.yticks(x_bar, syn_noise_FR_labels_r)
_=plt.xlabel("Firing Rate Proportion")
_=plt.legend()

if save_flag:
    fig_activation_h.savefig("SFigA_activation.tiff", dpi=600)
    fig_strength_ave_h.savefig("SFigB_strength.tiff", dpi=600)
    fig_delay_ave_h.savefig("SFigB_velocity.tiff", dpi=600)
    fig_afferent_delay_ave_h.savefig("SFigB_affdelay.tiff", dpi=600)
    fig_syn_noise_weight_ave_h.savefig("SFigC_noiseweight.tiff", dpi=600)
    fig_syn_noise_FR_ave_h.savefig("SFigC_noiserate.tiff", dpi=600)

plt.show()