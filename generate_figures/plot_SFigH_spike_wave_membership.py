# Plot probability of contributing to a corticospinal wave for S1 Appendix Fig H

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

spike_times = data["spike_times"]

# Load settings files
fname_settings = {}
fname_settings["D+"] = "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle"
fname_settings["D-"] = "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle"
params = {}
for Dtype in fname_settings:
    with open(join(dir_data, fname_settings[Dtype]), "rb") as f:
        params[Dtype] = pickle.load(f)

# Get stimulation times
state = "resting"
num_trials = params[Dtype]["num_trials"]
trial_interval = params["D+"]["trial_interval"][state]
tres = params["D+"]["tres"]
state = "resting"
stim_times = []
for ii in range(num_trials):
    stim_times.append(params["D+"]["stim_delay"] + ii * trial_interval)

# Sort spikes for L5 PTN by trial
celltype = "L5"
spikes = {}
for Dtype in params:
    spikes[Dtype] = [ [[], []] for ii in range(num_trials) ]
    for trial_ind, stim_time in enumerate(stim_times):
        ID_plot = 0
        for ID in data["spike_times"]["Unified"][Dtype][celltype]:
            trial_spikes = data["spike_times"]["Unified"][Dtype][celltype][ID]
            idx_spikes = np.all(
                [
                    trial_spikes >= stim_time ,
                    trial_spikes < stim_time + trial_interval
                ],
                axis=0
            )
            spikes[Dtype][trial_ind][0] += list(trial_spikes[idx_spikes] - stim_time - 1)
            spikes[Dtype][trial_ind][1] += [ID_plot] * len(trial_spikes[idx_spikes])
            ID_plot += 1
        
        spikes[Dtype][trial_ind] = np.array(spikes[Dtype][trial_ind])

## Calculate wave membership
# Define boundaries for spikes
wave_bounds = {}
wave_bounds["D+"] = {}
wave_bounds["D+"]["D"] = (0.9, 1.5)
wave_bounds["D+"]["I1"] = (2.0, 3.0)
wave_bounds["D+"]["I2"] = (3.2, 4.6)
wave_bounds["D+"]["I3"] = (4.6, 6.5)
wave_bounds["D-"] = {}
wave_bounds["D-"]["I1"] = (2.0, 3.0)
wave_bounds["D-"]["I2"] = (3.2, 4.6)
wave_bounds["D-"]["I3"] = (4.6, 6.5)

waves = {}
for Dtype in wave_bounds:
    waves[Dtype] = list(wave_bounds[Dtype].keys())

spikes_per_wave = {}
spikes_per_wave_per_cell = {}
wave_participation  = {}
for Dtype in spikes:
    # Number of spikes per wave per neuron
    spikes_per_wave[Dtype] = [{} for _ in range(num_trials)]
    spikes_per_wave_per_cell[Dtype] = [{} for _ in range(num_trials)]
    for trial in range(num_trials):
        for ind, wave in enumerate(wave_bounds[Dtype]):
            idx = np.all([
                spikes[Dtype][trial][0] >= wave_bounds[Dtype][wave][0],
                spikes[Dtype][trial][0] < wave_bounds[Dtype][wave][1]
            ], axis=0)
            IDs = spikes[Dtype][trial][1][idx]
            spikes_per_wave[Dtype][trial][wave] = IDs.size
            for ID in IDs:
                if ID not in spikes_per_wave_per_cell[Dtype] [trial]:
                    spikes_per_wave_per_cell[Dtype][trial][ID] = np.zeros(4)

                ID_count = (IDs == ID).sum()
                spikes_per_wave_per_cell[Dtype][trial][ID][ind] += ID_count

    # Wave participation (number of waves during which any spiking occurred)
    wave_participation[Dtype] = np.zeros(5)
    wave_participation[Dtype][0] = params[Dtype]["N"][celltype] - len(spikes_per_wave_per_cell[Dtype])
    for trial in range(num_trials):
        for ID in spikes_per_wave_per_cell[Dtype][trial]:
            ind = (spikes_per_wave_per_cell[Dtype][trial][ID] > 0).sum()
            wave_participation[Dtype][ind] += 1

    wave_participation[Dtype] /= num_trials

# Plot wave participation
fig_wave_participation = plt.figure(figsize=(4, 4))
fig_wave_participation.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.97)
x_bar = np.arange(5)
width = 0.33
for ind, Dtype in enumerate(wave_participation):
    plt.bar(
        x_bar + ind * width, 
        wave_participation[Dtype] / wave_participation[Dtype].sum(),
        width, 
        color=colors["popsmooth"][Dtype],
        label=Dtype,
    )
plt.xlabel("Number of Waves")
plt.grid(True, axis="y")
plt.ylabel("Proportion")
plt.legend()

if save_flag:
    fig_wave_participation.savefig("SFigH_wave_participation.tiff", dpi=600)

plt.show()