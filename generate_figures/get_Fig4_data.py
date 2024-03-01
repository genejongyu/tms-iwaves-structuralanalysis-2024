# Identify best unified model and save data from larger unified model search

from os.path import join
from get_paths import *

import pickle
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import objective

## Load data
# Load simulation results
fname = "data_unified_model_search.pickle"
with open(join(dir_data, fname), "rb") as f:
    results = pickle.load(f)

# Load settings files
fname_settings = {}
fname_settings["D+"] = "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle"
fname_settings["D-"] = "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle"
params = {}
for Dtype in fname_settings:
    with open(join(dir_data, fname_settings[Dtype]), "rb") as f:
        params[Dtype] = pickle.load(f)

# Load experimental data
data_exp = {}
data_exp["D-"] = np.load(join(dir_data, "data_experimental_multiamp_2020_43.npy"))
data_exp["D+"] = np.load(join(dir_data, "data_experimental_multiamp_2013_31.npy"))

t_dwave_exp = {}
t_dwave_exp["D+"] = 4.1
t_dwave_exp["D-"] = 4.2

Fs = {}
Fs["D+"] = 10000
Fs["D-"] = 10000

# Get experimental waveforms
t_exp = {}
popsmooth_exp = {}
filtered_exp = {}
for response_type in data_exp:
    t_exp[response_type] = data_exp[response_type][-1]
    popsmooth_exp[response_type] = data_exp[response_type][:-1]
    filtered_exp[response_type] = np.zeros(popsmooth_exp[response_type].shape)
    for ii in range(len(filtered_exp[response_type])):
        filtered_exp[response_type][ii] = objective.butter_bandpass_filter(
            popsmooth_exp[response_type][ii], 200, 2000, Fs[response_type]
        )

ind_exp = {}
ind_exp["D+"] = 1
ind_exp["D-"] = 1

## Compute popsmooths for all cell layers
# Calculate stimulation times
state = "resting"
stim_times = []
for ii in range(params["D+"]["num_trials"]):
    stim_times.append(params["D+"]["stim_delay"] + ii * params["D+"]["trial_interval"][state])

# Get spikes
print("Getting spikes")
trial_interval = params["D+"]["trial_interval"][state]
tres = params["D+"]["tres"]
poprates_all = {}
popsmooths_all = {}
spike_times = {}
cell_order_spikes = ["I6", "L6", "I5", "L5", "I23", "L23"]
for ave_type in results:
    poprates_all[ave_type] = {}
    popsmooths_all[ave_type] = {}
    spike_times[ave_type] = {}
    for Dtype in results[ave_type][0]:
        poprates_all[ave_type][Dtype] = {}
        popsmooths_all[ave_type][Dtype] = {}
        spike_times[ave_type][Dtype] = {}
        for weight in results[ave_type]:
            poprates_all[ave_type][Dtype][weight] = {}
            popsmooths_all[ave_type][Dtype][weight] = {}
            spike_times[ave_type][Dtype][weight] = {}
            ID_plot = 0
            for celltype in cell_order_spikes:
                poprates_all[ave_type][Dtype][weight][celltype] = np.zeros(int(trial_interval / tres))
                spike_times[ave_type][Dtype][weight][celltype] = [[], []]
                for ID in results[ave_type][weight][Dtype]["spikes"][state][celltype]:
                    trial_spikes = results[ave_type][weight][Dtype]["spikes"][state][celltype][ID]
                    for stim_time in stim_times:
                        idx_spikes = np.all(
                            [
                                trial_spikes >= stim_time ,
                                trial_spikes < stim_time + trial_interval
                            ],
                            axis=0
                        )
                        ind_spikes = ((trial_spikes[idx_spikes] - stim_time) / tres).astype(int)
                        spike_times[ave_type][Dtype][weight][celltype][0] += list(trial_spikes[idx_spikes] - stim_time - 1)
                        spike_times[ave_type][Dtype][weight][celltype][1] += [ID_plot] * len(trial_spikes[idx_spikes])
                        for ind in ind_spikes:
                            poprates_all[ave_type][Dtype][weight][celltype][ind] += 1
                    ID_plot += 1
                popsmooths_all[ave_type][Dtype][weight][celltype] = objective.butter_bandpass_filter(
                    poprates_all[ave_type][Dtype][weight][celltype], 
                    params[Dtype]["bandpass_cutoff_lo"],
                    params[Dtype]["bandpass_cutoff_hi"],
                    int(1000 / tres)
                )
                spike_times[ave_type][Dtype][weight][celltype] = np.array(spike_times[ave_type][Dtype][weight][celltype])

# ID offsets for ytick labels
ID_offsets = []
cell_ylabels = []
for cell in cell_order_spikes:
    ID_max = spike_times[ave_type][Dtype][weight][cell][1].max() + 1
    ID_min = spike_times[ave_type][Dtype][weight][cell][1].min()
    ID_offsets.append(0.5 * (ID_max + ID_min))
    if cell.startswith("I"):
        layer = cell[1:]
        if layer == "23":
            layer = "2/3"
        next = "L" + cell[1:]
        ID_max2 = spike_times[ave_type][Dtype][weight][next][1].max() + 1
        ID_offsets.append(0.5 * (ID_min + ID_max2))
        cell_ylabels.append("BC")
        cell_ylabels.append("Layer {}\n".format(layer))
    else:
        if cell.startswith("L5"):
            cell_ylabels.append("PTN")
        else:
            cell_ylabels.append("IT")

# Measure features
print("Measuring features")
props = {}
for ave_type in results:
    props[ave_type] = {}
    for Dtype in params:
        props[ave_type][Dtype] = {}
        for weight in results[ave_type]:
            props[ave_type][Dtype][weight] = {}
            for obj in params[Dtype]["objective_names"]:
                if obj in ["peaks_iwave", "tpeaks_iwave", "minima_iwave"]:
                    rows, cols = params[Dtype]["targets"][obj].shape
                    cols += params[Dtype]["extra_waves"]
                    props[ave_type][Dtype][weight][obj] = np.ones((rows, cols))
                else:
                    props[ave_type][Dtype][weight][obj] = np.ones(params[Dtype]["targets"][obj].shape)
            
            props[ave_type][Dtype][weight]["amp_I1"] = np.ones(params[Dtype]["num_subjects"])
            
            for func_obj in params[Dtype]["functions_obj"]:
                props[ave_type][Dtype][weight] = func_obj(results[ave_type][weight][Dtype], params[Dtype], props[ave_type][Dtype][weight])

# Calculate errors
print("Calculating errors")
cell_order = ["L2/3 PC", "L5 PC", "L6 PC", "L2/3 INT", "L5 INT", "L6 INT"]
ind_subj = 0
relative_error = {}
for ave_type in props:
    relative_error[ave_type] = {}
    for Dtype in props[ave_type]:
        relative_error[ave_type][Dtype] = {}
        for weight in props[ave_type][Dtype]:
            relative_error[ave_type][Dtype][weight] = {}
            for obj in params[Dtype]["objective_names"]:
                if obj in params[Dtype]["targets"]:
                    if obj.endswith("iwave"):
                        if len(params[Dtype]["targets"][obj].shape) > 1:
                            property = props[ave_type][Dtype][weight][obj][ind_subj].copy()
                            ind_start = 0
                        
                            # Normalize peaks before computing relative error
                            if obj in ["peaks_iwave", "minima_iwave"]:
                                property /= props[ave_type][Dtype][weight]["peaks_iwave"][ind_subj][params[Dtype]["ind_I1"]]
                        
                            # Compute relative error
                            num_targets = params[Dtype]["targets"][obj].shape[1]
                            for ind_wave in range(ind_start, num_targets):
                                if obj == "peaks_iwave":
                                    if ind_wave == 1:
                                        continue
                                if obj == "tpeaks_iwave":
                                    if "2013_31" in params[Dtype]["fname_root"]:
                                        if ind_wave == 1:
                                            continue
                                    elif "2020_43" in params[Dtype]["fname_root"]:
                                        if ind_wave < 2:
                                            continue
                                if obj == "tminima_iwave":
                                    if "2013_31" in params[Dtype]["fname_root"]:
                                        if ind_wave == 0:
                                            continue
                                    elif "2020_43" in params[Dtype]["fname_root"]:
                                        if ind_wave < 1:
                                            continue
                            
                                if ind_wave == 0:
                                    wave = "D"
                                else:
                                    wave = "I%i" % ind_wave
                            
                                target_name = obj+"-"+wave
                                if obj == "tpeaks_iwave":
                                    target = params[Dtype]["targets"][obj][ind_subj][ind_wave]
                                else:
                                    target = params[Dtype]["targets"][obj][ind_subj][ind_wave]
                            
                                if target.sum() > 0:
                                    relative_error[ave_type][Dtype][weight][target_name] = np.abs(
                                        (property[ind_wave] - target) / target
                                    )
                                else:
                                    relative_error[ave_type][Dtype][weight][target_name] = np.abs(property[ind_wave])
                    
                        else:
                            property = props[ave_type][Dtype][weight][obj]
                            for ii in range(len(params[Dtype]["targets"][obj])):
                                if obj != "postmin_iwave":
                                    obj_name = obj + "-" + cell_order[ii]
                                else:
                                    obj_name = obj
                                if np.sum(np.abs(params[Dtype]["targets"][obj])) > 0:
                                    relative_error[ave_type][Dtype][weight][obj_name] = np.abs(
                                        (property[ii] - params[Dtype]["targets"][obj][ii]) / params[Dtype]["targets"][obj][ii]
                                    )
                                else:
                                    relative_error[ave_type][Dtype][weight][obj_name] = np.abs((property[ii] - params[Dtype]["targets"][obj][ii]))

# Between average model types and weights, which have the lowest error across both Dtypes?
total_errors = {}
diff_errors = {}
for ave_type in relative_error:
    total_errors[ave_type] = np.zeros(len(relative_error[ave_type][Dtype]))
    diff_errors[ave_type] = np.zeros(len(relative_error[ave_type][Dtype]))
    for ind, weight in enumerate(relative_error[ave_type][Dtype]):
        for obj in relative_error[ave_type][Dtype][weight]:
            if "iwave" in obj:
                for Dtype in relative_error[ave_type]:
                    total_errors[ave_type][ind] += relative_error[ave_type][Dtype][weight][obj]
            
                diff_errors[ave_type][ind] += np.abs(relative_error[ave_type]["D+"][weight][obj] - relative_error[ave_type]["D-"][weight][obj])

cost_ave = {}
norm_total = {}
norm_diff = {}
for ave_type in total_errors:
    min_total = np.min([np.min(total_errors[ave_type]) for ave_type in total_errors])
    max_total = np.max([np.max(total_errors[ave_type]) for ave_type in total_errors])
    min_diff = np.min([np.min(diff_errors[ave_type]) for ave_type in diff_errors])
    max_diff = np.max([np.max(diff_errors[ave_type]) for ave_type in diff_errors])
    norm_total[ave_type] = total_errors[ave_type] / max_total
    norm_diff[ave_type] = diff_errors[ave_type] / max_diff
    cost_ave[ave_type] = norm_total[ave_type] + norm_diff[ave_type]

# Get the best model waveforms
popsmooth_best = {}
t_I1wave_best = {}
delay_best = {}
spikes = {}
particle_best = {}

ave_type = "similar"

Dtype = "D+"
weight = 1
popsmooth_best[Dtype] = np.mean(results[ave_type][weight][Dtype]["popsmooths"][state][0], axis=0)
t = params[Dtype]["tres"] * np.arange(len(popsmooth_best[Dtype]))
t_I1wave_best[Dtype] = np.mean(results[ave_type][weight][Dtype]["t_I1wave"][state])
delay_best[Dtype] = t_dwave_exp[Dtype] - (t_I1wave_best[Dtype] + 2)
spikes[Dtype] = results[ave_type][weight][Dtype]["spikes"][state]
particle_best[Dtype] = results[ave_type][weight][Dtype]["chromosome"]

Dtype = "D-"
weight = 0
popsmooth_best[Dtype] = np.mean(results[ave_type][weight][Dtype]["popsmooths"][state][0], axis=0)
t_I1wave_best[Dtype] = np.mean(results[ave_type][weight][Dtype]["t_I1wave"][state])
delay_best[Dtype] = t_dwave_exp[Dtype] - (t_I1wave_best[Dtype] + 2)
spikes[Dtype] = results[ave_type][weight][Dtype]["spikes"][state]
particle_best[Dtype] = results[ave_type][weight][Dtype]["chromosome"]

# Get unified model waveforms
weight = 1
popsmooth_best["Unified"] = {}
t_I1wave_best["Unified"] = {}
spikes["Unified"] = {}
for Dtype in ["D+", "D-"]:
    popsmooth_best["Unified"][Dtype] = np.mean(results[ave_type][weight][Dtype]["popsmooths"][state][0], axis=0)
    t_I1wave_best["Unified"][Dtype] = np.mean(results[ave_type][weight][Dtype]["t_I1wave"][state])
    spikes["Unified"][Dtype] = spikes[Dtype] = results[ave_type][weight][Dtype]["spikes"][state]

# Plot unified model search costs
weights = sorted(list(relative_error[ave_type][Dtype].keys()))
norm_cost = cost_ave[ave_type] / cost_ave[ave_type].max()
norm_total[ave_type] /= norm_total[ave_type].max()
norm_diff[ave_type] /= norm_diff[ave_type].max()
plt_cost = plt.figure()
_=plt.plot(weights, norm_total[ave_type], linewidth=2, label="Total Error")
_=plt.plot(weights, norm_diff[ave_type], linewidth=2, label="Error Difference")
_=plt.plot(weights, norm_cost, linewidth=2, label="Cost")
_=plt.legend()
_=plt.grid(True)
_=plt.xlabel("Weight")
_=plt.ylabel("Normalized Value")

# Save data
ave_type = "similar"
weight = 1
save_data = {}
save_data["weights"] = weights
save_data["popsmooth"] = {}
save_data["popsmooth"]["Unified"] = {}
save_data["spike_times"] = {}
save_data["spike_times"]["Unified"] = {}
save_data["particle_best"] = {}
save_data["t_sim"] = t - 1
save_data["norm_cost"] = norm_cost
save_data["norm_diff"] = norm_diff[ave_type]
save_data["norm_total"] = norm_total[ave_type]
save_data["relative_error"] = {}
save_data["relative_error"]["D+"] = relative_error[ave_type]["D+"][1]
save_data["relative_error"]["D-"] = relative_error[ave_type]["D-"][0]
save_data["relative_error"]["Unified"] = relative_error[ave_type]["D-"][1]
for Dtype in filtered_exp:
    save_data["popsmooth"][Dtype] = {}
    save_data["popsmooth"][Dtype]["filtered_exp"] = filtered_exp[Dtype][ind_exp[Dtype]] / np.max(filtered_exp[Dtype][ind_exp[Dtype]][t_exp[Dtype] > 1.5])
    save_data["popsmooth"][Dtype]["t_exp"] = t_exp[Dtype] - delay_best[Dtype] - 1
    save_data["popsmooth"][Dtype]["simulation"] = popsmooth_best[Dtype] / np.max(popsmooth_best[Dtype][t > (t_I1wave_best[Dtype]-0.5)])
    save_data["popsmooth"]["Unified"][Dtype] = popsmooth_best["Unified"][Dtype] / np.max(popsmooth_best["Unified"][Dtype][t > (t_I1wave_best["Unified"][Dtype]-0.5)])
    save_data["spike_times"][Dtype] = spikes[Dtype]
    save_data["spike_times"]["Unified"][Dtype] = spikes["Unified"][Dtype]
    save_data["particle_best"][Dtype] = particle_best[Dtype]

with open(join(dir_data, "data_unified_model_search_best.pickle"), "wb") as f:
    pickle.dump(save_data, f)

plt.show()
