# Extract and save smoothed population rates used in Fig 6
# from full dataset

from os.path import join
from get_paths import *

import pickle
import numpy as np
import h5py

## Load data
# Load simulation results
fname_simulations = "data_TVAT_popsmooths.h5"
fname_optimization = "data_unified_model_search.pickle"
with open(join(dir_data, fname_optimization), "rb") as f:
    results = pickle.load(f)

parameter_indices = {}
with h5py.File(join(dir_data, fname_simulations), "r") as dset:
    values = dset["values"][:]
    popsmooths = dset["popsmooths"][:]
    for key in dset["parameter_indices"]:
        parameter_indices[key] = dset["parameter_indices"][key][:]

parameters = sorted(list(parameter_indices.keys()))

## Get indices for all popsmooths for which a parameter was varied
pair_indices = []
parameter_crosses = []
for ii in range(len(parameters) - 1):
    for jj in range(ii + 1, len(parameters)):
        pair_indices.append((ii, jj))
        parameter_crosses.append(parameters[ii] + " x " + parameters[jj])

ind_array = np.arange(values.shape[0])
parameter_indices = {}
idx_val_zero = {}
for ind_p, parameter in enumerate(parameters):
    parameter_indices[parameter] = []
    idx_val_zero[parameter] = []
    for ind in range(len(pair_indices)):
        if ind_p in pair_indices[ind]:
            ind_val = 0
            if pair_indices[ind][1] == ind_p:
                ind_val = 1
            ii, jj = pair_indices[ind]
            idx = values[:, 0] == ii
            idx *= values[:, 1] == jj
            parameter_indices[parameter] += list(ind_array[idx])
            
            idx_zero = idx * (values[:, ind_val + 2] == 0)
            idx_val_zero[parameter] += list(ind_array[idx_zero])

    parameter_indices[parameter] = np.array(list(set(parameter_indices[parameter]))).astype(int)
    idx_val_zero[parameter] = np.array(list(set(idx_val_zero[parameter]))).astype(int)

square_root = np.sqrt(len(parameters))
if np.floor(square_root) < square_root:
    n_row = int(np.floor(square_root))
    n_col = int(np.ceil(square_root))
else:
    n_row = n_col = int(square_root)

n_samples = 275
selective_parameters = ["L5", "L5input", "L23", "L6input"]

save_data = {}
for parameter in selective_parameters:
    save_data[parameter] = {}
    save_data[parameter]["zero"] = []
    save_data[parameter]["nonzero"] = []
    for ii in range(len(parameter_indices[parameter])):
        ind = parameter_indices[parameter][ii]
        if ind in idx_val_zero[parameter]:
            save_data[parameter]["zero"].append(popsmooths[ind, : n_samples])
        else:
            save_data[parameter]["nonzero"].append(popsmooths[ind, : n_samples])

with open(join(dir_data, "data_TVAT_waveforms_examples.pickle"), "wb") as f:
    pickle.dump(save_data, f)