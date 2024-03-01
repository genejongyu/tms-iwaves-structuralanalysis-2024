# Script to save smoothed population rates obtained during TVAT search

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nhost = comm.Get_size()

import sys
sys.path.append("..")
import h5py
import numpy as np
import pso_comp
from SensitivityAnalysis import SensitivityAnalysis
import time as cookie
from os.path import join

if __name__ == "__main__":
    fpath_save = join(dir_data, "data_TVAT_popsmooths.h5")
    
    ## Load data
    # Load simulation results
    fname = "data_unified_model_search.pickle"
    with open(join(dir_data, fname), "rb") as f:
        results = pickle.load(f)

    # Get best particle
    ave_type = "similar"
    Dtype = "D+"
    weight = 1
    state = "resting"
    particle_best = results[ave_type][weight][Dtype]["chromosome"]
    
    # Load settings files
    fname_settings = {}
    fname_settings["D+"] = "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle"
    fname_settings["D-"] = "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle"
    with open(join(dir_data, fname_settings[Dtype]), "rb") as f:
        params = pickle.load(f)
    
    ## Obtain indices associated with each parameter
    con = params["con"]["p"]

    # Excitatory connections
    parameter_indices = {}
    for ii in range(len(params["parameters"])):
        if "AMPA" in params["save_keys"][ii].astype(str):
            ptype, pre, post, syn = params["parameters"][ii]
            parameter_indices[pre + "-" + post] = [ii]
    
    # Inhibitory connections will have two indices, one for GABAB and one for GABAB
    for pretype in con:
        if "input" in pretype:
            continue
        if pretype.startswith("I"):
            for posttype in con[pretype]:
                for ii in range(len(params["parameters"])):
                    if "GABA" in params["save_keys"][ii].astype(str):
                        ptype, pre, post, syn = params["parameters"][ii]
                        parameter = pre + "-" + post
                        if parameter not in parameter_indices:
                            parameter_indices[parameter] = [ii]
                        else:
                            parameter_indices[parameter].append(ii)
    
    # Activations
    for ii in range(len(params["parameters"])):
        if params["save_keys"][ii].astype(str).startswith("p_active"):
            ptype, subj, amp, pre = params["parameters"][ii]
            parameter_indices[pre] = [ii]
    
    parameter_names = sorted(list(parameter_indices.keys()))
    
    # Define TVAT sampling
    n_parameters = len(parameter_indices)
    bounds = np.zeros(n_parameters)
    for ind, parameter in enumerate(parameter_names):
        if "-" in parameter:
            bounds[ind] = 10
        else:
            bounds[ind] = 1
    
    n_search = 11
    base_scalars = np.linspace(0, 1, n_search)
    n_combos = n_parameters * (n_parameters - 1) // 2
    n_search2 = n_search ** 2
    n_values = n_combos * n_search2
    values = np.zeros((n_values, 4))
    count = 0
    for ii in range(n_parameters - 1):
        for jj in range(ii + 1, n_parameters):
            values[count * n_search2 : (count+1) * n_search2, 0] = ii
            values[count * n_search2 : (count+1) * n_search2, 1] = jj
            val1, val2 = np.meshgrid(base_scalars * bounds[ii], base_scalars * bounds[jj])
            values[count * n_search2 : (count+1) * n_search2, 2] = val1.ravel()
            values[count * n_search2 : (count+1) * n_search2, 3] = val2.ravel()
            count += 1
    
    # Create h5 file
    tres = params["tres"]
    t_window = 10
    n_samples = int(t_window / tres)
    if rank == 0:
        with h5py.File(fpath_save, "w") as dset:
            dset["particle_best"] = particle_best
            dset["popsmooths"] = np.zeros((n_values, n_samples))
            dset["values"] = values
            dset.create_group("parameter_indices")
            for parameter in parameter_indices:
                dset["parameter_indices"][parameter] = parameter_indices[parameter]

    # Launch simulations in parallel
    loops = 1
    ST = cookie.time()
    for ii in range(rank, n_values, nhost):
        popsmooths = {}
        if rank == 0:
            print("Evaluating {0} out of {1}".format(ii, n_values))
        
        # Modify parameters
        particle = particle_best.copy()
        ind_parameter1 = int(values[ii][0])
        ind_parameter2 = int(values[ii][1])
        parameter1 = parameter_names[ind_parameter1]
        parameter2 = parameter_names[ind_parameter2]
        
        value1 = values[ii][2]
        value2 = values[ii][3]
        for index in parameter_indices[parameter1]:
            particle[index] = value1

        for index in parameter_indices[parameter2]:
            particle[index] = value2
        
        # Run simulation
        props, sim_results = pso_comp.queue_evaluate(
            particle, params, validate=True
        )
        if "popsmooths" in sim_results:
            popsmooths[ii] = sim_results["popsmooths"]["resting"][0][:, : n_samples].mean(axis=0)
        else:
            popsmooths[ii] = -1 * np.ones(n_samples)
        
        # Send and receive
        if rank > 0:
            # Send
            msg = np.empty(n_samples + 1, dtype=np.float64)
            msg[0] = ii
            msg[1:] = popsmooths[ii]
            comm.Send(msg, dest=0, tag=13)
        else:
            # Receive
            for rank_sender in range(1, nhost):
                indices = np.arange(rank_sender, n_values, nhost)
                # Check that there is something to receive before receiving
                if loops > len(indices):
                    print("\t Skipped rank {0}; size {1}".format(rank_sender, len(indices)))
                    continue
                print("\t Receiving from rank {0}; size {1}".format(rank_sender, len(indices)))
                msg = np.empty(n_samples + 1, dtype=np.float64)
                comm.Recv(msg, source=rank_sender, tag=13)
                ind_val = int(msg[0])
                popsmooths[ind_val] = msg[1:]

            # Save data
            with h5py.File(fpath_save, "r+") as dset:
                for ii in popsmooths:
                    dset["popsmooths"][ii] = popsmooths[ii]

        loops += 1

    if rank == 0:
        print("Took {} seconds total".format(cookie.time() - ST))
