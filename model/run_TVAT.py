# Script to run TVAT analysis using identified models as fixed points

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nhost = comm.Get_size()
except ModuleNotFoundError:
    rank = 0
    nhost = 1

from os.path import join
from get_paths import *

import h5py
import numpy as np
import pso_comp
import time as cookie
import pickle

if __name__ == "__main__":
    fpath_save = join(dir_data, "data_TVAT_measurements.h5")
    
    start = 0
    
    ## Load data
    # Load simulation results
    fname_data = "data_unified_model_search.pickle"
    with open(fname_data, "rb") as f:
        results_orig = pickle.load(f)
    
    results_orig = results_orig["best"][(0, 0)]
    
    # Load settings file
    fname_settings = {}
    fname_settings["D+"] = "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle"
    fname_settings["D-"] = "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle"
    
    params = {}
    for Dtype in fname_settings:
        with open(join(dir_data, fname_settings[Dtype]), "rb") as f:
            params[Dtype] = pickle.load(f)
    
    ## Get best parameters
    ave_type = "similar"
    weight = 1
    state = "resting"
    particle_best = {}
    for Dtype in results_orig[weight]:
        particle_best[Dtype] = results_orig[weight][Dtype]["chromosome"]
    
    ## Get variable names
    parameter_names = []
    for ii in range(len(params[Dtype]["parameters"])):
        if "logistic" in params[Dtype]["parameters"][ii][0]:
            continue
        if "iclamp" in params[Dtype]["parameters"][ii][0]:
            continue
        parameter_name = ""
        for subname in params[Dtype]["parameters"][ii]:
            if str(subname).startswith("p_active"):
                subname = "p_active"
            parameter_name += str(subname)+"-"
        
        parameter_names.append(parameter_name[:-1])
    
    ## Get indices for projections
    con = params[Dtype]["con"]["p"]
    
    # Excitatory connections
    ind_syn = []
    projection_indices = {}
    for ii in range(len(parameter_names)):
        if "AMPA" in parameter_names[ii]:
            ptype, pre, post, syn = params[Dtype]["parameters"][ii]
            projection_indices[pre + "-" + post] = [ii]
            ind_syn.append(ii)

    # Inhibitory connections will have two indices, one for GABAB and one for GABAB
    for pre in con:
        if "input" in pre:
            continue
        if pre.startswith("I"):
            for post in con[pre]:
                projection = pre + "-" + post
                for ii in range(len(parameter_names)):
                    if parameter_names[ii].startswith("strength"):
                        if projection in parameter_names[ii]:
                            if projection not in projection_indices:
                                projection_indices[projection] = [ii]
                            else:
                                projection_indices[projection].append(ii)
                            ind_syn.append(ii)

    # Activations
    ind_tms = {}
    for Dtype in params:
        ind_tms[Dtype] = []
        for ii in range(len(parameter_names)):
            if parameter_names[ii].startswith("p_active"):
                ptype, subj, amp, pre = params[Dtype]["parameters"][ii]
                projection_indices[pre] = [ii]
                if pre != "L5":
                    ind_tms[Dtype].append(ii)
        ind_tms[Dtype] = np.array(ind_tms[Dtype])
    
    projection_names = sorted(list(projection_indices.keys()))
    ind_syn = np.array(ind_syn)

    ## Define TVAT sampling
    # Set bounds
    n_parameters = len(projection_indices)
    bounds = np.zeros(n_parameters)
    for ind, projection in enumerate(projection_names):
        if "-" in projection:
            bounds[ind] = 10
        else:
            bounds[ind] = 1
    
    n_search = 21
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

    ## Initialize h5 data file
    if start == 0:
        if rank == 0:
            print("Bounds", bounds)
            with h5py.File(fpath_save, "w") as dset:
                dset["n_search"] = n_search
                dset["values"] = values
                dset.create_group("particle_best")
                for Dtype in particle_best:
                    dset["particle_best"][Dtype] = particle_best[Dtype]
                dset["bounds"] = bounds
                dset.create_group("projection_indices")
                for projection in projection_names:
                    dset["projection_indices"][projection] = projection_indices[projection]
                
                for Dtype in params:
                    dset.create_group(Dtype)
                    for obj in params[Dtype]["objective_names"]:
                        dset[Dtype][obj] = np.zeros((n_values,) + params[Dtype]["targets"][obj].shape)
    
    comm.Barrier()

    ## Evaluate samples
    ST0 = cookie.time()
    for Dtype in params:
        loops = 1
        offset = 0
        if start > 0:
            loops = start
            offset = (start - 1) * nhost
    
        for ii in range(rank + offset, n_values, nhost):
            if rank == 0:
                ST = cookie.time()
                print("Evaluating loop {0} out of {1}".format(loops, len(np.arange(rank, n_values, nhost))))
        
            # Apply particle parameters to synapses/projections
            props = {}
            particle = particle_best[Dtype].copy()

            # Set remaining particle values
            ind_projection1 = int(values[ii][0])
            ind_projection2 = int(values[ii][1])
            projection1 = projection_names[ind_projection1]
            projection2 = projection_names[ind_projection2]
            
            value1 = values[ii][2]
            value2 = values[ii][3]
            for ind_particle in projection_indices[projection1]:
                particle[ind_particle] = value1
            
            for ind_particle in projection_indices[projection2]:
                particle[ind_particle] = value2
            
            # Run simulation
            props[ii] = pso_comp.queue_evaluate(particle, params[Dtype]) 
            
            # Un-normalize peaks_iwave
            props[ii]["peaks_iwave"] *= props[ii]["amp_I1"]
            props[ii]["minima_iwave"] *= props[ii]["amp_I1"]
            
            # Send and receive
            if rank > 0:
                # Send
                msg = np.empty(
                    (len(params[Dtype]["objective_names"]) + 1, params[Dtype]["size_obj_max"]),
                    dtype=np.float64
                )
                msg[0][0] = ii
                for ind_obj, obj in enumerate(params[Dtype]["objective_names"]):
                    if obj in ["peaks_iwave", "tpeaks_iwave", "minima_iwave"]:
                        rows, cols = params[Dtype]["targets"][obj].shape
                        cols += params[Dtype]["extra_waves"]
                        msg[ind_obj + 1, : rows * cols] = props[ii][obj].ravel()
                    else:
                        msg[ind_obj + 1, : params[Dtype]["targets"][obj].size] = props[ii][obj].ravel()
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
                    msg = np.empty(
                        (len(params[Dtype]["objective_names"]) + 1, params[Dtype]["size_obj_max"]),
                        dtype=np.float64
                    )
                    comm.Recv(msg, source=rank_sender, tag=13)
                    ind_val = int(msg[0][0])
                    props[ind_val] = {}
                    for ind_obj, obj in enumerate(params[Dtype]["objective_names"]):
                        if obj in ["peaks_iwave", "tpeaks_iwave", "minima_iwave"]:
                            rows, cols = params[Dtype]["targets"][obj].shape
                            cols += params[Dtype]["extra_waves"]
                            props[ind_val][obj] = msg[ind_obj + 1, : rows * cols].reshape(rows, cols)
                            props[ind_val][obj] = props[ind_val][obj][:, :params[Dtype]["measured_waves"]]
                        else:
                            props[ind_val][obj] = msg[ind_obj + 1, : params[Dtype]["targets"][obj].size]
                # Save data
                with h5py.File(fpath_save, "r+") as dset:
                    for ii in props:
                        for obj in params[Dtype]["objective_names"]:
                            try:
                                dset[Dtype][obj][ii] = props[ii][obj]
                            except TypeError:
                                dset[Dtype][obj][ii] = props[ii][obj][:, :params[Dtype]["measured_waves"]]
                
                print("Loop {0} finished in {1} seconds".format(loops, cookie.time() - ST))

        loops += 1
    
    comm.Barrier()
    
    if rank == 0:
        print("Took {} seconds total".format(cookie.time() - ST0))
