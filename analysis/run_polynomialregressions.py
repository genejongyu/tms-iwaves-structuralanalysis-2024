# Script to perform polynomial regressions on TVAT data

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
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import cross_validate
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if __name__ == "__main__":
    ## Load data
    fname = "data_TVAT_measurements.h5"
    fpath_save = join(dir_data, fname)
    Dtypes = ["D+", "D-"]
    data = {}
    with h5py.File(fpath_save, "r") as dset:
        for key in dset:
            if key not in ["projection_indices", "n_search", "particle_best"] + Dtypes:
                data[key] = dset[key][:]
        
        data["projection_indices"] = {}
        for lesion in dset["projection_indices"]:
            data["projection_indices"][lesion] = dset["projection_indices"][lesion][:]
        
        data["n_search"] = dset["n_search"][()]
        
        data["particle_best"] = {}
        for Dtype in Dtypes:
            data["particle_best"][Dtype] = dset["particle_best"][Dtype][:]
            data[Dtype] = {}
            for obj in dset[Dtype]:
                data[Dtype][obj] = dset[Dtype][obj][:]
    
    projections = list(data["projection_indices"].keys())
    
    # Convert celltype names to proper labels
    celltypes = [projections[ii].split("-")[0] for ii in range(len(projections))]
    celltypes = list(set(celltypes))
    
    labels = {}
    for ii in range(len(celltypes)):
        label = celltypes[ii]
        if "23" in label:
            split_name = label.split("23")
            label = split_name[0] + "2/3"
            if len(split_name) > 1:
                label += split_name[1]
        
        if label.startswith("L"):
            label += " PC"
        
        if label.startswith("I"):
            label= "L" + label[1:]
            label += " INT"
        
        if "input" in label:
            label = label.replace("input", "")
            label += " AFF"
        
        labels[celltypes[ii]] = label
    
    for ii in range(len(projections)):
        if "-" in projections[ii]:
            cell1, cell2 = projections[ii].split("-")
            projections[ii] = labels[cell1] + "-" + labels[cell2]
        else:
            projections[ii] = "TMS-" + labels[projections[ii]]
    
    # Create table matrices
    layers = ["L2/3", "L5", "L6"]
    celltypes_labels = ["PC AFF", "PC", "INT AFF", "INT"]
    rows = ["TMS"]
    cols = []
    for layer in layers:
        for celltype in celltypes_labels:
            rows.append(layer + " " + celltype)
            cols.append(layer + " " + celltype)

    projections_resorted = []
    inds_resorted = []
    for ii in range(len(rows)):
        for jj in range(len(cols)):
            for kk in range(len(projections)):
                pre, post = projections[kk].split("-")
                if pre == rows[ii]:
                    if post == cols[jj]:
                        projections_resorted.append(projections[kk])
                        inds_resorted.append(kk)
    
    inds_resorted = np.array(inds_resorted)
    
    # Create index pair combinations
    indices = []
    projection_crosses = []
    for ii in range(len(projections) - 1):
        for jj in range(ii + 1, len(projections)):
            indices.append((ii, jj))
            projection_crosses.append(projections[ii] + " x " + projections[jj])

    ## Perform regressions
    coefs = {}
    rsquared = {}
    waves = ["D", "I1", "I2", "I3"]
    num_coefs = 9
    l1_ratios = [0.05, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
    n_folds = 10
    max_iter = 500000
    n_jobs = 1
    verbose = False
    idx_shuffle = np.arange(data["n_search"] ** 2)
    np.random.shuffle(idx_shuffle)
    for Dtype in Dtypes:
        if rank == 0:
            print(Dtype)
        
        coefs[Dtype] = {}
        rsquared[Dtype] = {}
        for ind, wave in enumerate(waves):
            if rank == 0:
                print("\t" + wave)
            

            if rank == 0:
                coefs[Dtype][wave] = np.zeros((len(indices), num_coefs))
                rsquared[Dtype][wave] = np.zeros(len(indices))
            else:
                num_indices = np.arange(rank, len(indices), nhost).size
                coefs[Dtype][wave] = np.zeros((num_indices, num_coefs))
                rsquared[Dtype][wave] = np.zeros(num_indices)
            
            count = 0
            for ind in range(rank, len(indices), nhost):
                ii, jj = indices[ind]
                if rank == 0:
                    print("\t\t{} out of {}".format(ind, 42 * 41 // 2))
                
                idx = data["values"][:, 0] == ii
                idx *= data["values"][:, 1] == jj
                
                x = np.zeros((data["n_search"] ** 2, num_coefs))
                x[:, 0] = data["values"][:, 2][idx] / data["bounds"][ii]
                x[:, 1] = data["values"][:, 3][idx] / data["bounds"][jj]
                x[:, 0] -= x[:, 0].mean()
                x[:, 1] -= x[:, 1].mean()
                x[:, 0] /= x[:, 0].max()
                x[:, 1] /= x[:, 1].max()
                
                x[:, 2] = x[:, 0] * x[:, 1]
                x[:, 3] = x[:, 0] * x[:, 0]
                x[:, 4] = x[:, 1] * x[:, 1]
                
                x[:, 5] = x[:, 0] * x[:, 1] * x[:, 0]
                x[:, 6] = x[:, 0] * x[:, 1] * x[:, 1]
                
                x[:, 7] = x[:, 0] * x[:, 0] * x[:, 0]
                x[:, 8] = x[:, 1] * x[:, 1] * x[:, 1]
                
                y = data[Dtype]["peaks_iwave"][idx][:, 0, ind_wave]
                y += np.abs(data[Dtype]["minima_iwave"][idx][:, 0, ind_wave])
                y -= y.mean()
                
                x = x[idx_shuffle]
                y = y[idx_shuffle]
                
                model_cv = ElasticNetCV(
                    l1_ratio=l1_ratios,
                    cv=n_folds, 
                    max_iter=max_iter,
                    fit_intercept=False,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    selection="random",
                    random_state=1,
                ).fit(x, y)
                model = ElasticNet(
                    alpha=model_cv.alpha_,
                    l1_ratio=model_cv.l1_ratio_,
                    max_iter=max_iter,
                    fit_intercept=False,
                    selection="random",
                    random_state=1,
                )
                validate = cross_validate(model, x, y, cv=n_folds, return_estimator=True)
                coefs_val = np.zeros((n_folds, x.shape[1]))
                for fold in range(n_folds):
                    coefs_val[fold] = validate["estimator"][fold].coef_
                
                r2 = validate["test_score"].mean()
                
                if rank == 0:
                    coefs[Dtype][wave][ind] = coefs_val.mean(axis=0)
                    rsquared[Dtype][wave][ind] = r2
                else:
                    coefs[Dtype][wave][count] = coefs_val.mean(axis=0)
                    rsquared[Dtype][wave][count] = r2
                    count += 1
            
            # Collect data
            if nhost > 1:
                # Collecting data
                if rank == 0:
                    for send_rank in range(1, nhost):
                        send_indices = np.arange(send_rank, len(indices), nhost)
                        size_coefs = send_indices.size * num_coefs
                        size_rsquared = send_indices.size
                        msg = np.empty(size_coefs + size_rsquared, dtype=int)
                        comm.Recv(msg, source=send_rank, tag=13)
                        coefs_send = msg[: size_coefs].reshape(send_indices.size, num_coefs)
                        rsquared_send = msg[size_coefs :]
                        for send_ind, save_ind in enumerate(send_indices):
                            coefs[Dtype][wave][save_ind] = coefs_send[send_ind]
                            rsquared[Dtype][wave][save_ind] = rsquared_send[send_ind]
                    
                    rsquared[Dtype][wave] = np.array(rsquared[Dtype][wave])
                
                # Sending data
                else:
                    coefs_send = np.array(coefs[Dtype][wave]).flatten()
                    rsquared_send = np.array(rsquared[Dtype][wave])
                    
                    msg = np.hstack([coefs_send, rsquared_send])
                    comm.Send(msg, dest=0, tag=13)
    
    if rank == 0:
        save_data = {}
        save_data["coefs"] = coefs
        save_data["R^2"] = rsquared
        save_data["projection_crosses"] = projection_crosses
        save_data["projections_resorted"] = projections_resorted
        save_data["inds_resorted"] = inds_resorted
        save_data["particle_best"] = data["particle_best"]
        save_data["parameters"] = projections
        with open(join(dir_data, "data_tvat_polynomialregressions.pickle"), "wb") as f:
            pickle.dump(save_data, f)

        ET = cookie.time() - ST
        print("Took {} seconds".format(ET))
        
