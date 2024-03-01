# Extract and separately save out TVAT surfaces used in Fig 5A-B
# from full dataset

from os.path import join
from get_paths import *

import h5py
import numpy as np
import pickle

if __name__ == "__main__":
    ## File names
    fname_regressions = "data_tvat_polynomialregressions.pickle"
    fname_tvat = "data_TVAT_measurements.h5"
    fname_save = "data_tvat_surfaces_examples.pickle"
    
    fpath_regressions = join(dir_data, fname_regressions)
    fpath_tvat = join(dir_data, fname_tvat)
    fpath_save = join(dir_data, fname_save)

    ## Load data
    Dtypes = ["D+"]
    with open(fpath_regressions, "rb") as f:
        regr = pickle.load(f)
    
    rsquared = {}
    coefs = {}
    for Dtype in Dtypes:
        rsquared[Dtype] = regr["R^2"][Dtype]
        coefs[Dtype] = regr["coefs"][Dtype]
    
    projection_crosses = regr["parameter_crosses"]
    
    data = {}
    with h5py.File(fpath_tvat, "r") as dset:
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
            if label.endswith("5"):
                label += " PC"
            else:
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
    
    waves = ["D", "I1", "I2", "I3"]
    for ind_cross in range(len(projection_crosses)):
        r2s = []
        for ind_wave, wave in enumerate(waves):
            r2s.append(rsquared["D+"][wave][ind_cross])
    
    # Get examples of good and bad fits
    proj1 = {}
    proj2 = {}
    proj1["good"] = "L5 PC-L5 PC"
    proj2["good"] = "TMS-L5 PC"
    proj1["bad"] = "TMS-L6 INT"
    proj2["bad"] = "L6 INT AFF-L6 INT"
    plot_label1 = {}
    plot_label2 = {}
    for fit_type in proj1:
        plot_label1[fit_type] = proj1[fit_type].replace("L5 PC", "L5 PTN")
        plot_label1[fit_type] = plot_label1[fit_type].replace("PC", "IT")
        plot_label1[fit_type] = plot_label1[fit_type].replace("INT", "BC")
        plot_label2[fit_type] = proj2[fit_type].replace("L5 PC", "L5 PTN")
        plot_label2[fit_type] = plot_label2[fit_type].replace("PC", "IT")
        plot_label2[fit_type] = plot_label2[fit_type].replace("INT", "BC")
    wave_plot = ["D", "I1"]
    save_data = {}
    for fit_type in proj1:
        for ii in range(len(projections) - 1):
            for jj in range(ii + 1, len(projections)):
                flag = 0
                if (projections[ii] == proj1[fit_type]) or (projections[ii] == proj2[fit_type]):
                    flag += 1
                if (projections[jj] == proj1[fit_type]) or (projections[jj] == proj2[fit_type]):
                    flag += 1
                if flag < 2:
                    continue
                if projections[ii] == proj1[fit_type]:
                    xlabel = plot_label1[fit_type]
                    ylabel = plot_label2[fit_type]
                else:
                    xlabel = plot_label2[fit_type]
                    ylabel = plot_label1[fit_type]
                if xlabel.startswith("TMS"):
                    xticks = ["0", "0.5", "1"]
                else:
                    xticks = ["0", "5", "10"]
                if ylabel.startswith("TMS"):
                    yticks = ["0", "0.5", "1"]
                else:
                    yticks = ["0", "5", "10"]
                
                idx = data["values"][:, 0] == ii
                idx *= data["values"][:, 1] == jj
                projection_cross_12 = "{} x {}".format(proj1[fit_type], proj2[fit_type])
                projection_cross_21 = "{} x {}".format(proj2[fit_type], proj1[fit_type])
                for pp in range(len(projection_crosses)):
                    if projection_crosses[pp] == projection_cross_12:
                        ind_cross = pp
                        break
                    if projection_crosses[pp] == projection_cross_21:
                        ind_cross = pp
                        break

                save_data[projection_crosses[pp]] = {}
                save_data[projection_crosses[pp]]["xlabel"] = xlabel
                save_data[projection_crosses[pp]]["ylabel"] = ylabel
                save_data[projection_crosses[pp]]["r2"] = {}
                save_data[projection_crosses[pp]]["sim_data"] = {}
                save_data[projection_crosses[pp]]["regression"] = {}
                
                for ind_wave, wave in enumerate(wave_plot):
                    r2 = rsquared["D+"][wave][ind_cross]
                    y = data[Dtype]["peaks_iwave"][idx][:, 0, ind_wave].reshape(21, 21)
                    
                    # Get polynomial regressions
                    X, Y = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
                    x = np.zeros((data["n_search"], data["n_search"], 9))
                    x[:, :, 0] = X.copy()
                    x[:, :, 1] = Y.copy()
                    x[:, :, 2] = x[:, :, 0] * x[:, :, 1]
                    x[:, :, 3] = x[:, :, 0] * x[:, :, 0]
                    x[:, :, 4] = x[:, :, 1] * x[:, :, 1]
                    x[:, :, 5] = x[:, :, 0] * x[:, :, 1] * x[:, :, 0]
                    x[:, :, 6] = x[:, :, 0] * x[:, :, 1] * x[:, :, 1]
                    x[:, :, 7] = x[:, :, 0] * x[:, :, 0] * x[:, :, 0]
                    x[:, :, 8] = x[:, :, 1] * x[:, :, 1] * x[:, :, 1]
                
                    yhat = np.zeros((data["n_search"], data["n_search"]))
                    for ii in range(9):
                        yhat += coefs["D+"][wave][ind_cross][ii] * x[:, :, ii]

                    yhat += y.mean()
                    
                    save_data[projection_crosses[pp]]["r2"][wave] = r2
                    save_data[projection_crosses[pp]]["sim_data"][wave] = y
                    save_data[projection_crosses[pp]]["regression"][wave] = yhat 

    with open(fpath_save, "wb") as f:
        pickle.dump(save_data, f)
              
