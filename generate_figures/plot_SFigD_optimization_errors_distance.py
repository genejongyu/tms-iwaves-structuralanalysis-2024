# Plot optimization errors and best solution distance for S1 Appendix Fig D

from os.path import join
from get_paths import *

import matplotlib.pyplot as plt
import numpy as np
from OptimizationFitness import OptimizationFitness
from colors_paultol import *

if __name__ == "__main__":
    save_flag = 0
    
    # Load file
    fnames_data = {}
    fnames_settings = {}
    
    fnames_data["D+"] = [
        "data_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.h5",
        "data_pso_iwaves_column_restingstate_2013_31_120RMT_seed1.h5",
        "data_pso_iwaves_column_restingstate_2013_31_120RMT_seed2.h5",
        "data_pso_iwaves_column_restingstate_2013_31_120RMT_seed3.h5"
    ]
    fnames_settings["D+"] = [
        "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle",
        "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed1.pickle",
        "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed2.pickle",
        "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed3.pickle"
    ]
    
    fnames_data["D-"] = [
        "data_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.h5",
        "data_pso_iwaves_column_restingstate_2020_43_120RMT_seed1.h5",
        "data_pso_iwaves_column_restingstate_2020_43_120RMT_seed2.h5",
        "data_pso_iwaves_column_restingstate_2020_43_120RMT_seed3.h5"
    ]
    fnames_settings["D-"] = [
        "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle",
        "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed1.pickle",
        "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed2.pickle",
        "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed3.pickle"
    ]
    
    # Define histogram and smoothing parameters
    n_bins = 1001
    w_gaussian = 5
    Dtype = "D-"
    highest = 1000
    
    results = {}
    for Dtype in fnames_data:
        results[Dtype] = {}
        for ii in range(len(fnames_data[Dtype])):
            results[Dtype][ii] = OptimizationFitness(
                dir_data_original, 
                fnames_data[Dtype][ii], 
                fnames_settings[Dtype][ii],
                n_bins,
                w_gaussian
            )
        lowest = min([results[Dtype][ii].data["fitness"].min() for ii in results[Dtype]])
        for ii in results[Dtype]:
            results[Dtype][ii].set_plot_data(lo=lowest, hi=highest)
    
    # Get best individual for each optimization run
    ind_row = {}
    ind_col = {}
    particles_best = {}
    for Dtype in results:
        ind_row[Dtype] = []
        ind_col[Dtype] = []
        particles_best[Dtype] = []
        for ii in range(len(results[Dtype])):
            ind_best = np.argmin(results[Dtype][ii].data["fitness"])
            ind_row[Dtype].append(ind_best // results[Dtype][ii].n_pop)
            ind_col[Dtype].append(ind_best % results[Dtype][ii].n_pop)
            particles_best[Dtype].append(results[Dtype][ii].data["alleles"][ind_row[Dtype][ii]][ind_col[Dtype][ii]])

    # Get maximum distance for each dimension
    d_max = results[Dtype][ii].params["Bounder"]["hi"] - results[Dtype][ii].params["Bounder"]["lo"]

    # Compute normalized distance matrix
    d_mat = {}
    d_norm = {}
    for Dtype in particles_best:
        d_mat[Dtype] = np.zeros((len(results[Dtype]), len(results[Dtype])))
        d_norm[Dtype] = []
        for ii in range(len(results[Dtype]) - 1):
            for jj in range(ii + 1, len(results[Dtype])):
                d_norm[Dtype].append(np.abs(particles_best[Dtype][ii] - particles_best[Dtype][jj]) / d_max)
                d_mat[Dtype][ii][jj] = np.median(d_norm[Dtype][-1])
                d_mat[Dtype][jj][ii] = d_mat[Dtype][ii][jj]
    
    # Plot
    fig_distances = {}
    for Dtype in d_mat:
        fig_distances[Dtype] = plt.figure(figsize=(5, 4))
        fig_distances[Dtype].subplots_adjust(left=0.17, right=0.9, bottom=0.17, top=0.95)
        _=plt.imshow(d_mat[Dtype], interpolation="nearest", cmap=cmap_iridescent)
        cb = plt.colorbar()
        cb.set_label(label="Median Normalized Distance", size=16)
        _=plt.xticks(np.arange(len(results[Dtype])))
        _=plt.yticks(np.arange(len(results[Dtype])))
        _=plt.xlabel("Optimization Seed", fontsize=16)
        _=plt.ylabel("Optimization Seed", fontsize=16)

    fig_cumulative_mins_all = {}
    for Dtype in results:
        fig_cumulative_mins_all[Dtype] = plt.figure(figsize=(5, 4))
        fig_cumulative_mins_all[Dtype].subplots_adjust(left=0.17, right=0.96, bottom=0.17, top=0.9)
        for ii in results[Dtype]:
            _=plt.plot(results[Dtype][ii].cumulative_min_error_evolution, label="Seed: {}".format(ii), color=cset_muted[ii])
    
        _=plt.xlabel("Generation", fontsize=16)
        _=plt.ylabel("Total Error", fontsize=16)
        _=plt.title("{} Optimization".format(Dtype), fontsize=16)
        _=plt.grid(True)
        _=plt.legend()
        _=plt.ylim(0, 100)
        _=plt.xlim(0, results[Dtype][ii].n_gen)

    if save_flag:
        for Dtype in fig_distances:
            fig_distances[Dtype].savefig("SFigD_distances_{}.tiff".format(Dtype), dpi=600)
            fig_cumulative_mins_all[Dtype].savefig("SFigD_cumulativeminerror_{}.tiff".format(Dtype), dpi=600)
    
    plt.show()
