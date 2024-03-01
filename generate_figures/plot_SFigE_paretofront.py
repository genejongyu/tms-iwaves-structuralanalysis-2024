# Plot reduced pareto front for S1 Appendix Fig E

from os.path import join
from get_paths import *

import matplotlib.pyplot as plt
from OptimizationFitness import OptimizationFitness
import numpy as np
from colors_paultol import *

if __name__ == "__main__":
    save_flag = 0
    
    # Load file
    fname_data = "data_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.h5"
    fname_settings = "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle"
        
    # Define histogram and smoothing parameters
    n_bins = 1001
    w_gaussian = 5
    hi = 300
    
    results = OptimizationFitness(
        dir_data_original, 
        fname_data, 
        fname_settings,
        n_bins,
        w_gaussian)
    
    # Plot total error vs individual error
    results.calc_relative_error()

    # Compute pareto front
    groups = ["CS-Wave", "Spiking Activity", "Synchrony", "Well-Behaved"]
    results.get_pareto_dominants(groups)

    # Compute subgroups
    constraint_subgroups = {}
    for group in results.constraint_groups:
        constraint_subgroups[group] = {}
        for obj in results.params["objective_names"]:
            if group == "CS-Wave":
                if obj.endswith("_iwave"):
                    constraint_subgroups[group][obj] = []
            elif group == "Spiking Activity":
                if obj == "ISI_mean":
                    constraint_subgroups[group][obj] = []
            elif group == "Well-Behaved":
                if obj == "ISI_std_across_cell":
                    constraint_subgroups[group][obj] = []
            elif group == "Synchrony":
                if obj.startswith("baseline_"):
                    constraint_subgroups[group][obj] = []
    
    for group in constraint_subgroups:
        for subgroup in constraint_subgroups[group]:
            for obj in results.group_objectives:
                if obj.startswith(subgroup):
                    constraint_subgroups[group][subgroup].append(obj)
    
    subgrouped_error = {}
    for group in constraint_subgroups:
        subgrouped_error[group] = {}
        for subgroup in constraint_subgroups[group]:
            subgrouped_error[group][subgroup] = np.zeros(results.relative_error[obj].size)
            for obj in constraint_subgroups[group][subgroup]:
                subgrouped_error[group][subgroup] += results.relative_error[obj]
            
            subgrouped_error[group][subgroup] /= len(constraint_subgroups[group][subgroup])

    idx_lim = results.total_error < 150
    idx_good = results.grouped_error["CS-Wave"][results.dominators * idx_lim] < 0.5
    xticks = [30, 40, 60, 100]
    fig_total_error = {}
    for ind, group in enumerate(results.grouped_error):
        fig_total_error[group] = plt.figure(figsize=(5, 4))
        fig_total_error[group].subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.9)
        _=plt.title(group + " Error", fontsize=16)
        _=plt.scatter(results.total_error[idx_lim], results.grouped_error[group][idx_lim], 0.1, alpha=0.1, color=cset_muted[2])
        _=plt.scatter(results.total_error[results.dominators * idx_lim], results.grouped_error[group][results.dominators * idx_lim], 10, alpha=0.2, color=cset_muted[5], edgecolors="none")
        _=plt.xscale("log")
        _=plt.yscale("log")
        _=plt.xlabel("Total Error", fontsize=16)
        _=plt.ylabel(group + " Error", fontsize=16)
        _=plt.xticks(xticks)
        _=plt.yticks(font="Georgia")

    if save_flag:
        for group in fig_total_error:
            fig_total_error[group].savefig("SFigE_paretofront_{}.tiff".format(group), dpi=600)
    
    plt.show()

