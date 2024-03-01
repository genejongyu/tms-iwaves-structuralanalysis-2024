# Plots metrics used to identify ideal time-step to balance accuracy and speed
# for Fig 8

from os.path import join
from get_paths import *

import matplotlib.pyplot as plt
import numpy as np
import pickle
from kneed import KneeLocator
from colors_paultol import cset_high_contrast

if __name__ == "__main__":
    save_flag = 0
    
    # Load file
    with open(join(dir_data, "data_timestep.pickle"), "rb") as f:
        data = pickle.load(f)

    # Define metrics that were defined relative to a baseline
    metrics_relative = ["nrmse", "vr_dist"]
    
    # Define keys that are not metrics
    nonmetrics = ["treses", "spiketimes", "vs", "ts", "v_tstart", "v_tstop"]

    # Define axis labels for metrics
    ylabels = {}
    ylabels["num_spikes"] = "Number of Spikes"
    ylabels["mean_ISI"] = "Mean ISI (ms)"
    ylabels["cv_ISI"] = "CV ISI (%)"
    ylabels["nrmse"] = "NRMSE (rel. to 0.001 ms)"
    ylabels["vr_dist"] = "Spike Dist. (rel. to 0.001 ms)"
    
    # Define titles for metrics
    titles = {}
    titles["num_spikes"] = "Number of Spikes"
    titles["mean_ISI"] = "Mean ISI"
    titles["cv_ISI"] = "Coefficient of Variation ISI"
    titles["nrmse"] = "NRMSE of Somatic Membrane Potential"
    titles["vr_dist"] = "Van Rossum Spike Distance"
    
    # Convert cv_ISI into percentage
    data["cv_ISI"] *= 100
    
    # Interpolate data
    num_interp = 101
    tres_interp1 = np.logspace(np.log10(data["treses"][0]), np.log10(data["treses"][-1]), num_interp)
    tres_interp2 = np.logspace(np.log10(data["treses"][1]), np.log10(data["treses"][-1]), num_interp)
    data_interp = {}
    for metric in data:
        if metric in nonmetrics:
            continue
        elif metric in metrics_relative:
            data_interp[metric] = np.interp(tres_interp2, data["treses"][1:], data[metric].mean(axis=1))
        else:
            data_interp[metric] = np.interp(tres_interp1, data["treses"], data[metric].mean(axis=1))
    
    kneedles = {}
    kneedles['num_spikes'] = KneeLocator(np.log10(tres_interp1), data_interp['num_spikes'], curve='convex', direction='increasing', interp_method='polynomial', polynomial_degree=5)
    kneedles['mean_ISI'] = KneeLocator(np.log10(tres_interp1), data_interp['mean_ISI'], curve='concave', direction='decreasing', interp_method='polynomial', polynomial_degree=5)
    kneedles['cv_ISI'] = KneeLocator(np.log10(tres_interp1), data_interp['cv_ISI'], curve='concave', direction='decreasing', interp_method='polynomial', polynomial_degree=5)
    kneedles['nrmse'] = KneeLocator(np.log10(tres_interp2), data_interp['nrmse'], curve='convex', direction='increasing', interp_method='polynomial', polynomial_degree=5)
    kneedles['vr_dist'] = KneeLocator(np.log10(tres_interp2), data_interp['vr_dist'], curve='convex', direction='increasing', interp_method='polynomial', polynomial_degree=5)

    # Colors
    ind_order = [1, 2, 0]
    colors = [cset_high_contrast[ind] for ind in ind_order]
    colors_v = np.array([
        [181, 221, 216],
        [155, 210, 225],
        [129, 196, 231],
        [126, 178, 228],
        [147, 152, 210],
        [157, 125, 178],
        [144, 99, 136],
        [104, 73, 87]
    ]) / 255
    
    # Plot knees
    figsize = (3, 2.5)
    bottom = 0.2
    left = 0.2
    top = 0.97
    right = 0.97
    knees = {}
    fig_knees = {}
    for metric in kneedles:
        fig_knees[metric] = plt.figure(figsize=figsize)
        fig_knees[metric].subplots_adjust(bottom=bottom, left=left, top=top, right=right)
        # plt.title(titles[metric])
        plt.plot(kneedles[metric].x_normalized, kneedles[metric].y_normalized, color=colors[2], label="Normalized")
        plt.plot(kneedles[metric].x_difference, kneedles[metric].y_difference, color=colors[1], label="Difference")
        plt.xticks(
            np.arange(kneedles[metric].x_normalized.min(), kneedles[metric].x_normalized.max() + 0.1, 0.2)
        )
        plt.yticks(
            np.arange(0, 1.1, 0.2)
        )
        plt.axvline(
            kneedles[metric].norm_knee,
            linestyle="--",
            color="k",
            label="Knee"
        )
        plt.xlim(-0.1, 1.1)
        plt.ylabel("Normalized Curve")
        plt.xlabel("Normalized $Log_{10}$ of Time-Step")
        plt.legend(loc="best")
        plt.grid(True)
        knees[metric] = 10 ** kneedles[metric].knee

    # Plot metrics
    fig_scatter = {}
    for metric in data_interp:
        fig_scatter[metric] = plt.figure(figsize=figsize)
        fig_scatter[metric].subplots_adjust(bottom=bottom, left=left, top=top, right=right)
        if metric in metrics_relative:
            tres_plot = data["treses"][1:]
        else:
            tres_plot = data["treses"]
        _=plt.plot(tres_plot, data[metric].mean(axis=1), linewidth=2, color=colors[0])
        for ii in range(len(tres_plot)):
            _=plt.errorbar(
                [tres_plot[ii]],
                [data[metric].mean(axis=1)[ii]], 
                yerr=[data[metric].std(axis=1)[ii]], 
                color=colors[0],
                markeredgecolor="white",
                markeredgewidth=1.5,
                linewidth=2,
                fmt="o", 
                markersize=10, 
                capsize=3
            )
        
        plt.axvline(
            knees[metric],
            linestyle="--",
            color="k"
        )
        _=plt.xlabel('Time-Step Size (ms)')
        _=plt.ylabel(ylabels[metric])
        _=plt.xlim(data["treses"][0]/2, data["treses"][-1]*2)
        _=plt.xscale('log')
        _=plt.grid(True)
    
    figsize = (6, 2.3)
    bottom = 0.2
    left = 0.2
    t_offset = 6370
    v_offset = 3
    fig_v = plt.figure(figsize=figsize)
    fig_v.subplots_adjust(bottom=bottom, left=left)
    for ii in range(len(data["treses"])):
        v = data["vs"][data["treses"][ii]]
        t = data["ts"][data["treses"][ii]]
        _=plt.plot(t - t_offset, v - ii * v_offset, label=str(data["treses"][ii]), color=colors_v[ii], linewidth=2)

    _=plt.legend(ncol=2)
    _=plt.grid(True)
    _=plt.ylabel('Membrane Voltage (mV)')
    _=plt.xlabel('Time (ms)')
    _=plt.xlim(data["v_tstart"] - t_offset, data["v_tstop"] - t_offset)

    if save_flag:
        for metric in fig_knees:
            fig_knees[metric].savefig("Fig8_knee_{}.tiff".format(metric), dpi=600)
            fig_scatter[metric].savefig("Fig8_scatter_{}.tiff".format(metric), dpi=600)
    
        fig_v.savefig("Fig8_membranepotential.tiff", dpi=600)
    
    plt.show()
