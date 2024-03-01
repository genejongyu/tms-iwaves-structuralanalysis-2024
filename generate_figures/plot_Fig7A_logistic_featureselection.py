# Plots sum of coefficients as a function of regularization strength for 
# feature selection to classify preferential parameters for Fig 7A

from os.path import join
from get_paths import *

import classification
import numpy as np
import matplotlib.pyplot as plt
from TVATAnalysis import TVATAnalysis
from colors_paultol import cset_muted

if __name__ == "__main__":
    save_flag = 0
    
    ## Load data
    fname_regressions = "data_tvat_polynomialregressions.pickle"
    
    fname_settings = {}
    fname_settings["D+"] = "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle"
    fname_settings["D-"] = "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle"
   
    results = TVATAnalysis(dir_data, fname_regressions, fname_settings)
    
    cdf_threshold = 0.99
    
    interaction_flag = 0
    n_higher_order = 1
    intercept_flag = 0
    
    n_augment = 50
    stdev_noise = 0.3
    
    n_splits = 5
    n_repeats = 10
    n_total = n_splits * n_repeats
    
    regularization = "Lasso"
    Cs = np.logspace(-3, 1, 40)
    
    bar_labels = [
        "Weighted Average Path Delay",
        "Average Connection Probability (Log)",
        "Functional Effect",
        "Weighted Stdev Path Delay",
        "Total Simple Paths",
        "Divergence",
        "Convergence",
        "Ratio Excitatory",
        "Average Path Delay",
        "Stdev Path Delay",
        "Shortest Path Delay",
        "Closeness Centrality",
        "Betweenness Centrality",
        "Harmonic Centrality",
        "Connection Probability Shortest Path",
        "Stdev Connection Probability (Log)",
        "Weighted Functional Effect",
    ]
    
    # Identify nodes that were sensitive
    effectsize_threshold = classification.calc_effectsize_threshold(results, cdf_threshold)
    
    # Construct categories
    # Preferential - 1
    # Nonpreferential - 0
    y, ind_preferential = classification.label_preferential(results, effectsize_threshold)

    # Construct design matrix
    x = classification.create_design_matrix(bar_labels, results, ind_preferential)
    
    # Standardize variables
    x, x_means, x_stdevs = classification.standardize_design_matrix(x)
    
    # Add interactions
    if interaction_flag:
        x, bar_labels = classification.add_interactions(x, bar_labels)
    
    # Add second order
    if n_higher_order > 1:
        x, bar_labels = classification.add_higher_order(n_higher_order, x, bar_labels)
    
    # Augment data
    x, x0, y, y0 = classification.augment_data(x, y, n_augment, stdev_noise)
    
    # Create splits
    train_ind, test_ind = classification.create_splits_logistic(x, y, n_splits, n_repeats)
    
    # Perform logistic regression
    models, scores = classification.logistic_regression(
        x, y, 
        train_ind, test_ind, 
        Cs, n_total, regularization,
        fit_intercept=True,
    )
    
    # Get coefficients
    coefs = np.zeros((len(Cs), x.shape[1]))
    for ii in range(len(models)):
        for split in range(n_total):
            coefs[ii] += np.abs(models[ii][split].coef_[0])
        
        coefs[ii] /= n_total
    
    # Shortening names
    for ii in range(len(bar_labels)):
        if "Node" in bar_labels[ii]:
            bar_labels[ii] = bar_labels[ii].split(" Node")[0]
        if "(Log)" in bar_labels[ii]:
            bar_labels[ii] = bar_labels[ii].split(" (Log)")[0]

    # Sort based on most predictive
    final_nonzero = []
    for ii in range(coefs.shape[1]):
        inds_nonzero = np.arange(len(Cs))[np.flipud(coefs[:, ii]) > 0]
        if len(inds_nonzero) > 0:
            final_nonzero.append(-inds_nonzero[-1])
        else:
            final_nonzero.append(0)

    ind_plot = np.argsort(final_nonzero)

    # Sort based on maximum first coefficient value
    ind_firstval = np.argsort(-coefs[-1, ind_plot])

    # Differentiate linestyle for best two features
    best_features = [ bar_labels[ind_plot[ii]] for ii in range(2) ]

    # Plot coefficients vs regularization strength
    plot_count = 0
    fig = plt.figure(figsize=(5, 4))
    fig.subplots_adjust(bottom=0.13)
    for ii in range(10):
        ind = ind_plot[ii]
        ind_color = ind_firstval[ii]
        coefs_plot = np.flipud(coefs[:, ind])
        if bar_labels[ind] in best_features:
            linestyle = "-"
        else:
            linestyle = "--"
        _=plt.plot(Cs, coefs_plot, linewidth=2, linestyle=linestyle, label=bar_labels[ind], color=cset_muted[plot_count])
        plot_count += 1

    _=plt.legend()
    _=plt.ylabel("Absolute Value of Coefficient")
    _=plt.xlabel("Regularization Strength")
    _=plt.grid()
    _=plt.xlim(Cs[0], Cs[-1])
    _=plt.xscale("log")

    if save_flag:
        fig.savefig("Fig7A_logistic_featureselection.tiff", dpi=600)

    
    plt.show()
