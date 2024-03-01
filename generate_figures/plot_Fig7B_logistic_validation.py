# Plots probabilities/decision boundary to classify preferential parameters
# for Fig 7B

from os.path import join
from get_paths import *

import classification
import numpy as np
import matplotlib.pyplot as plt
from TVATAnalysis import TVATAnalysis
from colors_paultol import cmap_iridescent as cmap
import matplotlib as mpl

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
    
    regularization = "Ridge"
    Cs = np.logspace(-4, 4, 40)
    
    bar_labels = [
        "Weighted Functional Effect",
        "Average Connection Probability (Log)",
    ]
    
    # Identify nodes that were sensitive
    effectsize_threshold = classification.calc_effectsize_threshold(results, cdf_threshold)
    
    # Construct categories
    # Construct categories
    # Preferential - 1
    # Nonpreferential - 0
    categories = ["Nonpreferential", "Preferential"]
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

    # Choose best model
    scores = np.mean(scores, axis=1)
    ind_best = np.argmax(scores)
    C_best = Cs[ind_best]
    score_best = scores[ind_best]

    print("C_best", C_best)
    print("Accuracy", score_best)
    
    # Compute probabilities / decision functions
    n_samples = 1000
    ind1 = 0
    ind2 = 1
    xx = np.linspace(x[:, ind1].min(), x[:, ind1].max(), n_samples)
    yy = np.linspace(x[:, ind2].min(), x[:, ind2].max(), n_samples)
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    N = len(y)
    probabilities = np.zeros((n_samples * n_samples, len(categories)))
    X_plot = np.c_[x[:, ind1], x[:, ind2]]
    yhat = np.zeros(y[ :: n_augment].size)
    for split in range(n_total):
        models[ind_best][split].fit(X_plot, y)
        probabilities += models[ind_best][split].predict_proba(Xfull)
    
    probabilities /= n_total
    probabilities[probabilities <= 0.5] = 0
    probabilities[probabilities > 0.5] = 1
    yhat /= n_total
    
    # Custom color map
    num_colors = 3
    bounds = np.linspace(0, 1, num_colors)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Shortened bar labels
    short_labels = []
    for label in bar_labels:
        if label == "Average Connection Probability (Log)":
            short_labels.append("Ave. Connection Prob. (Log)")
        else:
            short_labels.append(label)

    # Plot
    xmin = x[:, ind1].min()
    xmax = x[:, ind1].max()
    ymin = x[:, ind2].min()
    ymax = x[:, ind2].max()
    fig_probabilities = plt.figure(figsize=(8, 4))
    fig_probabilities.subplots_adjust(left=0.1, right=0.92, bottom=0.13, top=0.92, hspace=0.5)
    for ii in range(len(categories)):
        ind_category = 1 - ii
        _=plt.subplot(1, len(categories), ii + 1)
        _=plt.title(categories[ind_category], fontsize=16)
        _=plt.pcolormesh(
            np.linspace(xmin, xmax, n_samples),
            np.linspace(ymin, ymax, n_samples),
            probabilities[:, ind_category].reshape(n_samples, n_samples),
            shading="gouraud",
            cmap=cmap
        )
        x_orig = x0[:, ind1]
        y_orig = x0[:, ind2]
        idx = y[ :: n_augment] == ind_category
        _=plt.scatter(x_orig[idx], y_orig[idx], marker="o", c="k", edgecolor="w")
        _=plt.scatter(x_orig[~idx], y_orig[~idx], marker="o", c="w", edgecolor="k")
        _=plt.xlabel(bar_labels[ind1], fontsize=16)
        if ii == 0:
            _=plt.ylabel(short_labels[ind2], fontsize=16)
        
        ax_cb = fig_probabilities.add_axes([0.94, 0.13, 0.025, 0.79])
        cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, spacing="proportional", ticks=[], boundaries=bounds)

    if save_flag:    
        fig_probabilities.savefig("Fig7B_logistic_validation.tiff", dpi=600)
    
    plt.show()
