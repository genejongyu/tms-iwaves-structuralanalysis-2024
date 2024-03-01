# Plots probabilities to prefer a corticospinal wave based on selected feature
# for Fig 7D

from os.path import join
from get_paths import *

import classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from TVATAnalysis import TVATAnalysis
from colors_paultol import colors

if __name__ == "__main__":
    save_flag = 0
    
    ## Load data
    fname_regressions = "data_tvat_polynomialregressions.pickle"
    
    fname_settings = {}
    fname_settings["D+"] = "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle"
    fname_settings["D-"] = "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle"
    
    results = TVATAnalysis(dir_data, fname_regressions, fname_settings)
    
    cdf_threshold = 0.96
    
    interaction_flag = 0
    n_higher_order = 1
    intercept_flag = 0
    
    n_augment = 50
    stdev_noise = 0.3
    
    n_splits = 5
    n_repeats = 10
    n_total = n_splits * n_repeats
    
    Cs = np.logspace(1, 5, 10)
    gammas = np.logspace(-4, 1, 10)
    
    cc, gg = np.meshgrid(Cs, gammas)
    metaparams = np.c_[cc.ravel(), gg.ravel()]
    
    max_iter = 50000
    n_jobs = 4
    verbose = False
    
    class_flag = 0
    
    bar_labels = [
        "Shortest Path Delay"
    ]
    
    # Identify nodes that were sensitive
    effectsize_threshold = classification.calc_effectsize_threshold(results, cdf_threshold)
    
    # Construct categories
    # Categories correspond to waves
    y, ind_preferential = classification.label_waves(results, effectsize_threshold)
    
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
    categories = np.arange(len(results.waves))
    train_ind, test_ind = classification.create_splits_svc(x, y, n_splits, n_repeats, categories)
    
    # Perform logistic regression
    models, scores = classification.svc(
        x, y,
        train_ind, test_ind,
        metaparams, n_total, 
    )

    # Choose best model
    scores = np.mean(scores, axis=1)
    ind_best = np.argmax(scores)
    C_best, gamma_best = metaparams[ind_best]
    score_best = scores[ind_best]
    
    print("C_best", C_best)
    print("gamma_best", gamma_best)
    print("Accuracy", score_best)
    
    # Compute probabilities / decision functions
    n_samples = 100
    Xfull = np.linspace(x[:, 0].min(), x[:, 0].max(), n_samples)
    X_plot = Xfull * x_stdevs[0] + x_means[0]
    probabilities = np.zeros((n_samples, len(categories)))
    for split in range(n_total):
        probabilities += models[ind_best][split].predict_proba(Xfull[:, np.newaxis])
    
    probabilities /= n_total
    
    # Plot
    fig_probabilities = plt.figure(figsize=(4, 4))
    fig_probabilities.subplots_adjust(left=0.15, bottom=0.16, right=0.96, top=0.93, wspace=0.5)
    _=plt.title("Corticospinal Wave Probabilities", fontsize=16)
    for ii in range(len(categories)):
        wave = results.waves[ii]
        _=plt.plot(X_plot, probabilities[:, ii], color=colors["waves"][wave], linewidth=2, label=wave)
    
        idx = y == ii
        for line in x[idx, 0][ :: n_augment]:
            _=plt.axvline(line * x_stdevs[0] + x_means[0], color=colors["waves"][wave], linestyle="--", linewidth=1)

    _=plt.legend()
    _=plt.xlabel("Conduction Delay (ms)", fontsize=16)
    _=plt.ylabel("Probability", fontsize=16)
    _=plt.xlim(X_plot[0], X_plot[-1])
    _=plt.grid(axis="x")

    if save_flag:
        fig_probabilities.savefig("Fig7D_svc_validation.tiff", dpi=600)
    
    plt.show()
