# Script to run recursive feature elimination for feature selection for
# classification of corticospinal wave preference

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nhost = comm.Get_size()
except ModuleNotFoundError:
    rank = 0
    nhost = 1

import sys
from os.path import join
from get_paths import *

import classification
import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from TVATAnalysis import TVATAnalysis
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if __name__ == "__main__":
    fpath_save = join(dir_data, "data_svc_recursivefeatureelimination.pickle")

    np.random.seed(rank)

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
    
    Cs = np.logspace(-4, 4, 20)
    gammas = np.logspace(-3, 3, 20)
    
    cc, gg = np.meshgrid(Cs, gammas)
    metaparams = np.c_[cc.ravel(), gg.ravel()]
    
    max_iter = 10000
    n_jobs = 4
    verbose = False
    
    class_flag = 0

    num_feature_shuffles = 500
    n_features = 5

    bar_labels = [
        "Convergence",
        "Divergence",
        "Total Simple Paths",
        "Connection Probability Shortest Path",
        "Average Connection Probability (Log)",
        "Stdev Connection Probability (Log)",
        "Shortest Path Delay",
        "Average Path Delay",
        "Weighted Average Path Delay",
        "Stdev Path Delay",
        "Weighted Stdev Path Delay",
        "Functional Effect",
        "Weighted Functional Effect",
        "Closeness Centrality",
        "Betweenness Centrality",
        "Harmonic Centrality"
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
    
    # Perform recursive feature elimination
    # Need to shuffle feature order
    scores = {}
    feature_rank = {}
    winner_count = {label: 0 for label in bar_labels}
    n_count = {label: 0 for label in bar_labels}
    for shuffle_count in range(rank, num_feature_shuffles, nhost):
        if rank == 0:
            print("Shuffle: {}".format(shuffle_count))

        ind_bar_shuffle = np.arange(len(bar_labels))
        np.random.shuffle(ind_bar_shuffle)
        ind_bar_shuffle = ind_bar_shuffle[:n_features]
        shuffled_labels = [bar_labels[ind] for ind in ind_bar_shuffle]
        for label in shuffled_labels:
            n_count[label] += 1
        
        scores[shuffle_count] = {}
        feature_rank[shuffle_count] = {}
        for label in shuffled_labels:
            feature_rank[shuffle_count][label] = (True, -1)
        
        idx_labels = np.ones(n_features, dtype=bool)
        remaining_labels = shuffled_labels.copy()
        loop = 0
        while np.sum(idx_labels) > 1:
            ST_loop = cookie.time()
            if rank == 0:
                print("\tLoop: {}".format(loop))
            
            # Augment data
            x_aug, x, y_aug, y = classification.augment_data(x, y, n_augment, stdev_noise)
            
            # Create splits
            categories = np.arange(len(results.waves))
            train_ind, test_ind = classification.create_splits_svc(x_aug, y_aug, n_splits, n_repeats, categories)

            # Shuffle features and get selected features
            x_shuffle = x_aug[:, ind_bar_shuffle]
            x_test = x_shuffle[:, idx_labels]

            # Evaluate models while removing each feature once
            scores[shuffle_count][loop] = np.zeros((len(remaining_labels), metaparams.shape[0], n_total))
            for ii in range(len(remaining_labels)):
                idx_test = np.ones(len(remaining_labels), dtype=bool)
                idx_test[ii] = False
                for ind in range(metaparams.shape[0]):
                    C, gamma = metaparams[ind]
                    gamma = gamma / x_test.shape[0]
                    for split in range(n_total):
                        model = SVC(
                            C=C,
                            gamma=gamma,
                            kernel="rbf",
                            class_weight="balanced",
                            max_iter=max_iter,
                            verbose=verbose,
                            random_state=1,
                            probability=True, 
                        ).fit(x_test[train_ind[split]][:, idx_test], y_aug[train_ind[split]])
                        scores[shuffle_count][loop][ii][ind][split] = model.score(x_test[test_ind[split]][:, idx_test], y_aug[test_ind[split]])
            
            scores[shuffle_count][loop] = np.mean(scores[shuffle_count][loop], axis=2)
            scores[shuffle_count][loop] = np.max(scores[shuffle_count][loop], axis=1)

            # Find feature that caused lowest decrease in accuracy after removal
            ind_worst = np.argmax(scores[shuffle_count][loop])
            removed_label = remaining_labels[ind_worst]
            feature_rank[shuffle_count][removed_label] = (False, len(bar_labels) - loop)

            # Eliminate worst feature
            remaining_labels = []
            for ind, label in enumerate(shuffled_labels):
                if feature_rank[shuffle_count][label][0] == False:
                    idx_labels[ind] = False
                else:
                    remaining_labels.append(label)
            
            if rank == 0:
                print("\tLoop {} took: {} seconds".format(loop, cookie.time() - ST_loop))
            loop += 1
        
        if rank == 0:
            print("\tWinner: {}".format(remaining_labels[0]))
            print("Shuffle {} took: {} seconds".format(shuffle_count, cookie.time() - ST_shuffle))
        winner_count[remaining_labels[0]] += 1
    
    comm.Barrier()
    
    # Collect data
    if nhost > 1:
        # Collecting data
        if rank == 0:
            for send_rank in range(1, nhost):
                msg = np.empty(2 * len(bar_labels), dtype=int)
                comm.Recv(msg, source=send_rank, tag=13)
                for ii in range(len(bar_labels)):
                    label = bar_labels[ii]
                    winner_count[label] += msg[ii]
                    n_count[label] += msg[ii + len(bar_labels)]
        # Sending data
        else:
            msg = np.empty(2 * len(bar_labels), dtype=int)
            for ind, label in enumerate(bar_labels):
                msg[ind] = winner_count[label]
                msg[ind + len(bar_labels)] = n_count[label]
            
            comm.Send(msg, dest=0, tag=13)
    
    comm.Barrier()

    if rank == 0:
        p = {}
        for label in n_count:
            p[label] = winner_count[label] / n_count[label]
        
        save_data = {}
        save_data["p"] = p
        save_data["winner_count"] = winner_count
        save_data["n_count"] = n_count
        with open(fpath_save, "wb") as f:
            pickle.dump(save_data, f)
        
        for label in p:
            print(label, p[label])
