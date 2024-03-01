# Functions for performing classification of prefential parameters and
# preferred corticospinal wave

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
np.random.seed(149)

# Identify nodes that were sensitive
def calc_effectsize_threshold(results, cdf_threshold):
    tms_effectsize = np.zeros(results.n_celltypes)
    for ind, celltype in enumerate(results.celltypes):
        for ii in range(len(results.parameters)):
            if results.parameters[ii].startswith("TMS"):
                if results.parameters[ii].endswith(celltype):
                    tms_effectsize[ind] = results.combined_sensitivity["D+"][ii]
    
    ind_cdf = np.argsort(-tms_effectsize)
    tms_cdf = np.cumsum(tms_effectsize[ind_cdf])
    tms_cdf /= tms_cdf.max()
    celltypes_sorted = [results.celltypes[ii] for ii in ind_cdf]
    if tms_effectsize[ind_cdf][tms_cdf > cdf_threshold].size > 0:
        effectsize_threshold = tms_effectsize[ind_cdf][tms_cdf > cdf_threshold].max()
    else:
        effectsize_threshold = 0

    return effectsize_threshold
    
# Construct categories
# Preferential - 1
# Nonpreferential - 0
def label_preferential(results, effectsize_threshold):
    ind_preferential = np.zeros(results.n_celltypes, dtype=bool)
    y = []
    for ind, celltype in enumerate(results.celltypes):
        for ii in range(len(results.parameters)):
            if results.parameters[ii].startswith("TMS"):
                if results.parameters[ii].endswith(celltype):
                    if results.combined_sensitivity["D+"][ii] > effectsize_threshold: 
                        ind_preferential[ind] = True
                        y.append(0)
                        for wave in results.selective["D+"]:
                            if results.selective["D+"][wave][ii] > 0:
                                y[-1] = 1
                                break

    y = np.array(y)
    return y, ind_preferential

def label_waves(results, effectsize_threshold):
    ind_preferential = np.zeros(len(results.celltypes), dtype=bool)
    y = []
    for ind_celltype, celltype in enumerate(results.celltypes):
        for ii in range(len(results.parameters)):
            if results.parameters[ii].startswith("TMS"):
                if results.parameters[ii].endswith(celltype):
                    if results.combined_sensitivity["D+"][ii] > effectsize_threshold:
                        for ind_wave, wave in enumerate(results.selective["D+"]):
                            if results.selective["D+"][wave][ii] > 0:
                                ind_preferential[ind_celltype] = True
                                y.append(ind_wave)

    y = np.array(y)
    return y, ind_preferential

# Construct design matrix    
def create_design_matrix(bar_labels, results, ind_preferential):
    n_col = len(bar_labels)
    n_orig = len(bar_labels)
    x = np.zeros((results.n_celltypes, n_col))
    for ind, feature in enumerate(bar_labels):
        if feature in ["Connection Probability Shortest Path"]:
            x[:, ind] = np.log(results.features["Node"][feature])
        elif "Centrality" in feature:
            x_tmp = results.features["Node"][feature].copy()
            x_tmp[x_tmp == 0] = 1e-1
            x[:, ind] = np.log(x_tmp)
        elif feature == "Functional Effect":
            x[:, ind] = results.features["Node"]["Weighted Functional Effect"]
            x[:, ind][x[:, ind] > 0] = 1
            x[:, ind][x[:, ind] < 0] = -1
        else:
            x[:, ind] = results.features["Node"][feature]
    
    return x[ind_preferential]
    
# Standardize variables
def standardize_design_matrix(x):
    x_means = np.zeros(x.shape[1])
    x_stdevs = np.zeros(x.shape[1])
    for ii in range(x.shape[1]):
        x_means[ii] = np.mean(x[:, ii])
        x_stdevs[ii] = np.std(x[:, ii])
        x[:, ii] -= x_means[ii]
        x[:, ii] /= x_stdevs[ii]

    return x, x_means, x_stdevs
    
# Add interactions
def add_interactions(x, bar_labels):
    x_interactions = []
    for ii in range(x.shape[1] - 1):
        for jj in range(ii + 1, x.shape[1]):
            interaction_label = bar_labels[ii] + " x " + bar_labels[jj]
            if "Excitatory Node" not in interaction_label:
                x_interactions.append(x[:, ii] * x[:, jj])
                bar_labels.append(interaction_label)

    x = np.hstack([x, np.array(x_interactions).T])
    return x, bar_labels
    
# Add second order
def add_higher_order(n_higher_order, x, bar_labels):
    for higher_order in range(2, n_higher_order + 1):
        x_higher_order = []
        for ii in range(n_orig):
            if bar_labels[ii] not in ["Excitatory Node", "Inhibitory Node", "Weighted Excitatory", "Circuit"]:
                x_higher_order.append(x[:, ii] ** higher_order)
                bar_labels.append(bar_labels[ii] + "^{}".format(higher_order))

        x = np.hstack([x, np.array(x_higher_order).T])

    return x, bar_labels
    
# Augment data
def augment_data(x, y, n_augment, stdev_noise):
    y0 = y.copy()
    x0 = x.copy()
    y = np.repeat(y, n_augment)
    x = np.repeat(x, n_augment, axis=0)
    for ii in range(x.shape[0]):
        if ii % n_augment != 0:
            x[ii] += np.random.normal(0, stdev_noise, x.shape[1])

    return x, x0, y, y0

# Create splits
def create_splits_logistic(x, y, n_splits, n_repeats):
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    train_ind = []
    test_ind = []
    for (train_index, test_index) in skf.split(x, y):
        train_ind.append(train_index)
        test_ind.append(test_index)

    return train_ind, test_ind

def create_splits_svc(x, y, n_splits, n_repeats, categories):
    N = y.size
    categories_dict = {}
    for category in categories:
        categories_dict[category] = np.arange(N)[y == category]
    
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    train_ind = []
    test_ind = []
    for (train_index, test_index) in skf.split(x, y):
        train_ind.append(train_index)
        test_ind.append(test_index)

    return train_ind, test_ind

# Perform logistic regression
def logistic_regression(x, y, train_ind, test_ind, Cs, n_total, regularization, random_state=1, fit_intercept=False, max_iter=500000, n_jobs=4):
    models = []
    scores = np.zeros((len(Cs), n_total))
    for ind, C in enumerate(Cs):
        models.append([])
        for split in range(n_total):
            if regularization == "Lasso":
                models[ind].append(LogisticRegression(
                    penalty="l1",
                    C=C,
                    class_weight="balanced",
                    multi_class="ovr",
                    solver="liblinear",
                    max_iter=max_iter,
                    fit_intercept=fit_intercept,
                    verbose=False,
                    random_state=random_state,
                    tol=1e-6,
                ).fit(x[train_ind[split]], y[train_ind[split]]))
            elif regularization == "Ridge":
                models[ind].append(LogisticRegression(
                    penalty="l2",
                    C=C,
                    class_weight="balanced",
                    multi_class="ovr",
                    solver="lbfgs",
                    max_iter=max_iter,
                    fit_intercept=fit_intercept,
                    verbose=False,
                    random_state=random_state,
                    tol=1e-6,
                ).fit(x[train_ind[split]], y[train_ind[split]]))

            # Score based on augmented data
            scores[ind][split] = models[ind][split].score(x[test_ind[split]], y[test_ind[split]])

    return models, scores
    
def svc(x, y, train_ind, test_ind, metaparams, n_total, random_state=1, fit_intercept=False, max_iter=500000, n_jobs=4):
    models = []
    scores = np.zeros((metaparams.shape[0], n_total))
    for ind in range(metaparams.shape[0]):
        C, gamma = metaparams[ind]
        gamma = gamma / x.shape[0]
        models.append([])
        for split in range(n_total):
            models[ind].append(SVC(
                C=C,
                gamma=gamma,
                kernel="rbf",
                class_weight="balanced",
                max_iter=max_iter,
                verbose=False,
                random_state=random_state,
                probability=True, 
            ).fit(x[train_ind[split]], y[train_ind[split]]))

            # Score based on augmented data
            scores[ind][split] = models[ind][split].score(x[test_ind[split]], y[test_ind[split]])

    return models, scores
