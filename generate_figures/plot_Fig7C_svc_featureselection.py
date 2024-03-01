# Plots probability of winning recursive feature selection to classify
# preferred corticospinal wave for Fig 7C

from os.path import join
from get_paths import *

import pickle
import matplotlib.pyplot as plt
import numpy as np
from colors_paultol import *

save_flag = 0

## Load data
fname = "data_svc_recursivefeatureelimination.pickle"
fpath = join(dir_data, fname)
with open(fpath, "rb") as f:
    data = pickle.load(f)

## Place data into lists for sorting/plotting
features = []
probabilities = []
for feature, p in data["p"].items():
    features.append(feature)
    probabilities.append(p)

## Sort data
ind_sort = np.argsort(probabilities)

features = [features[ii] for ii in ind_sort]
probabilities = [probabilities[ii] for ii in ind_sort]
features.reverse()
probabilities.reverse()

## Shortening feature names
for ii in range(len(features)):
    if "Node" in features[ii]:
        features[ii] = features[ii].split(" Node")[0]
    if "(Log)" in features[ii]:
        features[ii] = features[ii].split(" (Log)")[0]
    if "Probability" in features[ii]:
        features[ii] = features[ii].replace("Probability", "Prob.")
    if "Connection" in features[ii]:
        features[ii] = features[ii].replace("Connection", "Connect.")
    if "Average" in features[ii]:
        features[ii] = features[ii].replace("Average", "Ave.")

## Plot data
width = 0.5
x_bar = np.arange(len(probabilities))
fig_bar = plt.figure(figsize=(4, 4))
fig_bar.subplots_adjust(left=0.16, right=0.98, bottom=0.55, top=0.97)
plt.bar(x_bar, probabilities, width, color=cset_bright[4])
plt.xticks(x_bar, features, rotation=90)
plt.ylabel("Probability of Selection")
plt.grid(axis="y")

if save_flag:
    fig_bar.savefig("Fig7C_svc_featureselection.tiff", dpi=600)

plt.show()