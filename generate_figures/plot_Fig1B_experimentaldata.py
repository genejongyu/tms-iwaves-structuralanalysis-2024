# Plots the individual trials and trial average of the epidural recordings
# used in Fig 1B

from os.path import join
from get_paths import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from colors_paultol import colors
plt.rcParams["mathtext.default"] = "regular"

save_flag = 0

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    y = signal.sosfiltfilt(sos, data)
    return y

# Bandpass parameters
cutoff_lo = 200  # Hz
cutoff_hi = 1500  # Hz

# Load data
dir_data = "../data"

subjects = ["2013_31", "2020_43"]
data_raw = {}
data_raw["2013_31"] = loadmat(join(dir_data, "s2013_031_3ch.mat"))
data_raw["2020_43"] = loadmat(join(dir_data, "s2020_043_3ch.mat"))

# Set names of channels
name_PA = {}
name_PA["2013_31"] = ["f001_wave_data", "f000_wave_data"]
name_PA["2020_43"] = ["f002_wave_data", "f000_wave_data", "f001_wave_data"]

# Set stimulation intensity labels based on channel name
labels = {}
labels["2013_31"] = {}
labels["2013_31"]["f001_wave_data"] = "PA 110% RMT"
labels["2013_31"]["f000_wave_data"] = "PA 120% RMT"

labels["2020_43"] = {}
labels["2020_43"]["f002_wave_data"] = "PA 100% RMT"
labels["2020_43"]["f000_wave_data"] = "PA 120% RMT"
labels["2020_43"]["f001_wave_data"] = "PA 140% RMT"

# Time-to-peak of D-wave
t_dwave = {}
t_dwave["2013_31"] = 2.8 # ms
t_dwave["2020_43"] = 2.9 # ms

# Channel index
channel = {}
channel["2013_31"] = 1
channel["2020_43"] = 1

# Stimulation parameters
delay_tms = 30 # ms
tstart = 32 # ms

# Load relevant data
data = {}
artifact = {}
n_trials = {}
Fs = {}
t_full = {}
t = {}
t_artifact = {}
n_samples = {}
for subject in name_PA:
    data[subject] = {}
    artifact[subject] = {}
    n_trials[subject] = {}
    Fs[subject] = {}
    t_full[subject] = {}
    t[subject] = {}
    t_artifact[subject] = {}
    n_samples[subject] = {}
    for fname in name_PA[subject]:
        Fs_flag = 0
        Fs[subject][fname] = int(round(1 / data_raw[subject][fname][0][0][3][0][0]))
        if Fs[subject][fname] == 5000:
            Fs[subject][fname] = 10000
            n_samples[subject][fname] = 2 * data_raw[subject][fname][-1][-1][-1].shape[0] - 1
            t_old = 1000 / 5000 * np.arange(data_raw[subject][fname][-1][-1][-1].shape[0])
            Fs_flag = 1
        else: 
            n_samples[subject][fname] = data_raw[subject][fname][-1][-1][-1].shape[0]
        t_full[subject][fname] = 1000 / Fs[subject][fname] * np.arange(n_samples[subject][fname])
        idx_t = t_full[subject][fname] >= tstart
        t[subject][fname] = t_full[subject][fname][idx_t]
        n_trials[subject][fname] = data_raw[subject][fname][-1][-1][-1].shape[2]
        data[subject][fname] = np.zeros((n_trials[subject][fname], len(t[subject][fname])))
        for trial in range(n_trials[subject][fname]):
            if Fs_flag:
                trial_data = data_raw[subject][fname][-1][-1][-1][
                    :, channel[subject], trial
                ]
                t_new = np.linspace(t_full[subject][fname][0], t_full[subject][fname][-1], )
                data[subject][fname][trial] = np.interp(t_full[subject][fname], t_old, trial_data)[idx_t]
            else:
                data[subject][fname][trial] = data_raw[subject][fname][-1][-1][-1][
                    :, channel[subject], trial
                ][idx_t]
        
        idx_t = t_full[subject][fname] >= delay_tms - 1
        t_artifact[subject][fname] = t_full[subject][fname][idx_t]
        artifact[subject][fname] = np.zeros((n_trials[subject][fname], len(t_artifact[subject][fname])))
        for trial in range(n_trials[subject][fname]):
            if Fs_flag:
                trial_data = data_raw[subject][fname][-1][-1][-1][
                    :, channel[subject], trial
                ]
                t_new = np.linspace(t_full[subject][fname][0], t_full[subject][fname][-1], )
                artifact[subject][fname][trial] = np.interp(t_full[subject][fname], t_old, trial_data)[idx_t]
            else:
                artifact[subject][fname][trial] = data_raw[subject][fname][-1][-1][-1][
                    :, channel[subject], trial
                ][idx_t]

# Filter the signal
filtered = {}
for subject in name_PA:
    filtered[subject] = {}
    for fname in name_PA[subject]:
        filtered[subject][fname] = np.zeros(data[subject][fname].shape)
        for trial in range(n_trials[subject][fname]):
            filtered[subject][fname][trial] = butter_bandpass_filter(
                data[subject][fname][trial], cutoff_lo, cutoff_hi, Fs[subject][fname]
            )

# Parameters for plotting
# Name of channel based on subject
fname_plot = {}
fname_plot["2013_31"] = "f000_wave_data"
fname_plot["2020_43"] = "f000_wave_data"

# Get I1-wave amplitudes for normalization
tstart_I1 = {}
tstart_I1["2013_31"] = 1.5
tstart_I1["2020_43"] = 0
norm_factor = {}
for subject in subjects:
    fname = fname_plot[subject]
    idx = (t[subject][fname] - tstart) > tstart_I1[subject]
    norm_factor[subject] = filtered[subject][fname].mean(axis=0)[idx].max()

# Y-axis bounds and tick locations
ylims = {}
ylims["2013_31"] = (-15, 40)
ylims["2020_43"] = (-9, 9)

yticks = {}
yticks["2013_31"] = [-10, 0, 10, 20, 30, 40]
yticks["2020_43"] = [-9, -5, 0, 5, 9]

# Plot including artifact
fig_artifact = {}
for subject in subjects:
    fname = fname_plot[subject]
    fig_artifact[subject] = plt.figure(figsize=(8, 4))
    fig_artifact[subject].subplots_adjust(left=0.13, right=0.97, bottom=0.2, top=0.9, hspace=0.4)
    for ii in range(n_trials[subject][fname]):
        _ = plt.plot(
            t_artifact[subject][fname] - delay_tms,
            artifact[subject][fname][ii],
            color=colors["popsmooth"][subject],
            linewidth=0.5,
            alpha=0.75,
        )
    _ = plt.plot(
        t_artifact[subject][fname] - delay_tms,
        artifact[subject][fname].mean(axis=0),
        color="k",
        linewidth=2,
    )
    _ = plt.xlim(-1, 9)
    _=plt.ylim(ylims[subject][0], ylims[subject][1])
    _=plt.grid(True)
    _=plt.yticks(yticks[subject], fontsize=16)
    _=plt.xticks([0, 2, 4, 6, 8, 10], fontsize=16)
    _=plt.ylabel(r"Voltage ($\mu$V)", fontsize=16)
    _=plt.xlabel("Time After Stimulus (ms)", fontsize=16)

# Plot bandpass filtered signale
fig_filtered = {}
for subject in subjects:
    fname = fname_plot[subject]
    idx_t = t_artifact[subject][fname] >= tstart
    fig_filtered[subject] = plt.figure(figsize=(4.8, 4))
    fig_filtered[subject].subplots_adjust(left=0.2, right=0.97, bottom=0.2, top=0.9, hspace=0.4)
    for ii in range(n_trials[subject][fname]):
        _ = plt.plot(
            t_artifact[subject][fname][idx_t] - delay_tms,
            filtered[subject][fname][ii],
            color=colors["popsmooth"][subject],
            linewidth=0.5,
            alpha=0.75,
        )
    _ = plt.plot(
        t_artifact[subject][fname][idx_t] - delay_tms,
        filtered[subject][fname].mean(axis=0),
        color="k",
        linewidth=2,
    )
    _ = plt.xlim(2, 8.5)
    _=plt.ylim(ylims[subject][0], ylims[subject][1])
    _=plt.grid(True)
    _=plt.yticks(yticks[subject], fontsize=16)
    _=plt.xticks([2, 4, 6, 8], fontsize=16)
    _=plt.ylabel(r"Voltage ($\mu$V)", fontsize=16)
    _=plt.xlabel("Time After Stimulus (ms)", fontsize=16)


if save_flag:
    for subject in fig_filtered:
        fig_artifact[subject].savefig("Fig1B_experimental_withartifact_{}.tiff".format(subject), dpi=600)
        fig_filtered[subject].savefig("Fig1B_experimental_filtered_{}.tiff".format(subject), dpi=600)

plt.show()