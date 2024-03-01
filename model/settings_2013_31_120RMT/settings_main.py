#########################################
# Script for importing major parameters #
#########################################

from numpy import array
from os.path import join
dir_data = join("..", "data")
from time import strftime
import sys

import settings_model_connectivity
import settings_model_neurons
import settings_model_synapse
import settings_optimization_analysis
import settings_optimization_metaparameters
import settings_optimization_objectives
import settings_optimization_parameterbounds
import settings_optimization_regularization
import settings_optimization_targets
import settings_random_seeds
import settings_simulation
import settings_stimulation

# Check to see if a key currently exists in the dictionary before adding the key
def check_duplicate(dict_main, dict_new, dict_new_name):
    for key in dict_new:
        if key not in dict_main:
            dict_main[key] = dict_new[key]
        else:
            raise Exception("Key {} exists!\n Double-check {} to make sure keys are unique across all settings files!".format(
                    key, dict_new_name
                )
            )

    return dict_main

# Parse arguments
if len(sys.argv) > 2:
    settings_random_seeds.params["seed_evo"] = int(sys.argv[2])

# Build parameters dictionary
params = {}

# Set file names
params["data_dir"] = dir_data
params["date_time"] = strftime("%Y%m%d-%H%M")
params["fname_root"] = "pso_iwaves_2013_31_120RMT_seed%i" % settings_random_seeds.params["seed_evo"]
params["fname_data"] = join(
    params["data_dir"],
    "data_%s_%s.h5" % (params["fname_root"], params["date_time"])
)
params["fname_settings"] = join(
    params["data_dir"],
    "settings_%s_%s.pickle" % (params["fname_root"], params["date_time"])
)
# Load settings
params = check_duplicate(params, settings_model_connectivity.params, "settings_model_connectivity.py")
params = check_duplicate(params, settings_model_neurons.params, "settings_model_neurons.py")
params = check_duplicate(params, settings_model_synapse.params, "settings_model_synapse.py")
params = check_duplicate(params, settings_optimization_analysis.params, "settings_optimization_analysis.py")
params = check_duplicate(params, settings_optimization_metaparameters.params, "settings_optimization_metaparameters.py")
params = check_duplicate(params, settings_optimization_objectives.params, "settings_optimization_objectives.py")
params = check_duplicate(params, settings_optimization_targets.params, "settings_optimization_targets.py")
params = check_duplicate(params, settings_optimization_parameterbounds.params, "settings_optimization_parameterbounds.py")
params = check_duplicate(params, settings_optimization_regularization.params, "settings_optimization_regularization.py")
params = check_duplicate(params, settings_random_seeds.params, "settings_random_seeds.py")
params = check_duplicate(params, settings_simulation.params, "settings_simulation.py")
params = check_duplicate(params, settings_stimulation.params, "settings_stimulation.py")

for key in ["Bounder", "parameters", "save_keys"]:
    if key in params:
        raise Exception("Bounder, parameters, and save_keys should not be in settings files! They are defined here!")

# Define open parameters for optimization
# Create lower and upper bound array for inspyred's Bounder class
params["Bounder"] = {}
params["Bounder"]["lo"] = []
params["Bounder"]["hi"] = []

# Get parameter keys for connectivity strength
params["parameters"] = []
for pre in params["con"]["strength"]:
    for post in params["con"]["strength"][pre]:
        if "AMPA" in params["con"]["strength"][pre][post]:
            syntypes = ["AMPA-NMDA"]
        elif "GABAA" in params["con"]["strength"][pre][post]:
            syntypes = ["GABAA", "GABAB"]

        for syn in syntypes:
            params["parameters"].append(["strength", pre, post, syn])
            pname = ""
            for p in params["parameters"][-1]:
                pname += p
            
            if pname in params["bounds"]:
                params["Bounder"]["lo"].append(
                    params["bounds"][pname][0]
                )
                params["Bounder"]["hi"].append(
                    params["bounds"][pname][1]
                )
            else:
                params["Bounder"]["lo"].append(
                    params["bounds"]["strength"][0]
                )
                params["Bounder"]["hi"].append(
                    params["bounds"]["strength"][1]
                )

# Get parameter keys for circuit delays
for pre in params["con"]["delay_mean"]:
    if "input" not in pre:
        for post in params["con"]["strength"][pre]:
            params["parameters"].append(["delay", pre, post])
            if pname in params["bounds"]:
                params["Bounder"]["lo"].append(
                    params["bounds"][pname][0]
                )
                params["Bounder"]["hi"].append(
                    params["bounds"][pname][1]
                )
            else:
                params["Bounder"]["lo"].append(
                    params["bounds"]["delay"][0]
                )
                params["Bounder"]["hi"].append(
                    params["bounds"]["delay"][1]
                )

# Get parameter keys for afferent delays
for pre in params["con"]["delay_mean"]:
    if "input" in pre:
        for post in params["con"]["strength"][pre]:
            params["parameters"].append(["delay_afferent", pre, post])
            pname = ""
            for p in params["parameters"][-1]:
                pname += p
            
            if pname in params["bounds"]:
                params["Bounder"]["lo"].append(
                    params["bounds"][pname][0]
                )
                params["Bounder"]["hi"].append(
                    params["bounds"][pname][1]
                )
            else:
                params["Bounder"]["lo"].append(
                    params["bounds"]["delay_afferent"][0]
                )
                params["Bounder"]["hi"].append(
                    params["bounds"]["delay_afferent"][1]
                )

# Get parameter keys for afferent delay standard deviations
for pre in params["con"]["delay_mean"]:
    if "input" in pre:
        for post in params["con"]["strength"][pre]:
            params["parameters"].append(["delay_stdev", pre, post])
            pname = ""
            for p in params["parameters"][-1]:
                pname += p
            
            if pname in params["bounds"]:
                params["Bounder"]["lo"].append(
                    params["bounds"][pname][0]
                )
                params["Bounder"]["hi"].append(
                    params["bounds"][pname][1]
                )
            else:
                params["Bounder"]["lo"].append(
                    params["bounds"]["delay_stdev"][0]
                    )
                params["Bounder"]["hi"].append(
                    params["bounds"]["delay_stdev"][1]
                )

# Get parameter keys for p_active
for subj in range(params['num_subjects']):
    for amp in range(params['num_amps']['resting'][subj]):
        param_name = 'p_active_subject%i-amp%i' % (subj, amp)
        for celltype in params['cell_types']:
            params['parameters'].append([param_name, subj, amp, celltype])
            params['Bounder']['lo'].append(params['bounds']["p_active"][0])
            params['Bounder']['hi'].append(params['bounds']["p_active"][1])

# Get parameter keys for noise stim
for celltype in params["cell_types"]:
    if "input" not in celltype:
        params["parameters"].append(["syn_noise_weight", celltype])
        params["Bounder"]["lo"].append(params["bounds"]["syn_noise_weight"][0])
        params["Bounder"]["hi"].append(params["bounds"]["syn_noise_weight"][1])

# Get parameter keys for syn_noise FR
for celltype in params["cell_types"]:
    if "input" not in celltype:
        params["parameters"].append(["syn_noise_FR", celltype])
        params["Bounder"]["lo"].append(params["bounds"]["syn_noise_FR"][0])
        params["Bounder"]["hi"].append(params["bounds"]["syn_noise_FR"][1])

# Get parameter keys for iclamp
for celltype in params["cell_types"]:
    if "input" not in celltype:
        params["parameters"].append(["iclamp_mean", celltype])
        params["Bounder"]["lo"].append(params["bounds"]["iclamp_mean"][0])
        params["Bounder"]["hi"].append(params["bounds"]["iclamp_mean"][1])

params["Bounder"]["lo"] = array(params["Bounder"]["lo"])
params["Bounder"]["hi"] = array(params["Bounder"]["hi"])

# Reformatting params for saving into hdf5
params["save_keys"] = []
for ii in range(len(params["parameters"])):
    key = ""
    for jj in range(len(params["parameters"][ii])):
        try:
            key += params["parameters"][ii][jj] + "-"
        except TypeError:
            key += str(params["parameters"][ii][jj]) + "-"
    
    params["save_keys"].append(key[:-1])

params["save_keys"] = array(params["save_keys"]).astype("S")
