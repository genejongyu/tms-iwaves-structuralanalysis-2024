# Parameters for evolution
import sys
sys.path.append("..")

params = {}

# States to be simulated
params["states"] = ["resting"]

# Objectives used to evaluate fitness - pso_comp.py
params["objective_names"] = [
    "peaks_iwave",
    "minima_iwave",
    "tpeaks_iwave",
    "tminima_iwave",
    "postmin_iwave",
    "firingrate_mean",
    "ISI_mean",
    "ISI_std_across_cell",
    "syn_noise_weight",
    "iclamp_mean",
    "response_cv",
]

# Functions for evaluating objectives - pso_comp.py
params["objective_functions"] = {
    "peaks_iwave": "calc_props_tms_acute",
    "minima_iwave": "calc_props_tms_acute",
    "tpeaks_iwave": "calc_props_tms_acute",
    "tminima_iwave": "calc_props_tms_acute",
    "postmin_iwave": "calc_props_tms_acute",
    "firingrate_mean": "calc_firingrates_mean",
    "ISI_mean": "calc_ISIs_mean",
    "ISI_std_across_cell": "calc_across_cell_ISI_std",
    "syn_noise_weight": "calc_syn_noise_weight",
    "iclamp_mean": "calc_iclamp_mean",
    "response_cv": "calc_response_cv",
}
