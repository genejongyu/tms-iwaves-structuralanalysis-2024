# Parameters for evolution
from numpy import array, zeros, ones, cumsum
from copy import deepcopy
params = {}

# Number of subjects
params["num_subjects"] = 1

# Does or does not have D-wave (0-no, 1-yes)
params["D+"] = [1]

# Total number of waves (D-wave + I-waves)
params["measured_waves"] = 4

# Extra I-waves for future data analysis
params["extra_waves"] = 0
params["total_waves"] = params["measured_waves"] + params["extra_waves"]

# Stimulation amplitudes (%)
params["stim_amps"] = {}
params["stim_amps"]["resting"] = [
    [120], 
]
assert len(params["stim_amps"]["resting"]) == params["num_subjects"], \
    "Need to specify a list of amplitudes for each subject"

# Number of stimulation amplitudes
params["num_amps"] = {}
for state in params["stim_amps"]:
    params["num_amps"][state] = array(
        [len(amps) for amps in params["stim_amps"][state]]
    )

# Index of I1-wave (should equal 1 regardless of D-wave or no D-wave)
params["ind_I1"] = 1

# Index for stimulus amplitude that elicited the largest I1-wave
params["ind_subj_start"] = [0] + list(cumsum(params["num_amps"]["resting"][:-1]))
params["ind_I1max_amp"] = [0]
assert len(params["ind_I1max_amp"]) == params["num_subjects"], \
    "Need to specify ind of max amp for each subject"

# Targets for the objectives
# Why not format targets as (num_subjects, num_amps, n_targets)
# instead of (num_subjects*num_amps, n_targets)?
# Is it bad that I want to keep the array nice and tidy?
params["targets"] = {}
params["targets"]["peaks_iwave"] = array(
    [
        [1.452, 0.989, 0.487, 0.852]
    ]
) / 0.989
params["targets"]["minima_iwave"] = array(
    [
        [1.034, 0.616, 0.669, 0.487]
    ]
) / 0.989
params["targets"]["tpeaks_iwave"] = array(
    [
        [-1.300, 0.000, 1.500, 2.800]
    ]
)
params["targets"]["tminima_iwave"] = array(
    [
        [-0.6, 0.6, 2.2, 3.5]
    ]
)
params["targets"]["postmin_iwave"] = zeros(params["targets"]["peaks_iwave"].shape[0])
params["targets"]["firingrate_mean"] = array([3, 5, 5, 10, 15, 15])
params["targets"]["ISI_mean"] = array([333.33, 200., 200., 100., 66.67, 66.67])
params["targets"]["ISI_std_across_cell"] = params["targets"]["ISI_mean"] ** 0.5
params["targets"]["syn_noise_weight"] = array([0, 0, 0, 0, 0, 0])
params["targets"]["iclamp_mean"] = array([0, 0, 0, 0, 0, 0])
params["targets"]["response_cv"] = array([0])

# Size of largest target vector
params["size_obj_max"] = max(
    [params["targets"][obj].size for obj in params["targets"]]
)
params["size_obj_max"] += params["targets"]["peaks_iwave"].shape[0] * params["extra_waves"]

# Normalization factors for objectives
params["normalization"] = {}
for obj in params["targets"]:
    params["normalization"][obj] = ones(params["targets"][obj].shape)
    
    idx = params["targets"][obj] > 0
    if params["targets"][obj][idx].size > 0:
        params["normalization"][obj][idx] = params["targets"][obj][idx]

for obj in params["normalization"]:
    if "iwave" in obj:
        params["normalization"][obj][:] = 1

# Initial weights for objectives
params["weights"] = {}
params["weights"]["peaks_iwave"] = 5 * ones(
    params["targets"]["peaks_iwave"].shape
)
params["weights"]["peaks_iwave"][:, 3] = 10
params["weights"]["minima_iwave"] = 3 * ones(
    params["targets"]["minima_iwave"].shape
)
params["weights"]["tpeaks_iwave"] = 6.25 * ones(
    params["targets"]["tpeaks_iwave"].shape
)
params["weights"]["tminima_iwave"] = 5 * ones(
    params["targets"]["tminima_iwave"].shape
)
params["weights"]["postmin_iwave"] = 10 * ones(
    params["targets"]["postmin_iwave"].shape
)
params["weights"]["firingrate_mean"] = ones(
    params["targets"]["firingrate_mean"].shape
)
params["weights"]["ISI_mean"] = ones(
    params["targets"]["ISI_mean"].shape
)
params["weights"]["ISI_std_across_cell"] = ones(
    params["targets"]["ISI_std_across_cell"].shape
)
params["weights"]["syn_noise_weight"] = 0.01 * ones(
    params["targets"]["syn_noise_weight"].shape
)
params["weights"]["iclamp_mean"] = 0.05 * ones(
    params["targets"]["iclamp_mean"].shape
)
params["weights"]["response_cv"] = 2 * ones(
    params["targets"]["response_cv"].shape
)
