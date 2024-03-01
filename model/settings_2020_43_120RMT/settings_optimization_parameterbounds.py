# Open parameters for optimization and bounds

##################################################
# Choose which parameters are going to optimized #
##################################################
# Specified here because params_net needs to be loaded first
params = {}

# Set lower and upper bounds for parameter types (lo, hi)
params["bounds"] = {}

# Network parameters
params["bounds"]["strength"] = (0.1, 10)
params["bounds"]["delay"] = (0.25, 2)
params["bounds"]["delay_afferent"] = (1.25, 2.5)
params["bounds"]["delay_stdev"] = (0.1, 1)

# Activation parameters
params["bounds"]["p_active"] = (1e-5, 1)

# Resting state parameters
params["bounds"]["iclamp_mean"] = (-300, 100)

# Noise stim parameters
params["bounds"]["syn_noise_weight"] = (1, 50)
params["bounds"]["syn_noise_FR"] = (0, 1)
