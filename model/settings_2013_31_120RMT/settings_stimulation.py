# Parameters for stimulation
from numpy import array

params = {}

# Stimulation times (ms)
params["trial_interval"] = {}
params["trial_interval"]["resting"] = 800
params["num_trials"] = 3

# Time at which stimulus is applied (ms)
params["stim_delay"] = 2000

# Time at which to start steady-state analysis
params["analysis_tstart"] = 500

# Strength of applied voltage via TMS pulse (mV)
params["v_tms"] = 10
params["i_tms"] = 1

# Delay between stimulus and direct activation (ms)
params["delay_direct"] = 1
params["delay_std_direct"] = 0.1

# Delay from tms pulse to synaptic activation (ms)
params["syn_delay"] = 0.5
