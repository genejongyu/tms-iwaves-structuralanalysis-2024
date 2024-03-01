# Parameters used in analysis of simulations

params = {}

# Window size for analysis of I-waves (ms)
params["trange_iwaves"] = 50 # objective.py

# Voltage value above which simulation will terminate (mV)
params["vthresh_error"] = 1000 # simulation.py

# Offset before time of I1-wave to remaining I-waves (ms)
params["I1wave_offset"] = 0.1 # objective.py

# Offset before time of I1-wave to identify D-wave (ms)
params["Dwave_offset"] = 0.1 # objective.py

# Time after final I-wave to start evaluating post-stimulus synchrony across trials (ms)
params["tpoststim_synchrony"] = 2 # objective.py

# Time after direct activation to look for D-wave
params["tstop_dwave"] = 1 # pso_comp.py

# Time after direct activation to look for I1-wave
params["tstop_I1wave"] = 1.5 # pso_comp.py

# Bandpass filter cutoff frequencies (Hz)
params["bandpass_cutoff_lo"] = 200 # pso_comp.py
params["bandpass_cutoff_hi"] = 2000 # pso_comp.py
