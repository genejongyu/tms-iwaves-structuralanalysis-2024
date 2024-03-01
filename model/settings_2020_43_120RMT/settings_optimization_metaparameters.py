# Metaparameters for optimization

params = {}

# Base 2 power to generate Sobol samples
params["base2_power"] = 9 # ec.py

# Number of particles
params["n_particles"] = 2 ** params["base2_power"] # run_pso.py

# Number of iterations
params["n_gen"] = 1000 # run_pso.py

# Total number of models evaluated
params["max_eval"] = params["n_particles"] * params["n_gen"] # run_pso.py

# Size of neighborhood
if params["n_particles"] >= 40:
    params["neighborhood_size"] = params["n_particles"] // 10 # run_pso.py
else:
    params["neighborhood_size"] = 2

