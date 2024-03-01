# Script to launch particle swarm optimization
# Requires an argument that is the directory with the settings files that specifies objectives/constraints
# Ex: settings_2013_31_120RMT, settings_2020_43_120RMT

# Must be run with at least 2 processors

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from os import getcwd
from os.path import join
import sys

cwd = getcwd()
dir_params = sys.argv[1]
sys.path.append(join(cwd, dir_params))

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nhost = comm.Get_size()
if rank == 0:
    print("Running %s" % dir_params)

import numpy as np
import time as cookie
import inspyred
import inspyred_swarm_modified
import h5py
import pso_comp
import settings_main
import pickle
import objective

#################################
# Instantiate and run algorithm #
#################################
if __name__ == "__main__":
    settings_main.params["nhost"] = nhost
    settings_main.params["rank"] = rank
    
    # Choose evolution settings
    rangen = np.random.default_rng(seed=settings_main.params["seed_evo"])
    algorithm = inspyred_swarm_modified.PSO(rangen)
    algorithm.terminator = inspyred.ec.terminators.evaluation_termination
    algorithm.topology = inspyred.swarm.topologies.ring_topology
    bounder = inspyred.ec.Bounder(
        settings_main.params["Bounder"]["lo"],
        settings_main.params["Bounder"]["hi"],
    )

    # Start optimization if rank is 0, else await orders
    if rank == 0:
        # Initialize file for saving data
        with h5py.File(settings_main.params["fname_data"], "w") as dset:
            dset["fname_settings"] = settings_main.params["fname_settings"]
            dset["alleles"] = np.zeros(
                (settings_main.params["n_gen"], settings_main.params["n_particles"], len(settings_main.params["parameters"]))
            )
            dset["fitness"] = np.zeros((settings_main.params["n_gen"], settings_main.params["n_particles"]))
            for obj in settings_main.params["objective_names"]:
                if obj in ["peaks_iwave", "tpeaks_iwave", "minima_iwave"]:
                    rows, cols = settings_main.params["targets"][obj].shape
                    cols += settings_main.params["extra_waves"]
                    dset[obj] = np.zeros(
                        (settings_main.params["n_gen"], settings_main.params["n_particles"], rows, cols)
                    )
                else:
                    dset[obj] = np.zeros(
                        (settings_main.params["n_gen"], settings_main.params["n_particles"])
                        + settings_main.params["targets"][obj].shape
                    )
        
        # Save params to file
        with open(settings_main.params["fname_settings"], "wb") as f:
            pickle.dump(settings_main.params, f)
        
        # Get objective functions that need to be run to evaluate particles
        # Functions added after saving to avoid pickling function dependencies
        objective_function_names = list(set(settings_main.params["objective_functions"].values()))
        settings_main.params["functions_obj"] = []
        for objective_name in objective_function_names:
            settings_main.params["functions_obj"].append(objective.__dict__[objective_name])
        
        # Start optimization
        ST = cookie.time()
        final_pop = algorithm.evolve(
            generator=pso_comp.net_generator,
            evaluator=pso_comp.mpi_evaluator,
            pop_size=settings_main.params["n_particles"],
            maximize=False,
            bounder=bounder,
            opt_args=settings_main.params,
            neighborhood_size=settings_main.params["neighborhood_size"],
            max_evaluations=settings_main.params["max_eval"],
            n_gen=settings_main.params["n_gen"],
        )

        # Terminate secondaries
        for dest in range(1, nhost):
            comm.Send(
                np.nan * np.ones(len(settings_main.params["parameters"]) + 1),
                dest=dest,
                tag=14,
            )

        # Print elapsed time
        ET = cookie.time() - ST
        print("Finished at %s" % cookie.strftime("%m/%d-%H:%M:%S"))
        h = int(ET / 3600)
        m = int(np.ceil((ET / 3600 - h) * 60))
        print("Took %ih %im" % (h, m))
    else:
        # If not primary process, wait to receive parameter values
        pso_comp.evaluate_secondary(settings_main.params)
