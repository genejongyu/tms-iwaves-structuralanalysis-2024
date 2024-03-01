# Re-run simulations for particles for D+ and D- runs
# that have similar network strength values to find unified model

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nhost = comm.Get_size()
except ModuleNotFoundError:
    rank = 0
    nhost = 1

from os.path import join
from get_paths import *

import pickle
import pso_comp as evo_comp
from OptimizationData import OptimizationData
import numpy as np
import time as cookie

if __name__ == "__main__":
    fpath_save = join(dir_data, "data_unified_model_search.pickle")

    # Load files
    fname_data1 = [
        "data_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.h5",
        "data_pso_iwaves_column_restingstate_2013_31_120RMT_seed1.h5",
        "data_pso_iwaves_column_restingstate_2013_31_120RMT_seed2.h5",
        "data_pso_iwaves_column_restingstate_2013_31_120RMT_seed3.h5"
    ]
    fname_settings1 = [
        "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle",
        "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed1.pickle",
        "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed2.pickle",
        "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed3.pickle"
    ]
    fname_data2 = [
        "data_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.h5",
        "data_pso_iwaves_column_restingstate_2020_43_120RMT_seed1.h5",
        "data_pso_iwaves_column_restingstate_2020_43_120RMT_seed2.h5",
        "data_pso_iwaves_column_restingstate_2020_43_120RMT_seed3.h5"
    ]
    fname_settings2 = [
        "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle",
        "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed1.pickle",
        "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed2.pickle",
        "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed3.pickle"
    ]
    
    # Instantiate class
    results = {}
    results["D+"] = OptimizationData(
        dir_data, 
        fname_data1, 
        fname_settings1,
    )
    results["D-"] = OptimizationData(
        dir_data, 
        fname_data2, 
        fname_settings2,
    )

    # Find best individual model
    particles_best = {}
    for Dtype in results:
        ind_best = np.argmin(results[Dtype].data["fitness"])
        particles_best[Dtype] = results[Dtype].data["alleles"].reshape(results[Dtype].n_gen * results[Dtype].n_pop, results[Dtype].n_allele)[ind_best]

    # Filter particles for good fitness
    threshold_peaks = 0.1
    threshold_tpeaks = 0.1
    threshold_postmin = 0.1
    fitness = {}
    particles = {}
    for Dtype in results:
        results[Dtype].calc_relative_error()
        idx_good = np.ones(results[Dtype].relative_error["peaks_iwave-D"].size, dtype=bool)
        for obj in results[Dtype].relative_error:
            if obj.startswith("peaks"):
                idx_good *= results[Dtype].relative_error[obj] < threshold_peaks
            if obj.startswith("tpeaks"):
                idx_good *= results[Dtype].relative_error[obj] < threshold_tpeaks
            if obj.startswith("postmin"):
                idx_good *= results[Dtype].relative_error[obj] < threshold_postmin
        idx_good = idx_good.reshape(results[Dtype].n_gen, results[Dtype].n_pop)
        fitness[Dtype] = results[Dtype].data["fitness"][idx_good]
        particles[Dtype] = results[Dtype].data["alleles"][idx_good]
    
    # Find the D+ and D- model pair that has the most similar strengths
    ind_end_strength = 38
    pair_best = (0, 0)
    similarity_best = 1000 * np.ones(ind_end_strength)
    for ii in range(fitness["D+"].size):
        for jj in range(fitness["D-"].size):
            similarity = np.abs((particles["D+"][ii, :ind_end_strength] - particles["D-"][jj, :ind_end_strength]) / particles["D+"][ii, :ind_end_strength])
            similarity += np.abs((particles["D+"][ii, :ind_end_strength] - particles["D-"][jj, :ind_end_strength]) / particles["D-"][jj, :ind_end_strength])
            similarity *= 0.5
            if similarity.sum() < similarity_best.sum():
                similarity_best = similarity
                pair_best = (ii, jj)

    particles_similar = {}
    particles_similar["D+"] = particles["D+"][pair_best[0]]
    particles_similar["D-"] = particles["D-"][pair_best[1]]

    # Average particles that aren't p_active for L5 PC
    n_search = 5
    weights = np.linspace(0, 1, n_search)
    particles_ave_similar = {}
    particles_ave_best = {}
    for weight1 in weights:
        weight2 = 1 - weight1
        weight1_key = np.round(weight1, 1)
        weight2_key = np.round(weight2, 1)
        particles_ave_similar[weight1_key] = {}
        particles_ave_best[weight1_key] = {}
        for Dtype in results:
            particles_ave_similar[weight1_key][Dtype] = weight1 * particles_similar["D+"] + weight2 * particles_similar["D-"]
            particles_ave_best[weight1_key][Dtype] = weight1 * particles_best["D+"] + weight2 * particles_best["D-"]
            for ind_p in range(len(results[Dtype].params["parameters"])):
                if "p_active" in results[Dtype].params["parameters"][ind_p][0]:
                    if results[Dtype].params["parameters"][ind_p][-1] == "L5":
                        particles_ave_similar[weight1_key][Dtype][ind_p] = particles_similar[Dtype][ind_p]
                        particles_ave_best[weight1_key][Dtype][ind_p] = particles_best[Dtype]][ind_p]
        
    # Run simulations
    sim_results = {}
    sim_results["similar"] = {}
    sim_results["best"] = {}
    ST = cookie.time()
    for weight in particles_ave_best:
        print(weight)
        sim_results["similar"][weight] = {}
        sim_results["best"][weight] = {}
        for Dtype in particles_ave_best[weight]:
            props, sim_results["similar"][weight][Dtype] = evo_comp.queue_evaluate(
                particles_ave_similar[weight][Dtype], results[Dtype].params, validate=True
            )
            props, sim_results["best"][weight][Dtype] = evo_comp.queue_evaluate(
                particles_ave_best[weight][Dtype], results[Dtype].params, validate=True
            )
    
    with open(fpath_save, "wb") as f:
        pickle.dump(sim_results, f)
    
    print("Took {} seconds".format(cookie.time() - ST))
