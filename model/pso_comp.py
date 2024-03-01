# Functions using inspyred and NEURON to perform PSO

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nhost = comm.Get_size()
except ModuleNotFoundError:
    rank = 0
    nhost = 1

import numpy as np
import h5py
import multiprocessing
import multiprocessing.connection as conn

# If larger data are being plasced in the queue,
# then BUFSIZE will need to be increased
conn.BUFSIZE = 2**24
import time as cookie
import objective
import simulation
import matplotlib.pyplot as plt

############################################
# Generator function to create individuals #
############################################
def net_generator(random, args):
    bound_lo = args["opt_args"]["Bounder"]["lo"]
    bound_hi = args["opt_args"]["Bounder"]["hi"]
    amps = bound_hi - bound_lo
    parameters = args["opt_args"]["parameters"]
    
    chromosome = []
    for ind, param in enumerate(parameters):
        param_type = param[0]
        amp = amps[ind]
        chromosome.append(amp * random.uniform(0, 1) + bound_lo[ind])
    
    return chromosome

#######################################################
# Distributes individuals across cores for evaluation #
#######################################################
def mpi_evaluator(candidates, args):
    opt_args = args["opt_args"]
    
    data = {}
    data["fitness"] = np.zeros(len(candidates))
    data["alleles"] = np.array(candidates)
    for obj in opt_args["objective_names"]:
        if obj in ["peaks_iwave", "tpeaks_iwave", "minima_iwave"]:
            rows, cols = opt_args["targets"][obj].shape
            cols += opt_args["extra_waves"]
            data[obj] = np.zeros((len(candidates), rows * cols))
        else:
            data[obj] = np.zeros((len(candidates), opt_args["targets"][obj].size))

    errors = {}
    for obj in opt_args["objective_names"]:
        errors[obj] = np.zeros((len(candidates), opt_args["targets"][obj].size))

    gen = args["_ec"].num_generations + 1
    print("Gen %i started at %s" % (gen, cookie.strftime("%m/%d-%H:%M:%S")))

    # New method for distributing jobs to properly load balance
    # Distribute initial jobs
    for ii in range(opt_args["nhost"] - 1):
        msg = np.hstack([[ii], candidates[ii]])
        comm.Send(msg, dest=ii + 1, tag=14)
   
    # Timing how much time is spent performing simulations vs message passing
    tstart_jobs = cookie.time()

    # Wait for jobs to complete before sending out next job
    n_sent = opt_args["nhost"] - 1
    n_done = 0
    amps_I1 = np.zeros((len(candidates), opt_args["num_subjects"]))
    while n_done < len(candidates):
        msg = np.empty(
            (len(opt_args["objective_names"]) + 2, opt_args["size_obj_max"]),
            dtype=np.float64,
        )
        comm.Recv(msg, source=MPI.ANY_SOURCE, tag=13)
        ind_data = int(msg[0][0])
        ID_rec = int(msg[0][1])
        for ind_msg, obj in enumerate(opt_args["objective_names"]):
            if obj in ["peaks_iwave", "tpeaks_iwave", "minima_iwave"]:
                rows, cols = opt_args["targets"][obj].shape
                cols += opt_args["extra_waves"]
                data[obj][ind_data] = msg[ind_msg + 1][: rows * cols]
            else:
                data[obj][ind_data] = msg[ind_msg + 1][: opt_args["targets"][obj].size]
        
        amps_I1[ind_data] = msg[-1][:opt_args["num_subjects"]]
        
        if n_sent < len(candidates):
            msg = np.hstack([[n_sent], candidates[n_sent]])
            comm.Send(msg, dest=ID_rec, tag=14)

        n_done += 1
        n_sent += 1

    ttotal = cookie.time() - tstart_jobs
    print("Took %i seconds" % int(ttotal))

    # Compute errors for each objective
    for ii in range(len(candidates)):
        for obj in opt_args["objective_names"]:
            if obj in ["peaks_iwave", "tpeaks_iwave", "minima_iwave"]:
                # Need to reshape data to retrieve only the values that are to be constrained
                num_subj_amp = opt_args["targets"][obj].shape[0]
                props = data[obj][ii].reshape(num_subj_amp, opt_args["total_waves"])
                props = props[:, :opt_args["measured_waves"]]
                
                # For tpeaks_iwave, only calculate error if the objective is nonzero
                if obj == "tpeaks_iwave":
                    mask = opt_args["targets"]["tpeaks_iwave"] == 0
                    props[mask] = 0
                
                errors[obj][ii] = abs_error(
                    opt_args["targets"][obj].ravel(), props.ravel()
                )
            else:
                errors[obj][ii] = abs_error(
                    opt_args["targets"][obj].ravel(), data[obj][ii]
                )
            
            if opt_args["normalization"][obj].shape[0] == 1:
                errors[obj][ii] /= opt_args["normalization"][obj][0]
            else:
                errors[obj][ii] /= opt_args["normalization"][obj]
    
    # Sum weighted errors as fitness
    for ii in range(len(candidates)):
        for obj in opt_args["objective_names"]:
            data["fitness"][ii] += np.sum(
                opt_args["weights"][obj].ravel() * errors[obj][ii]
            )

    # Replace NaNs
    data["fitness"][np.isnan(data["fitness"])] = 1e9
    idx_min = np.argmin(data["fitness"])
    for obj in errors:
        print(obj)
        num_subj_amp = opt_args["targets"][obj].shape[0]
        try:
            props = data[obj][idx_min].reshape(num_subj_amp, opt_args["total_waves"])
            props = props[:, :opt_args["measured_waves"]]
        except ValueError:
            props = data[obj][idx_min]
        string_val = ""
        for val in props.ravel():
            string_val += "%.3f, " % val
        string_target= ""
        for val in opt_args["targets"][obj].ravel():
            string_target += "%.3f, " % val
        print("\tValue: ", string_val[:-1])
        print("\tTarget: ", string_target[:-1])
    
    print("Min error: %.2f\n" % np.min(data["fitness"]))
    
    weights = {}
    for obj in opt_args["objective_names"]:
        weights[obj] = opt_args["weights"][obj]

    if opt_args["regularization_flag"]:
        penalty = calc_regularization(
            candidates, 
            data,
            opt_args,
        )
        data["fitness"] += opt_args["regularization_lambda"] * penalty
    
    # Save data
    save_flag = 1
    while save_flag:
        try:
            with h5py.File(opt_args["fname_data"], "a") as dset:
                dset["fitness"][gen] = data["fitness"]
                dset["alleles"][gen] = data["alleles"]
                
                for obj in opt_args["objective_names"]:
                    # Un-normalize peaks_iwave before saving 
                    if obj in ["peaks_iwave", "minima_iwave"]:
                        rows, cols = opt_args["targets"][obj].shape
                        rows, cols = opt_args["targets"][obj].shape
                        data[obj] = data[obj].reshape(
                            (len(candidates), rows, opt_args["total_waves"])
                        )
                        for subj in range(opt_args["num_subjects"]):
                            istart = opt_args["ind_subj_start"][subj]
                            n_amps = opt_args["num_amps"]["resting"][subj]
                            for ii in range(len(candidates)):
                                data[obj][ii, istart : istart + n_amps] *= amps_I1[ii, subj]
                        
                        dset[obj][gen] = data[obj]
                    else:
                        if obj == "tpeaks_iwave":
                            rows, cols = opt_args["targets"][obj].shape
                            dset[obj][gen] = data[obj].reshape(
                                (len(candidates),  rows, opt_args["total_waves"])
                            )
                        else:
                            dset[obj][gen] = data[obj].reshape(
                                (len(candidates),) + opt_args["targets"][obj].shape
                            )

                if opt_args["adaptive_weighting"]:
                    for obj in weights:
                        dset["weights"][obj][gen] = weights[obj]
                
            save_flag = 0
        except BlockingIOError:
            cookie.sleep(10)
    
    return data["fitness"], weights


#########################################################################
# Queue-based evaluation so that a "fresh" NEURON is launched each time #
#########################################################################
def queue_evaluate(chromosome, opt_args, norm=True, validate=False):
    # Construct stim times
    opt_args["stim_times"] = objective.get_stim_times(
        opt_args["stim_delay"],
        opt_args["num_subjects"],
        opt_args["stim_amps"],
        opt_args["num_trials"],
        opt_args["trial_interval"],
    )

    # Compute simulation tstop for each state
    opt_args["tstop"] = objective.get_tstops(
        opt_args["stim_times"],
        opt_args["num_subjects"],
        opt_args["trial_interval"],
        opt_args["stim_delay"],
    )
    
    # Initialize values for objectives
    props = {}
    for obj in opt_args["objective_names"]:
        if obj in ["peaks_iwave", "tpeaks_iwave", "minima_iwave"]:
            rows, cols = opt_args["targets"][obj].shape
            cols += opt_args["extra_waves"]
            props[obj] = 100 * np.ones((rows, cols))
        else:
            props[obj] = 1000 * np.ones(opt_args["targets"][obj].shape)

    props["amp_I1"] = np.ones(opt_args["num_subjects"])
    
    # Perform simulations
    queue = multiprocessing.Queue()
    sim_results = {}
    sim_results["poprates"] = {}
    sim_results["currents"] = {}
    sim_results["spikes"] = {}
    bad_flag = 0
    for state in opt_args["states"]:
        if not bad_flag:
            p = multiprocessing.Process(
                target=worker,
                args=(queue, chromosome, opt_args, state),
            )
            p.start()
            sim_results["poprates"][state] = queue.get()
            sim_results["currents"][state] = queue.get()
            sim_results["spikes"][state] = {}
            ID = 0
            for celltype in opt_args["cell_types"]:
                if "input" not in celltype:
                    sim_results["spikes"][state][celltype] = {}
                    for ii in range(opt_args["N"][celltype]):
                        sim_results["spikes"][state][celltype][ID] = queue.get()
                        ID += 1

            #sim_results["spikes"][state] = queue.get()
            queue_flag = queue.get()
            if isinstance(queue_flag, int):
                bad_flag = np.max([bad_flag, queue_flag])
            else:
                bad_flag = 1
                print("bad_flag was not an int")
            p.join()
        else:
            if not validate:
                return props
            else:
                return props, sim_results
    
    if bad_flag:
        if not validate:
            return props
        else:
            return props, sim_results
    
    # Check for NaN values in currents
    bad_flag = np.max([bad_flag, objective.check_current_nan(sim_results["currents"])])
    if bad_flag:
        if not validate:
            return props
        else:
            return props, sim_results

    # Smooth poprates to measure TMS response
    sim_results["popsmooths"] = objective.get_filtered_poprates_all(
        sim_results["poprates"],
        opt_args["tres"],
        opt_args["bandpass_cutoff_lo"],
        opt_args["bandpass_cutoff_hi"],
    )
    
    # Get time to first peak in response (D-wave vs I1-wave)
    if "peaks_iwave" in opt_args["targets"]:
        sim_results["t_I1wave"] = objective.get_time_to_I1_wave(
            sim_results["popsmooths"],
            opt_args["tres"],
            opt_args["ind_subj_start"],
            opt_args["num_amps"],
            opt_args["delay_direct"],
            opt_args["tstop_dwave"],
            opt_args["tstop_I1wave"]
        )
    
    # Get chromosome
    sim_results["chromosome"] = chromosome
    
    # Evaluate all objectives
    flag_get_spikes = 0
    for obj in opt_args["objective_names"]:
        if "ISI" in obj:
            flag_get_spikes = 1
            break
        if "firingrate" in obj:
            flag_get_spikes = 1
            break
    if flag_get_spikes:
        sim_results["spikes_resting"] = objective.get_restingstate_spikes(sim_results, opt_args)
    for func_obj in opt_args["functions_obj"]:
        props = func_obj(sim_results, opt_args, props)
    
    if not validate:
        return props
    else:
        return props, sim_results


#####################################
# Worker that is added to the queue #
#####################################
def worker(queue, chromosome, opt_args, state):
    bad_flag = 0
    spikes, currents, bad_flag = simulation.sim_tms_singlepulse(
        chromosome, opt_args, state=state
    )

    # Activate bad flag if no spikes were generated
    n_spikes = 0
    for celltype in spikes:
        for ID in spikes[celltype]:
            if spikes[celltype][ID].size > 0:
                if spikes[celltype][ID].min() < 0:
                    bad_flag = 1
                    break
                else:
                    n_spikes += len(spikes[celltype][ID])

    if n_spikes == 0:
        bad_flag = 1
    
    if not bad_flag:
        # Obtain population rate
        poprates = objective.get_poprates_all(spikes, state, opt_args)
        
        # Combine currents to reduce data size
        celltype = "L5"
        currents_total = np.zeros(len(currents[celltype][0]))
        for icell in range(len(currents[celltype])):
            currents_total += np.array(currents[celltype][icell])

        currents_total /= len(currents[celltype])

        queue.put(poprates)
        queue.put(currents_total)
        for celltype in spikes:
            for ID in spikes[celltype]:
                queue.put(spikes[celltype][ID])
        
    else:
        # Add zeros to queue so that process can terminate
        # Needs to equal the number of queue.put() calls in the previous code block
        num_queue = 2
        for celltype in opt_args["N"]:
            for celltype in spikes:
                num_queue += len(spikes[celltype])
        
        for ii in range(num_queue):
            queue.put(0)

    queue.put(bad_flag)


def relative_error(target, val):
    error = target - val
    idx_nonzero = target != 0
    error[idx_nonzero] /= target[idx_nonzero]
    return np.abs(error)


def squared_error(target, val):
    return (target - val) ** 2


def abs_error(target, val):
    return np.abs(target - val)


###########################################################################
# Subroutine for non-primary workers to receive alleles and evaluate them #
###########################################################################
def evaluate_secondary(opt_args):
    while True:
        msg = np.empty(len(opt_args["parameters"]) + 1, dtype=np.float64)
        comm.Recv(msg, source=0, tag=14)

        # If msg contains nan, then terminate secondary
        if msg[np.isnan(msg)].size > 0:
            break

        index = int(msg[0])
        chromosome = msg[1:]
        props = queue_evaluate(chromosome, opt_args)
        
        msg = np.empty(
            (len(opt_args["objective_names"]) + 2, opt_args["size_obj_max"]),
            dtype=np.float64,
        )
        msg[0][0] = index
        msg[0][1] = opt_args["rank"]
        for ind_msg, obj in enumerate(opt_args["objective_names"]):
            if obj in ["peaks_iwave", "tpeaks_iwave", "minima_iwave"]:
                rows, cols = opt_args["targets"][obj].shape
                cols += opt_args["extra_waves"]
                msg[ind_msg + 1][: rows*cols] = props[obj].ravel()
            else:
                msg[ind_msg + 1][: opt_args["targets"][obj].size] = props[obj].ravel()

        msg[-1][:opt_args["num_subjects"]] = props["amp_I1"]
        
        comm.Send(msg, dest=0, tag=13)

##############################################
# Add regularization penalty term to fitness #
##############################################
def calc_regularization(candidates, data, opt_args):
    idx_param = []
    for param_type in opt_args["regularization_param_types"]:
        for ii in range(len(opt_args["parameters"])):
            if param_type in opt_args["parameters"][ii]:
                idx_param.append(ii)

    if "regularization_groups" in opt_args:
        penalty = np.zeros(len(candidates))
        for idx in opt_args["regularization_groups"]:
            penalty += np.sqrt(len(idx)) * np.sqrt(
                np.sum(data["alleles"][:, idx] ** 2, axis=1)
            )
    else:
        idx_param = np.array(idx_param)
        penalty = np.sum(
            np.abs(data["alleles"][:, idx_param])
            ** opt_args["q"],
            axis=1,
        )

    return penalty
