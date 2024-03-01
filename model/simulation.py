# Functions to run simulations for PSO

from neuron import h
h.load_file("stdrun.hoc")

import numpy as np
import cell_esser
from net_esser import net_esser


#########################################
# Defines and launches a TMS simulation #
#########################################
def sim_tms_singlepulse(chromosome, opt_args, state="resting"):
    seed_con = opt_args["seed_con"]
    seed_tms = opt_args["seed_tms"]
    seed_input = opt_args["seed_input"]

    # Modify params using chromosome
    (
        opt_args_new,
        iclamp_amps,
        iclamp_stdevs,
        p_active,
        afferent_fr,
        syn_noise_weight,
        syn_noise_FR,
        gfluct2,
        delay_jitter,
    ) = parse_chromosome(opt_args, chromosome, state)
   
    # Instantiate network
    net = net_esser(
        opt_args_new, seed_con, seed_tms, seed_input, opt_args_new["tstop"][state], delay_jitter
    )

    # Add current clamps
    net.iclamp(iclamp_amps)

    # Add noise stim
    if "syn_noise_weight" in opt_args["objective_names"]:
        net.noise_stim(syn_noise_weight, syn_noise_FR, state, opt_args_new["tstop"][state], seed_input)
    
    # Apply tms stimulation
    net.tms_stim_div(
        p_active,
        opt_args_new["delay_direct"],
        opt_args_new["delay_std_direct"],
        opt_args_new["v_tms"],
        opt_args_new["stim_times"][state],
    )

    # Record spike times
    tvec = h.Vector()
    idvec = h.Vector()
    for celltype in net.cells:
        if "input" not in celltype:
            for cell in net.cells[celltype]:
                cell.nc_spike.record(tvec, idvec, cell.ID)

    # Record v
    vs = {}
    currents = {}
    for celltype in net.cells:
        if "input" not in celltype:
            vs[celltype] = []
            currents[celltype] = []
            for cell in net.cells[celltype]:
                vs[celltype].append(h.Vector())
                vs[celltype][-1].record(cell.esser_mech._ref_v_iaf)
                currents[celltype].append(h.Vector())
                currents[celltype][-1].record(cell.esser_mech._ref_i_total)
    
    # Run simulation
    net.initsim(cvode=0, dt=opt_args_new["tres"], neg_init=0, state_load=0)
    tfin = 0
    num_runs = int(opt_args_new["tstop"][state] / opt_args_new["tstep"])

    bad_flag = 0
    n_step = int(opt_args_new["tstep"] / opt_args_new["tres"])
    for state in range(num_runs):
        if not bad_flag:
            tfin += opt_args_new["tstep"]
            h.continuerun(tfin)
            # Compute mean cell voltage across entire network
            for celltype in vs:
                for ii in range(len(vs[celltype])):
                    n_nan = np.arange(len(vs[celltype][ii]))[
                        np.isnan(vs[celltype][ii])
                    ].size
                    if n_nan > 0:
                        bad_flag = 1
                        break
                    if (
                        np.mean(vs[celltype][ii].as_numpy())
                        >= opt_args_new["vthresh_error"]
                    ):
                        bad_flag = 1
                        break

                    vs[celltype][ii].resize(0)
    
    # Get spike times
    try:
        tvec = np.array(tvec)
        idvec = np.array(idvec)
    except ValueError:
        tvec = np.array([])
        idvec = np.array([])
        bad_flag = 1

    spikes = {}
    for celltype in net.cells:
        if "input" not in celltype:
            spikes[celltype] = {}
            for ID in net.IDs[celltype]:
                if not bad_flag:
                    spikes[celltype][ID] = tvec[idvec == ID]
                else:
                    spikes[celltype][ID] = np.array([-1])

    return spikes, currents, bad_flag


##########################################
# Applies chromosome parameters to model #
##########################################
def parse_chromosome(opt_args, chromosome, state):
    # Get stim_amps depending on state
    stim_amps = opt_args["stim_amps"][state]

    # Modify opt_args using chromosome
    opt_args_new = opt_args.copy()
    iclamp_amps = {}
    iclamp_stdevs = {}
    p_active = [{} for ii in range(opt_args["num_subjects"])]
    afferent_fr = {}
    syn_noise_weight = {}
    syn_noise_FR = {}
    gfluct2 = {}
    delay_jitter = {}
    flag_logistic = 0
    for ii in range(len(chromosome)):
        if opt_args["parameters"][ii][0] == "strength":
            prop, pre, post, syn = opt_args["parameters"][ii]
            if syn == "AMPA-NMDA":
                opt_args_new["con"]["strength"][pre][post]["AMPA"] *= chromosome[ii]
                opt_args_new["con"]["strength"][pre][post]["NMDA"] *= chromosome[ii]
            else:
                opt_args_new["con"]["strength"][pre][post][syn] *= chromosome[ii]
        elif opt_args["parameters"][ii][0] == "delay":
            prop, pre, post = opt_args["parameters"][ii]
            if pre not in delay_jitter:
                delay_jitter[pre] = {}
            delay_jitter[pre][post] = chromosome[ii]
        elif opt_args["parameters"][ii][0] == "delay_afferent":
            prop, pre, post = opt_args["parameters"][ii]
            opt_args_new["con"]["delay_mean"][pre][post] = chromosome[ii]
        elif opt_args["parameters"][ii][0] == "delay_stdev":
            prop, pre, post = opt_args["parameters"][ii]
            opt_args_new["con"]["delay_std"][pre][post] = chromosome[ii]
        elif "iclamp_mean" in opt_args["parameters"][ii][0]:
            prop, post = opt_args["parameters"][ii]
            iclamp_amps[post] = chromosome[ii]
        elif "iclamp_stdev" in opt_args["parameters"][ii][0]:
            prop, post = opt_args["parameters"][ii]
            if state in prop:
                iclamp_stdevs[post] = chromosome[ii]
        elif "syn_noise_weight" in opt_args["parameters"][ii][0]:
            prop, post = opt_args["parameters"][ii]
            syn_noise_weight[post] = chromosome[ii]
        elif "syn_noise_FR" in opt_args["parameters"][ii][0]:
            prop, post = opt_args["parameters"][ii]
            syn_noise_FR[post] = chromosome[ii]
        elif "p_active" in opt_args["parameters"][ii][0]:
            prop, subj, amp, post = opt_args["parameters"][ii]
            if post not in p_active[subj]:
                p_active[subj][post] = np.zeros(len(stim_amps[subj]))
            
            p_active[subj][post][amp] = chromosome[ii]
        elif "afferent_fr" in opt_args["parameters"][ii][0]:
            prop, post = opt_args["parameters"][ii]
            if state == "isometric":
                afferent_fr[post] = chromosome[ii]
    
    return opt_args_new, iclamp_amps, iclamp_stdevs, p_active, afferent_fr, syn_noise_weight, syn_noise_FR, gfluct2, delay_jitter

