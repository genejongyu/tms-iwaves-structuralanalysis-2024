# Functions used to run simulations to evaluate ideal time-step

from neuron import h
import numpy as np
h.load_file('stdrun.hoc')
h.load_file('negative_init.hoc')

#################################
# Custom simulation initializer #
#################################
def initsim(v_init=-65, cvode=1, atol=5e-5, dt=0.025, neg_init=1):
    h.celsius = 36.0
    h.v_init = v_init
    
    if neg_init:
        h.load_file('negative_init.hoc')
        h.negative_init()
    else:
        h.finitialize(v_init)
    
    h.cvode_active(cvode)
    if cvode:
        h.cvode.atol(atol)
    else:
        h.dt = dt
    
    if cvode:
        h.cvode.re_init()
    else:
        h.fcurrent()
    
    h.frecord_init()

################################
# Generate Poisson spike train #
################################
def poisson(rangen, FR, delay, tstop):
    events = []
    mu = 1000./FR
    elapsed = delay
    while elapsed < tstop:
        interval = -mu*np.log(rangen.uniform(0, 1))
        elapsed += interval
        if elapsed < tstop:
            events.append(elapsed)
    
    return events

#####################
# Define objectives #
#####################
def sim_poisson_esser(func_cell, celltype, rangen, FR, delay, tstop, tres, cvode_flag=0):
    cell = func_cell(celltype, 0, 'cn')
    
    spikes = poisson(rangen, FR, delay, tstop)
    vecevents = h.Vector(spikes)
    vecstim = h.VecStim()
    vecstim.play(vecevents)
    
    syn = cell.syn['AMPA']
    syn.tau1 = 0.5
    syn.tau2 = 2.4
    syn.e = 0
    
    nc_input = h.NetCon(vecstim, syn)
    nc_input.weight[0] = 0.1
    nc_input.delay = 0
    
    v = h.Vector()
    v.record(cell.esser_mech._ref_v_iaf, sec=cell.soma)
    
    t = h.Vector()
    t.record(h._ref_t)
    
    nc = h.NetCon(cell.esser_mech._ref_v_iaf, None, sec=cell.soma)
    nc.threshold = -10
    spikes = h.Vector()
    nc.record(spikes)
    
    if cvode_flag:
        initsim(cvode=1, neg_init=1)
    else:
        initsim(cvode=0, dt=tres, neg_init=1)
    
    h.continuerun(tstop)
    
    v = np.array(v)
    t = np.array(t)
    if len(spikes) > 0:
        spikes = np.array(spikes)
    else:
        spikes = np.array([])
    
    return t, v, spikes