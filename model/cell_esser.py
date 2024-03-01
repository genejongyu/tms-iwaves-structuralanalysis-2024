# Classes used in esser network model
# Class for a point neuron with integrate-and-fire dynamics from Esser 2005
# Class to generate Poisson distributed ISIs for afferents

from neuron import h
import numpy as np

class cell_esser:
    def __init__(self, celltype, ID, loc):
        # Store parameters
        self.celltype = celltype
        self.ID = ID
        self.loc = loc # units: microns
        self.rangen = np.random.default_rng(seed=ID)
        self.ncs = []

        # Create section
        self.soma = h.Section(name=celltype + "-" + str(ID))
        self.soma.nseg = 1

        # Insert esser mechanism
        self.esser_mech = h.esser_mech_pp(0.5, sec=self.soma)
        self.esser_mech.ena_iaf = 30
        self.esser_mech.ek_iaf = -90

        # Create synapses
        self.syn = {}
        self.syn["AMPA"] = self.create_syn("AMPA")
        self.syn["NMDA"] = self.create_syn("NMDA")
        self.syn["GABAA"] = self.create_syn("GABAA")
        self.syn["GABAB"] = self.create_syn("GABAB")
        self.esser_mech._ref_i_ampa = self.syn["AMPA"]._ref_i
        self.esser_mech._ref_i_nmda = self.syn["NMDA"]._ref_i
        self.esser_mech._ref_i_gabaa = self.syn["GABAA"]._ref_i
        self.esser_mech._ref_i_gabab = self.syn["GABAB"]._ref_i

        # Spontaneous activity generator
        self.syn["Noise"] = self.create_syn("Noise")
        self.esser_mech._ref_i_noise = self.syn["Noise"]._ref_i
        self.syn["Noise"].e = 0
        self.syn["Noise"].tau1 = 0.1
        self.syn["Noise"].tau2 = 0.2
        
        if celltype.startswith("L"):
            if celltype == "L5":
                self.esser_mech.v_iaf0 = -78.33
                self.esser_mech.theta_eq = -53
                self.esser_mech.tau_theta = 0.5
                self.esser_mech.tau_spike = 0.6
                self.esser_mech.tau_m = 13
                self.esser_mech.gNa_leak = 0.14
                self.esser_mech.gK_leak = 1.3
                self.esser_mech.tspike = 0.75
            else:
                self.esser_mech.v_iaf0 = -75.263
                self.esser_mech.theta_eq = -53
                self.esser_mech.tau_theta = 2
                self.esser_mech.tau_spike = 1.75
                self.esser_mech.tau_m = 15
                self.esser_mech.gNa_leak = 0.14
                self.esser_mech.gK_leak = 1.0
                self.esser_mech.tspike = 2.0
        elif celltype.startswith("I"):
            self.esser_mech.v_iaf0 = -70.0
            self.esser_mech.theta_eq = -54
            self.esser_mech.tau_theta = 1.0
            self.esser_mech.tau_spike = 0.48
            self.esser_mech.tau_m = 7
            self.esser_mech.gNa_leak = 0.2
            self.esser_mech.gK_leak = 1.0
            self.esser_mech.tspike = 0.75
        else:
            print("Error: celltype must be L23, L5, L6, or I##")

        # Create spike generator
        self.nc_spike = h.NetCon(
            self.esser_mech._ref_v_iaf, None, 0, 0, 0, sec=self.soma
        )

    def create_syn(self, syntype):
        if syntype == "NMDA":
            syn = h.Exp2NMDA_iaf(self.soma(0.5), sec=self.soma)
        else:
            syn = h.Exp2Syn_iaf(self.soma(0.5), sec=self.soma)
        syn._ref_v_iaf = self.esser_mech._ref_v_iaf
        return syn


class poisson:
    def __init__(self, firing_rate, ID, loc, N_total, tstop, seed_input, tstart, noise_flag=0):
        self.firing_rate = firing_rate
        self.ID = ID
        self.loc = loc
        seed = ID + seed_input * N_total
        self.rangen = np.random.default_rng(seed=seed)
        self.vecstim = h.VecStim()
        self.vecevents = h.Vector()
        if firing_rate > 0:
            events = []
            mu = 1000.0 / firing_rate
            if not noise_flag:
                delay = self.rangen.uniform(0, 500)
                elapsed = delay
            else:
                elapsed = 0
            while elapsed < tstop:
                interval = -mu * np.log(self.rangen.uniform(0, 1))
                elapsed += interval
                events.append(elapsed)

            for spike in events[:-1]:
                if spike > tstart:
                    self.vecevents.append(spike)
        self.vecstim.play(self.vecevents)
