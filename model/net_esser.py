# Class to instantiate Esser model network

from neuron import h
import numpy as np
import cell_esser

cvode_i = h.CVode()
cvode_i.use_fast_imem(1)

class net_esser:
    def __init__(self, params, seed_con, seed_tms, seed_input, tstop, delay_jitter, tstart=0):
        self.params = params
        self.seed_con = seed_con
        self.seed_tms = seed_tms
        self.ncs = []
        self.rangen_con = np.random.RandomState(seed_con)
        self.rangen_tms = np.random.RandomState(seed_tms)

        # Generate column locations
        pos_column = self.gen_pos_circle()
        num_column = pos_column.shape[0]
        
        # Get total cell numbers
        self.params["N"] = {}
        self.N_total = 0
        for celltype in params["cell_types"]:
            if "input" not in celltype:
                num_cell = self.params["N_per_column"][celltype] * num_column
                self.params["N"][celltype] = num_cell
                afferent = celltype + "input"
                self.params["N"][afferent] = num_column#num_cell
                self.N_total += 2 * num_cell
        
        # Create cells
        self.cells = {}
        self.IDs = {}
        ID = 0
        for celltype in params["cell_types"]:
            self.cells[celltype] = []
            self.IDs[celltype] = np.zeros(params["N"][celltype], dtype="int")
            if "input" not in celltype:
                layer = "L" + celltype[1:]
                if layer == "L23":
                    previous_layer = "L1"
                elif layer == "L5":
                    previous_layer = "L23"
                elif layer == "L6":
                    previous_layer = "L5"
                
                count = 0
                for nn in range(num_column):
                    for cc in range(self.params["N_per_column"][celltype]):
                        depth = self.rangen_con.uniform(self.params["depth"][previous_layer], self.params["depth"][layer])
                        loc = np.hstack([pos_column[nn], depth])
                        if params["model_types"][celltype] == "esser":
                            self.cells[celltype].append(cell_esser.cell_esser(celltype, ID, loc))
                        self.IDs[celltype][count] = ID
                        count += 1
                        ID += 1
            else:
                count = 0
                depth = self.params["depth"]["L6"]
                for nn in range(num_column):
                    loc = np.hstack([pos_column[nn], depth])
                    if params["model_types"][celltype] == "poisson":
                        self.cells[celltype].append(
                            cell_esser.poisson(
                                params["firing_rate"][celltype],
                                ID,
                                loc,
                                self.N_total,
                                tstop,
                                seed_input,
                                tstart,
                            )
                        )
                    self.IDs[celltype][count] = ID
                    count += 1
                    ID += 1
        
        # Convert all cell locations into a numpy array for faster connectivity computation
        self.x_locs = {}
        self.y_locs = {}
        self.z_locs = {}
        for celltype in self.cells:
            self.x_locs[celltype] = np.zeros(len(self.cells[celltype]))
            self.y_locs[celltype] = np.zeros(len(self.cells[celltype]))
            self.z_locs[celltype] = np.zeros(len(self.cells[celltype]))
            for icell in range(len(self.cells[celltype])):
                self.x_locs[celltype][icell] = self.cells[celltype][icell].loc[0]
                self.y_locs[celltype][icell] = self.cells[celltype][icell].loc[1]
                self.z_locs[celltype][icell] = self.cells[celltype][icell].loc[2]
        
        # Set synaptic parameters
        for celltype in self.cells:
            if "input" not in celltype:
                for cell in self.cells[celltype]:
                    for syntype in cell.syn:
                        if syntype != "Noise":
                            cell.syn[syntype].tau1 = params["syn"][syntype]["tau1"]
                            cell.syn[syntype].tau2 = params["syn"][syntype]["tau2"]
                            cell.syn[syntype].e = params["syn"][syntype]["e"]

        # Initialize divergence connectivity for point process type
        # inputs that can be activated by tms pulse
        self.div = {}
        for precell in self.cells:
            self.div[precell] = {}
            for ii in range(len(self.cells[precell])):
                preID = self.cells[precell][ii].ID
                self.div[precell][preID] = {}

        self.delays = {}
        for precell in self.cells:
            self.delays[precell] = {}
            for ii in range(len(self.cells[precell])):
                preID = self.cells[precell][ii].ID
                self.delays[precell][preID] = {}
        
        # Connect cells
        for precell in self.cells:
            for postcell in params["con"]["p"][precell]:
                pmax = self.params["con"]["p"][precell][postcell]
                sigma = self.params["con"]["sigma"][precell][postcell]
                for ipre in range(len(self.cells[precell])):
                    preID = self.cells[precell][ipre].ID
                    x_pre = self.cells[precell][ipre].loc[0]
                    y_pre = self.cells[precell][ipre].loc[1]
                    z_pre = self.cells[precell][ipre].loc[2]
                    
                    # Compute Euclidean distance to all postsynaptic cells
                    x_dist = self.x_locs[postcell] - x_pre
                    y_dist = self.y_locs[postcell] - y_pre
                    dist = np.sqrt( x_dist**2 + y_dist ** 2)
                    
                    # Compute probabilities
                    p = pmax * self.gaussian(dist, sigma)
                    
                    # Stochastically determine whether neurons are connected
                    rolls = self.rangen_con.uniform(0, 1, dist.size)
                    ind_connected = np.arange(len(self.cells[postcell]))[rolls < p]

                    if postcell not in self.div[precell][preID]:
                        self.div[precell][preID][postcell] = []
                        self.delays[precell][preID][postcell] = []

                    self.div[precell][preID][postcell] += list(
                        self.IDs[postcell][rolls < p]
                    )

                    for ipost in ind_connected:
                        # Compute transmission delay
                        distance = np.sqrt(np.sum((self.cells[precell][ipre].loc - self.cells[postcell][ipost].loc)**2))
                        if "input" not in celltype:
                            delay = delay_jitter[precell][postcell] * distance / params["conduction_velocity"] + params["syn_delay"]
                        else:
                            delay = distance / params["conduction_velocity"] + params["syn_delay"]

                        self.delays[precell][preID][postcell].append(delay)
                        
                        # Delay cannot be faster than synaptic transmission
                        if delay < params["syn_delay"]:
                            delay = params["syn_delay"]

                        for syntype in params["con"]["syn_types"][precell][postcell]:
                            syn = self.cells[postcell][ipost].syn[syntype]

                            # Connect presynaptic voltage to synapse
                            if params["model_types"][precell] == "esser":
                                nc = h.NetCon(
                                    self.cells[precell][ii].esser_mech._ref_v_iaf,
                                    syn,
                                    sec=self.cells[precell][ii].soma,
                                )
                            elif params["model_types"][precell] == "poisson":
                                nc = h.NetCon(self.cells[precell][ii].vecstim, syn)
                            else:
                                print(
                                    "Error: params['model_types'][precell] must be 'esser' or 'poisson'"
                                )
                            nc.weight[0] = (
                                params["syn"][syntype]["gpeak"]
                                * params["con"]["strength"][precell][postcell][syntype]
                            )
                            nc.delay = delay
                            self.cells[postcell][ipost].ncs.append(nc)
        
        # Convert divergence connectivity lists into numpy arrays
        for precell in self.div:
            for preID in self.div[precell]:
                for postcell in self.div[precell][preID]:
                    self.div[precell][preID][postcell] = np.array(
                        self.div[precell][preID][postcell]
                    )
    
    def gen_pos_circle(self):
        lattice = self.gen_triangular_lattice()
        return self.check_circle(lattice)
    
    def gen_triangular_lattice(self):
        x_dist = self.params["spacing_column"]
        offset = self.params["diameter_column"] / 2
        y_dist = np.sqrt(x_dist**2 - (x_dist / 2)**2)
        y = np.arange(0, self.params["diameter_column"], y_dist) - offset

        x_even = np.arange(0, self.params["diameter_column"], x_dist) - offset
        x_odd = np.arange(x_dist / 2, self.params["diameter_column"], x_dist) - offset
        
        X, Y = np.meshgrid(x_even, x_odd)

        X[1::2] = x_odd
        
        return np.c_[X.ravel(), Y.ravel()]
    
    def check_circle(self, points):
        r = self.params["diameter_column"] / 2
        circle = points[:, 0]**2 + points[:, 1]**2
        idx = circle <= r**2
        return points[idx, :]

    def gaussian(self, distance, sigma):
        return np.exp(-0.5 * (distance / sigma)**2)
    
    def create_syn(self, syntype, cell, tau1=0.5, tau2=2.4, e=0):
        if syntype == "NMDA":
            syn = h.Exp2NMDA_iaf(cell.soma(0.25), sec=cell.soma)
            syn._ref_v_iaf = cell.esser_mech._ref_v_iaf
        else:
            syn = h.Exp2Syn_iaf(cell.soma(0.25), sec=cell.soma)
            syn._ref_v_iaf = cell.esser_mech._ref_v_iaf
        syn.tau1 = tau1
        syn.tau2 = tau2
        syn.e = e
        return syn

    def calc_convergence(self):
        self.conv = {}
        for postcell in self.params["cell_types"]:
            if "input" not in postcell:
                self.conv[postcell] = {}
                for ii in range(self.params["N"][postcell]):
                    postID = self.cells[postcell][ii].ID
                    self.conv[postcell][postID] = {}
                    for precell in self.div:
                        for preID in self.div[precell]:
                            if postcell in self.div[precell][preID]:
                                if postID in self.div[precell][preID][postcell]:
                                    if precell not in self.conv[postcell][postID]:
                                        self.conv[postcell][postID][precell] = []
                                    self.conv[postcell][postID][precell].append(preID)
                    for precell in self.conv[postcell][postID]:
                        self.conv[postcell][postID][precell] = np.array(
                            self.conv[postcell][postID][precell]
                        )

    def tms_stim_div(self, p_active, delay_direct_mean, delay_direct_std, v_tms, stim_times):
        self.vecstims = []
        self.vecevents = []
        self.vecplays = []
        self.timeplays = []
        for subj in stim_times:
            amps, num_trials = stim_times[subj].shape
            for amp in range(amps):
                for trial in range(num_trials):
                    t_stim = stim_times[subj][amp][trial]
                    # Iterate through presynaptic cells
                    for precell in p_active[subj]:  # self.cells:
                        # Obtain probability of activation
                        p = p_active[subj][precell][amp]
                        num_activated = int(round(p * self.params["N"][precell]))
                        idx_active = np.arange(self.params["N"][precell])
                        self.rangen_tms.shuffle(idx_active)
                        # VecStim style activation for point process inputs
                        if "input" in precell:
                            # Create VecStim stimulus for point process inputs
                            self.vecstims.append(h.VecStim())
                            self.vecevents.append(h.Vector([t_stim]))
                            self.vecstims[-1].play(self.vecevents[-1])
                            preIDs_active = self.IDs[precell][idx_active[:num_activated]]
                            for preID in preIDs_active:
                                # Connect VecStim to downstream synapses
                                for postcell in self.div[precell][preID]:
                                    delay_mean = self.params["con"]["delay_mean"][precell][
                                        postcell
                                    ]
                                    delay_std = self.params["con"]["delay_std"][precell][
                                        postcell
                                    ]
                                    for postID in self.div[precell][preID][postcell]:
                                        #delay = delay_mean + np.abs(
                                        #    self.rangen_tms.normal(
                                        #        0, delay_std
                                        #    )
                                        #)
                                        delay = self.rangen_tms.normal(
                                            delay_mean,
                                            delay_std
                                        )
                                        # Make sure delay is not smaller than synaptic transmission
                                        min_delay = 1 + self.params["syn_delay"]
                                        if delay < min_delay:#self.params["tres"]
                                            diff = min_delay - delay
                                            delay = min_delay + diff#self.params["tres"]
                                        idx_cell = np.arange(len(self.IDs[postcell]))[
                                            self.IDs[postcell] == postID
                                        ][0]
                                        for syntype in self.params["con"]["syn_types"][
                                            precell
                                        ][postcell]:
                                            syn = self.cells[postcell][idx_cell].syn[
                                                syntype
                                            ]
                                            nc = h.NetCon(self.vecstims[-1], syn)
                                            nc.delay = delay
                                            nc.weight[0] = (
                                                self.params["syn"][syntype]["gpeak"]
                                                * self.params["con"]["strength"][precell][
                                                    postcell
                                                ][syntype]
                                            )
                                            self.ncs.append(nc)
                        # Vector play style activation for esser model neurons
                        else:
                            # Create Vector that will be played as extracellular
                            # voltages for esser model neurons.
                            # Variability will be given to the input time to represent
                            # uncertainty in where along the neuron the AP is initiated.
                            self.timeplays.append([])
                            self.vecplays.append(h.Vector([v_tms, 0]))
                            
                            delays = delay_direct_mean + np.abs(
                                self.rangen_tms.normal(
                                    0, delay_direct_std, num_activated
                                )
                            )
                            # For direct activation, delay cannot be smaller than time step
                            delays[delays < self.params["tres"]] = self.params["tres"]
                            for idx_delay, idx_cell in enumerate(
                                idx_active[:num_activated]
                            ):
                                self.timeplays[-1].append(
                                    h.Vector(
                                        [
                                            t_stim + delays[idx_delay],
                                            t_stim
                                            + delays[idx_delay]
                                            + 1.5 * self.params["tres"],
                                        ]
                                    )
                                )
                                cell = self.cells[precell][idx_cell]
                                self.vecplays[-1].play(
                                    cell.esser_mech._ref_v_tms,
                                    self.timeplays[-1][idx_delay],
                                )

    def iclamp(self, stim_amps):
        for celltype in stim_amps:
            for cell in self.cells[celltype]:
                cell.esser_mech.i_stim = stim_amps[celltype]

    def noisy_iclamp(self, stim_amps, stim_stdevs, tstop):
        tres = 2* self.params["tres"]
        n = int(tstop / tres)
        time = h.Vector( tres * np.arange(n) )
        for celltype in stim_amps:
            for cell in self.cells[celltype]:
                cell.i_stim_amps = h.Vector(
                    cell.rangen.normal(stim_amps[celltype], stim_stdevs[celltype], n)
                )
                cell.i_stim_amps.play(cell.esser_mech._ref_i_stim, time, True)#self.params["tres"])

    def noise_stim(self, syn_noise_weight, syn_noise_FR, state, tstop, seed_input):
        FR = {}
        FR["L23"] = 3
        FR["L5"] = 5
        FR["L6"] = 5
        FR["I23"] = 10
        FR["I5"] = 15
        FR["I6"] = 15
        for celltype in syn_noise_FR:
            FR[celltype] *= syn_noise_FR[celltype]

        if state == "isometric":    
            for celltype in FR:     
                FR[celltype] = 25
        
        loc = None
        tstart = 0
        self.stim_noise = {}
        self.stim_noise["point_process"] = []
        self.stim_noise["nc"] = []
        for celltype in FR:
            if "input" not in celltype:
                for ii in range(len(self.cells[celltype])):
                    syn = self.cells[celltype][ii].syn["Noise"]
                    self.stim_noise["point_process"].append(
                        cell_esser.poisson(
                            FR[celltype],
                            self.cells[celltype][ii].ID,
                            loc,
                            int(1e5),
                            tstop,
                            seed_input,
                            tstart,
                            noise_flag=1,
                        )
                    )
                    nc = h.NetCon(
                        self.stim_noise["point_process"][-1].vecstim,
                        syn,
                        sec=self.cells[celltype][ii].soma,
                    )
                    nc.weight[0] = syn_noise_weight[celltype]
                    nc.delay = 0
                    self.stim_noise["nc"].append(nc)
    
    #################################
    # Custom simulation initializor #
    #################################
    def initsim(
        self,
        v_init=-65,
        cvode=1,
        atol=5e-5,
        dt=0.025,
        neg_init=1,
        state_load=0,
        rank=0,
        date_time="2022",
    ):
        h.celsius = 36.0
        h.v_init = v_init

        if state_load:
            neg_init = 0

        if neg_init:
            h.load_file("negative_init.hoc")
            h.negative_init()
        else:
            h.finitialize(v_init)
            if state_load:
                svstate = h.SaveState()
                f = h.File(
                    "/work/wmglab/gy42/scratch_neuron/states_%s_%i.dat"
                    % (date_time, rank)
                )
                svstate.fread(f)
                svstate.restore(1)

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
