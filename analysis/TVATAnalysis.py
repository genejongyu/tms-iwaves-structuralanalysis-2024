# Class to load and analyze TVAT sensitivity data

from os.path import join
import pickle
import h5py
import numpy as np
import networkx as nx

class TVATAnalysis:
    def __init__(self, dir_data, fname_sensitivity, fname_settings):
        ## Loading data and variable initialization
        self.dir_data = dir_data
        self.fname_sensitivity = fname_sensitivity
        self.fname_settings = fname_settings
        
        # Load sensitivities from TVAT regressions
        with open(join(dir_data, fname_sensitivity), "rb") as f:
            self.data = pickle.load(f)

        self.parameters_abbrev = self.data["parameters"]
        self.particle_best = self.data["particle_best"]
        
        # Load settings for optimization
        self.params = {}
        for Dtype in fname_settings:
            with open(join(dir_data, fname_settings[Dtype]), "rb") as f:
                self.params[Dtype] = pickle.load(f)
        
        # Number of self.parameters
        self.n_parameters = len(self.parameters_abbrev)
        
        # Extract data from TVAT regressions
        self.coefs = {}
        self.rsquared = {}
        if "multiply" in fname_sensitivity:
            for Dtype in self.data["coefs"][()]:
                self.coefs[Dtype] = self.data["coefs"][()][Dtype]
                self.rsquared[Dtype] = self.data["R^2"][()][Dtype]
        else:
            for Dtype in self.data["coefs"]:
                self.coefs[Dtype] = self.data["coefs"][Dtype]
                self.rsquared[Dtype] = self.data["R^2"][Dtype]
        
        self.parameter_crosses = self.data["parameter_crosses"]
        self.parameters_resorted = self.data["parameters_resorted"]
        self.inds_resorted = self.data["inds_resorted"]
        
        # Replace PC with IT or PTN
        for ii in range(len(self.parameters_resorted)):
            pre, post = self.parameters_resorted[ii].split("-")
            if "INT" in pre:
                pre = pre.replace("INT", "BC")
            if "INT" in post:
                post = post.replace("INT", "BC")
            if "PC" in pre:
                if pre.startswith("L5"):
                    pre = pre.replace("PC", "PTN")
                else:
                    pre = pre.replace("PC", "IT")
            if "PC" in post:
                if post.startswith("L5"):
                    post = post.replace("PC", "PTN")
                else:
                    post = post.replace("PC", "IT")
            self.parameters_resorted[ii] = pre + "-" + post
        
        self.sensitivity = {}
        for Dtype in self.coefs:
            self.sensitivity[Dtype] = {}
            for wave in self.coefs[Dtype]:
                self.sensitivity[Dtype][wave] = np.abs(self.coefs[Dtype][wave])
        
        # Convert celltype names to proper naming conventions
        self.celltypes_abbrev = ["L23input", "L23", "I23input", "I23", "L5input", "L5", "I5input", "I5", "L6input", "L6", "I6input", "I6"]
        self.n_celltypes = len(self.celltypes_abbrev)
        
        self.labels = {}
        for ii in range(len(self.celltypes_abbrev)):
            label = self.celltypes_abbrev[ii]
            if "23" in label:
                split_name = label.split("23")
                label = split_name[0] + "2/3"
                if len(split_name) > 1:
                    label += split_name[1]
            
            if label.startswith("L"):
                label += " PC"
            
            if label.startswith("I"):
                label= "L" + label[1:]
                label += " INT"
            
            if "input" in label:
                label = label.replace("input", "")
                label += " AFF"
            
            self.labels[self.celltypes_abbrev[ii]] = label
        
        self.celltypes = [self.labels[celltype] for celltype in self.celltypes_abbrev]
        self.activations = ["TMS-"+celltype for celltype in self.celltypes]
        self.parameters = []
        for ii in range(len(self.parameters_abbrev)):
            if "-" in self.parameters_abbrev[ii]:
                cell1, cell2 = self.parameters_abbrev[ii].split("-")
                self.parameters.append(self.labels[cell1] + "-" + self.labels[cell2])
            else:
                self.parameters.append("TMS-" + self.labels[self.parameters_abbrev[ii]])
        
        ## Get/compute features for regression
        self.features = {}
        self.features["Node"] = {}
        
        # Get outbound outbound number for each cell type
        con = self.params[Dtype]["con"]
        con_p = {}
        self.features["Node"]["Divergence"] = np.zeros(self.n_celltypes, dtype=int)
        for ind, celltype in enumerate(self.celltypes_abbrev):
            self.features["Node"]["Divergence"][ind] = len(con["p"][celltype])
            con_p[self.labels[celltype]] = {}
            for postcell in con["p"][celltype]:
                con_p[self.labels[celltype]][self.labels[postcell]] = con["p"][celltype][postcell]
        
        self.features["Node"]["Convergence"] = np.zeros(self.n_celltypes)
        for ind, postcell in enumerate(self.celltypes_abbrev):
            self.features["Node"]["Convergence"][ind] = 0
            for precell in con["p"]:
                if postcell in con["p"][precell]:
                    self.features["Node"]["Convergence"][ind] += 1

        ## Get delay jitter
        self.delay_jitter = {}
        for ii in range(len(self.params[Dtype]["parameters"])):
            if self.params[Dtype]["parameters"][ii][0] == "delay":
                ptype, pre, post= self.params[Dtype]["parameters"][ii]
                pre_label = self.labels[pre]
                post_label = self.labels[post]
                if pre_label not in self.delay_jitter:
                    self.delay_jitter[pre_label] = {}
                
                self.delay_jitter[pre_label][post_label] = self.particle_best[Dtype][ii]
        
        ## Generate locations
        locs = {}
        for celltype in self.params[Dtype]["cell_types"]:
            if "input" not in celltype:
                pos_layer = self.gen_pos_circle(self.params[Dtype]["N"][celltype], self.params[Dtype])
                depth = self.params[Dtype]["depth"][celltype]
                locs[celltype] = np.vstack([pos_layer.T, depth*np.ones(self.params[Dtype]["N"][celltype])])

        ## Compute delay
        self.distances = {}
        self.delays = {}
        for pre in locs:
            pre_label = self.labels[pre]
            self.delays[pre_label] = {}
            self.distances[pre_label] = {}
            for post in self.params[Dtype]["con"]["p"][pre]:
                post_label = self.labels[post]
                self.delays[pre_label][post_label] = []
                self.distances[pre_label][post_label] = []
                if pre == post:
                    for ii in range(self.params[Dtype]["N"][pre] - 1):
                        for jj in range(ii + 1, self.params[Dtype]["N"][post]):
                            distance = np.sqrt(np.sum((locs[pre][:, ii] - locs[post][:, jj])**2))
                            self.distances[pre_label][post_label].append(distance)
                            self.delays[pre_label][post_label].append(self.delay_jitter[pre_label][post_label] * distance / self.params[Dtype]["conduction_velocity"] + self.params[Dtype]["syn_delay"])
                else:
                    for ii in range(self.params[Dtype]["N"][pre]):
                        for jj in range(self.params[Dtype]["N"][post]):
                            distance = np.sqrt(np.sum((locs[pre][:, ii] - locs[post][:, jj])**2))
                            self.distances[pre_label][post_label].append(distance)
                            self.delays[pre_label][post_label].append(self.delay_jitter[pre_label][post_label] * distance / self.params[Dtype]["conduction_velocity"] + self.params[Dtype]["syn_delay"])
        
        ## Get delays for afferents
        for ii in range(len(self.params[Dtype]["parameters"])):
            if self.params[Dtype]["parameters"][ii][0].startswith("delay_afferent"):
                pname, pre, post = self.params[Dtype]["parameters"][ii]
                pre_label = self.labels[pre]
                post_label = self.labels[post]
                if pre not in self.delays:
                    self.delays[pre_label] = {}
                
                self.delays[pre_label][post_label] = self.particle_best[Dtype][ii] - 1
        
        ## Determine whether the projection is excitatory or inhibitory
        projection_excitatory = {}
        for pre in self.delays:
            projection_excitatory[pre] = {}
            for post in self.delays[pre]:
                exc1 = -1 if pre.endswith("INT") else 1
                exc2 = -1 if post.endswith("INT") else 1
                projection_excitatory[pre][post] = exc1 * exc2 
        
        self.features["Node"]["TMS Activation"] = {}
        for Dtype in self.params:
            self.features["Node"]["TMS Activation"][Dtype] = np.zeros(self.n_parameters)
            for ind, parameter in enumerate(self.parameters):
                precell, postcell = parameter.split("-")
                if precell == "TMS":
                    for ii in range(len(self.params[Dtype]["parameters"])):
                        if self.params[Dtype]["parameters"][ii][0].startswith("p_active"):
                            ptype, _, _, postcheck = self.params[Dtype]["parameters"][ii]
                            if self.labels[postcheck] == postcell:
                                self.features["Node"]["TMS Activation"][Dtype][ind] = self.particle_best[Dtype][ii]
        
        # Combine coefficients for self.parameters
        # TODO make thresh_good an argument
        thresh_good = 0.5
        self.rsquared_ave = {}
        self.total_effectsize_all = {}
        self.waves = ["D", "I1", "I2", "I3"]
        for Dtype in self.coefs:
            self.rsquared_ave[Dtype] = {}
            self.total_effectsize_all[Dtype] = {}
            for wave in self.waves:
                self.rsquared_ave[Dtype][wave] = np.zeros(len(self.parameters_resorted))
                self.total_effectsize_all[Dtype][wave] = np.zeros(len(self.parameters_resorted))
                count = np.zeros(len(self.parameters))
                for ii in range(len(self.parameters)):
                    for jj in range(len(self.parameter_crosses)):
                        parameter1, parameter2 = self.parameter_crosses[jj].split(" x ")
                        flag = 0
                        if self.rsquared[Dtype][wave][jj] < thresh_good:
                            continue
                        
                        if self.parameters[ii] == parameter1:
                            self.total_effectsize_all[Dtype][wave][ii] += self.sensitivity[Dtype][wave][jj][0]
                            self.total_effectsize_all[Dtype][wave][ii] += self.sensitivity[Dtype][wave][jj][3]
                            self.total_effectsize_all[Dtype][wave][ii] += self.sensitivity[Dtype][wave][jj][7]
                            flag = 1
                        
                        if self.parameters[ii] == parameter2:
                            self.total_effectsize_all[Dtype][wave][ii] += self.sensitivity[Dtype][wave][jj][1]
                            self.total_effectsize_all[Dtype][wave][ii] += self.sensitivity[Dtype][wave][jj][4]
                            self.total_effectsize_all[Dtype][wave][ii] += self.sensitivity[Dtype][wave][jj][8]
                            flag = 1
                        
                        if flag:
                            self.total_effectsize_all[Dtype][wave][ii] += self.sensitivity[Dtype][wave][jj][2]
                            self.total_effectsize_all[Dtype][wave][ii] += self.sensitivity[Dtype][wave][jj][5]
                            self.total_effectsize_all[Dtype][wave][ii] += self.sensitivity[Dtype][wave][jj][6]
                            self.rsquared_ave[Dtype][wave][ii] += self.rsquared[Dtype][wave][jj]
                            count[ii] += 1
                
                count[count == 0] = 1
                self.rsquared_ave[Dtype][wave] /= count
        
        # Identify selective parameters
        ratio_selective = 1.5  # 2  # 1.5
        self.selective = {}
        self.nonselective = {}
        self.selectivity = {}
        self.combined_sensitivity = {}
        for Dtype in self.coefs:
            self.selective[Dtype] = {}
            self.nonselective[Dtype] = {}
            self.selectivity[Dtype] = {}
            self.combined_sensitivity[Dtype] = np.zeros(self.total_effectsize_all[Dtype]["D"].size)
            for ind, wave in enumerate(self.waves):
                self.selective[Dtype][wave] = np.abs(self.total_effectsize_all[Dtype][wave]) / np.abs(self.total_effectsize_all[Dtype][wave].max())
                self.nonselective[Dtype][wave] = np.abs(self.total_effectsize_all[Dtype][wave]) / np.abs(self.total_effectsize_all[Dtype][wave].max())
                self.selectivity[Dtype][wave] = np.abs(self.total_effectsize_all[Dtype][wave]).copy()
                self.combined_sensitivity[Dtype] += np.abs(self.total_effectsize_all[Dtype][wave])
            
            self.combined_sensitivity[Dtype] /= self.combined_sensitivity[Dtype].max()
            
            for ii in range(len(self.parameters_resorted)):
                sensitivities = np.array([np.abs(self.total_effectsize_all[Dtype][wave][ii]) for wave in self.waves])
                for wave in self.waves:
                    self.selectivity[Dtype][wave][ii] /= sensitivities.sum()
                
                if sensitivities.sum() == 0:
                    continue
                second_max = sensitivities[sensitivities < sensitivities.max()].max()
                ratio = sensitivities.max() / second_max
                ind_max = np.argmax(sensitivities)
                if ratio >= ratio_selective:
                    for ind, wave in enumerate(self.waves):
                        if ind != ind_max:
                            self.selective[Dtype][wave][ii] = 0
                        
                        self.nonselective[Dtype][wave][ii] = 0
                else:
                    for ind, wave in enumerate(self.waves):
                        self.selective[Dtype][wave][ii] = 0
        
        # Get graph theory metrics
        self.g = nx.DiGraph()
        has_recursion = {}  
        for precell in con["p"]:
            prelabel = self.labels[precell]
            has_recursion[prelabel] = 0
            for postcell in con["p"][precell]:
                postlabel = self.labels[postcell]
                self.g.add_edge(prelabel, postlabel, delay=np.mean(self.delays[prelabel][postlabel]), p=con_p[prelabel][postlabel])
                if prelabel == postlabel:
                    has_recursion[precell] = 1
        
        # Path metrics
        self.nodes = list(self.g.nodes)
        self.shortest_paths = {}
        self.features["Node"]["Shortest Path Order"] = np.zeros(self.n_celltypes)
        self.features["Node"]["Shortest Path Delay"] = np.zeros(self.n_celltypes)
        self.features["Node"]["Total Simple Paths"] = np.zeros(self.n_celltypes, dtype=int)
        self.simple_paths = {}
        self.all_shortest_paths = {}
        for ind, celltype in enumerate(self.celltypes):
            self.shortest_paths[celltype] = nx.shortest_path(self.g, source=celltype, target="L5 PC", weight="delay")
            self.features["Node"]["Shortest Path Delay"][ind] = nx.shortest_path_length(self.g, source=celltype, target="L5 PC", weight="delay", method="dijkstra")
            self.features["Node"]["Shortest Path Order"][ind] = len(self.shortest_paths[celltype]) - 1
            all_simple_paths = nx.all_simple_paths(self.g, source=celltype, target="L5 PC")
            all_shortest_paths = nx.shortest_simple_paths(self.g, source=celltype, target="L5 PC", weight="delay")
            self.simple_paths[celltype] = []
            self.all_shortest_paths[celltype] = []
            for path in all_simple_paths:
                self.simple_paths[celltype].append(path)
            
            self.features["Node"]["Total Simple Paths"][ind] = len(self.simple_paths[celltype])
            
            for path in all_shortest_paths:
                self.all_shortest_paths[celltype].append(path)
            
            if celltype == "L5 PC":
                self.features["Node"]["Shortest Path Order"][ind] = 0
                self.features["Node"]["Shortest Path Delay"][ind] = 0
        
        # Compute distances for simple_paths
        self.path_delays = {}
        for celltype in self.simple_paths:
            self.path_delays[celltype] = np.zeros(len(self.simple_paths[celltype]))
            for ind, path in enumerate(self.simple_paths[celltype]):
                for ii in range(len(path) - 1):
                    precell = path[ii]
                    postcell = path[ii + 1]
                    self.path_delays[celltype][ind] += np.mean(self.delays[precell][postcell])
        
        self.path_delays["L5 PC"] = np.array([0])
        
        # Compute connection probabilities for simple_paths in log-scale
        self.features["Node"]["Best Connected Path"] = np.zeros(self.n_celltypes)
        self.p_con = {}
        for ind_cell, celltype in enumerate(self.simple_paths):
            self.p_con[celltype] = np.ones(len(self.simple_paths[celltype]))
            for ind, path in enumerate(self.simple_paths[celltype]):
                for ii in range(len(path) - 1):
                    precell = path[ii]
                    postcell = path[ii + 1]
                    self.p_con[celltype][ind] *= con_p[precell][postcell]

            if len(self.p_con[celltype]) > 0:
                self.features["Node"]["Best Connected Path"][ind_cell] = np.min(np.log(self.p_con[celltype])) / np.sum(np.log(self.p_con[celltype]))
        
        self.p_con["L5 PC"] = np.array([np.mean(con_p["L5 PC"]["L5 PC"])])
        
        self.features["Node"]["Average Connection Probability (Log)"] = np.zeros(self.n_celltypes)
        self.features["Node"]["Stdev Connection Probability (Log)"] = np.zeros(self.n_celltypes)
        for ind, celltype in enumerate(self.celltypes):
            if len(self.p_con[celltype]) > 0:
                self.features["Node"]["Average Connection Probability (Log)"][ind] = np.mean(np.log(self.p_con[celltype]))
                self.features["Node"]["Stdev Connection Probability (Log)"][ind] = np.std(np.log(self.p_con[celltype]))
        
        self.features["Node"]["Average Connection Probability (Log)"][np.isnan(self.features["Node"]["Average Connection Probability (Log)"])] = 0
        self.features["Node"]["Stdev Connection Probability (Log)"][np.isnan(self.features["Node"]["Stdev Connection Probability (Log)"])] = 0
        
        # Averages weighted by connection probability
        self.weighted_path_delay = {}
        for celltype in self.simple_paths:
            self.weighted_path_delay[celltype] = self.path_delays[celltype] * self.p_con[celltype]
        
        self.features["Node"]["Weighted Average Path Delay"] = np.zeros(self.n_celltypes)
        for ind, celltype in enumerate(self.celltypes):
            if len(self.p_con[celltype]) > 0:
                self.features["Node"]["Weighted Average Path Delay"][ind] = np.sum(self.weighted_path_delay[celltype]) / np.sum(self.p_con[celltype])
        
        self.features["Node"]["Weighted Average Path Delay"][np.isnan(self.features["Node"]["Weighted Average Path Delay"])] = 0
        
        self.features["Node"]["Weighted Stdev Path Delay"] = np.zeros(self.n_celltypes)
        for ind, celltype in enumerate(self.celltypes):
            if len(self.p_con[celltype]) > 0:
                numerator = np.sum(self.p_con[celltype] * (self.path_delays[celltype] - self.features["Node"]["Weighted Average Path Delay"][ind]) ** 2)
                denominator = (len(self.p_con[celltype]) - 1) / len(self.p_con[celltype]) * np.sum(self.p_con[celltype])
                if denominator > 0:
                    self.features["Node"]["Weighted Stdev Path Delay"][ind] = np.sqrt(numerator / denominator)
        
        self.features["Node"]["Weighted Stdev Path Delay"][np.isnan(self.features["Node"]["Weighted Stdev Path Delay"])] = 0
        
        # Compute ratio of excitatory to inhibitory paths
        self.features["Node"]["Ratio Excitatory"] = np.zeros(self.n_celltypes)
        self.weighted_excitatory= {}
        for ind_celltype, celltype in enumerate(self.simple_paths):
            num_excitatory = np.zeros(len(self.simple_paths[celltype]))
            self.weighted_excitatory[celltype] = np.zeros(len(self.simple_paths[celltype]))
            for ind, path in enumerate(self.simple_paths[celltype]):
                excitatory = 1
                for ii in range(len(path)):
                    if path[ii].endswith("INT"):
                        excitatory *= -1
                
                self.weighted_excitatory[celltype][ind] = excitatory * self.p_con[celltype][ind]
                
                if excitatory == 1:
                    num_excitatory[ind] = 1
            
            if len(num_excitatory) > 1:
                self.features["Node"]["Ratio Excitatory"][ind_celltype] = np.sum(num_excitatory) / len(num_excitatory)
            else:
                self.features["Node"]["Ratio Excitatory"][ind_celltype] = 1
        
        self.weighted_excitatory["L5 PC"] = self.p_con["L5 PC"].copy()
        
        self.features["Node"]["Weighted Functional Effect"] = np.ones(self.n_celltypes)
        for ind, celltype in enumerate(self.celltypes):
            if len(self.p_con[celltype]) > 0:
                self.features["Node"]["Weighted Functional Effect"][ind] = np.sum(self.weighted_excitatory[celltype]) / np.sum(self.p_con[celltype])
        
        self.features["Node"]["Excitatory (Functional)"] = np.zeros(len(self.features["Node"]["Ratio Excitatory"]))
        self.features["Node"]["Excitatory (Functional)"][self.features["Node"]["Ratio Excitatory"] > 0.5] = 1
        
        self.features["Node"]["Average Path Delay"] = np.zeros(self.n_celltypes)
        self.features["Node"]["Stdev Path Delay"] = np.zeros(self.n_celltypes)
        for ind, celltype in enumerate(self.celltypes):
            if len(self.path_delays[celltype]) > 0:
                self.features["Node"]["Average Path Delay"][ind] = np.mean(self.path_delays[celltype])
                self.features["Node"]["Stdev Path Delay"][ind] = np.std(self.path_delays[celltype])
        
        # Centrality metrics
        closeness_centrality = nx.closeness_centrality(self.g, distance="delay")
        self.features["Node"]["Closeness Centrality"] = np.array([closeness_centrality[celltype] for celltype in self.celltypes])
        betweenness_centrality = nx.betweenness_centrality(self.g, weight="delay")
        self.features["Node"]["Betweenness Centrality"] = np.array([betweenness_centrality[celltype] for celltype in self.celltypes])
        harmonic_centrality = nx.harmonic_centrality(self.g, distance="delay")
        self.features["Node"]["Harmonic Centrality"] = np.array([harmonic_centrality[celltype] for celltype in self.celltypes])
        
        # Connection probability
        self.features["Node"]["Connection Probability Shortest Path"] = np.zeros(self.n_celltypes)
        for ind, celltype in enumerate(self.celltypes):
            probability = 1
            for ii in range(len(self.shortest_paths[celltype]) - 1):
                precell = self.shortest_paths[celltype][ii]
                postcell = self.shortest_paths[celltype][ii + 1]
                probability *= con_p[precell][postcell]
            
            self.features["Node"]["Connection Probability Shortest Path"][ind] = probability
        
        # Excitatory/inhibitory effect
        self.features["Node"]["Excitatory (One Hot)"] = np.zeros(self.n_celltypes)
        self.features["Node"]["Inhibitory (One Hot)"] = np.zeros(self.n_celltypes)
        for ind, celltype in enumerate(self.celltypes):
            excitatory = 1
            for ii in range(len(self.shortest_paths[celltype])):
                if self.shortest_paths[celltype][ii].endswith("INT"):
                    excitatory *= -1
            
            if excitatory == 1:
                self.features["Node"]["Excitatory (One Hot)"][ind] = 1
            else:
                self.features["Node"]["Inhibitory (One Hot)"][ind] = 1
    
    def gen_pos_circle(self, num_neurons, params):
        rangen_con = np.random.RandomState(params["seed_con"])
        pos = np.zeros((num_neurons, 2))
        radius = 0.5 * params["diameter_column"]
        for ii in range(num_neurons):
            flag = 1
            while flag:
                pos_tmp = rangen_con.uniform(-radius, radius, 2)
                circle = np.sum(pos_tmp**2)
                if circle < radius**2:
                    pos[ii] = pos_tmp
                    flag = 0

        return pos