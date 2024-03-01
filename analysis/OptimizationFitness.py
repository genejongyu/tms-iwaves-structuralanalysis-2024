# Inherits OptimizationData and adds functions for plotting smoothed error

from OptimizationData import OptimizationData
import numpy as np
import matplotlib.pyplot as plt

class OptimizationFitness(OptimizationData):
    def __init__(self, dir_data, fname_data, fname_settings, n_bins=1001, w_gaussian=5, load=True):
        # Load data
        OptimizationData.__init__(self, dir_data, fname_data, fname_settings, load)
        
        # Set plotting parameters
        self.n_bins = n_bins
        self.w_gaussian = w_gaussian
    
    
    def set_plot_data(self, lo=None, hi=None):
        # Set max and min of error
        self.set_error_range(lo=lo, hi=hi)
        
        # Calculate evolution of error using smoothed histograms
        self.calc_smooth_error_histogram_evolution(self.n_bins, self.w_gaussian)
        
        # Calculate min and mode of fitness
        self.calc_min_error_evolution()
        self.calc_mode_error_evolution()
        
        # Calculate cumulative best
        self.calc_cumulative_min_error_evolution()
    
    
    def set_error_range(self, lo=None, hi=None):
        if lo:
            self.lo_error = lo
        else:
            self.lo_error = np.min(self.data["fitness"])
        if hi:
            self.hi_error = hi
        else:
            self.hi_error = np.max(self.data["fitness"])
    
    
    def calc_smooth_error_histogram_evolution(self, n_bins, w_gaussian):
        self.smooth_histogram_error = np.zeros((self.n_gen, n_bins))
        for gen in range(self.n_gen):
            self.smooth_histogram_error[gen], self.bins_error = self.smooth_histogram(
                self.data["fitness"][gen].ravel(),
                n_bins,
                self.lo_error,
                self.hi_error,
                w_gaussian
                )
    
    
    def calc_min_error_evolution(self):
        self.min_error_evolution = np.min(self.data["fitness"], axis=1)
    
    
    def calc_cumulative_min_error_evolution(self):
        self.cumulative_min_error_evolution = np.zeros(self.n_gen)
        for gen in range(self.n_gen):
            self.cumulative_min_error_evolution[gen] = np.min(self.data["fitness"][:gen+1])
    
    
    def calc_mode_error_evolution(self):
        self.mode_error_evolution = self.bins_error[
            np.argmax(
                self.smooth_histogram_error, 
                axis=1)
        ]
    
    
    def plot_errors(self):
        gens = np.linspace(-0.5, self.n_gen - 0.5, self.n_gen + 1)
        self.fig_errors = plt.figure(figsize=(12, 4))
        self.fig_errors.subplots_adjust(
            left=0.05, right=0.97, bottom=0.12, top=0.97, wspace=0.1
        )
        _=plt.subplot(1, 2, 1)
        _ = plt.pcolormesh(gens, self.bins_error, self.smooth_histogram_error.T)
        _ = plt.ylabel("Fitness")
        _ = plt.xlabel("Generation")

        _=plt.subplot(1, 2, 2)
        _=plt.plot(self.cumulative_min_error_evolution, label="Min")
        _=plt.plot(self.mode_error_evolution, label="Mode")
        _=plt.legend()
        _=plt.xlabel("Generation")
        _=plt.ylabel("Fitness")
        #_=plt.yscale("log")
        _=plt.ylim(self.lo_error, self.hi_error)
        _=plt.xlim(0, self.n_gen)
        _=plt.grid(True)


    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    # Faster than is_pareto_efficient_simple, but less readable.
    def is_pareto_efficient(self, costs, return_mask=True):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a masl
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array.
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient      


    def group_errors(self, groups=[]):
        self.group_objectives = [obj for obj in self.relative_error if "syn_noise" not in obj]
        if len(groups) == 0:
            groups = ["CS-Wave", "Well-Behaved", "Activity", "CV"]
        
        self.constraint_groups = {}
        for group in groups:
            self.constraint_groups[group] = []
        
        for obj in self.group_objectives:
            if "CS-Wave" in self.constraint_groups:
                if "_iwave" in obj:
                    self.constraint_groups["CS-Wave"].append(obj)
            if "Activity" in self.constraint_groups:
                if ("firingrate" in obj) or ("ISI_mean" in obj):
                    self.constraint_groups["Activity"].append(obj)
            if "Spiking Activity" in self.constraint_groups:
                if "ISI_mean" in obj:
                    self.constraint_groups["Spiking Activity"].append(obj)
            if "Well-Behaved" in self.constraint_groups:
                if "ISI_std" in obj:
                    self.constraint_groups["Well-Behaved"].append(obj)
                # self.constraint_groups["Well-Behaved"].append(obj)
            if "Synchrony" in self.constraint_groups:
                if "baseline" in obj:
                    self.constraint_groups["Synchrony"].append(obj)
            if "CV" in self.constraint_groups:
                if "response_cv" in obj:
                    self.constraint_groups["CV"].append(obj)
        
        self.grouped_error = {}
        for group in self.constraint_groups:
            self.grouped_error[group] = np.zeros(self.relative_error[obj].size)
            for obj in self.constraint_groups[group]:
                self.grouped_error[group] += self.relative_error[obj]
        
            self.grouped_error[group] /= len(self.constraint_groups[group])

    def get_pareto_dominants(self, groups=[]):
        self.group_errors(groups)
        N_particles = self.total_error.size
        all_error = np.zeros((N_particles, len(self.grouped_error)))
        dominated = {}
        for ii in range(N_particles):
            dominated[ii] = 0
            all_error[ii] = np.array([self.grouped_error[group][ii] for group in self.grouped_error])
        
        self.dominators = self.is_pareto_efficient(all_error, return_mask = True)
