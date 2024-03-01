# Class to load and analyze data

from os.path import join
import h5py
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter

class OptimizationData:
    def __init__(self, dir_data, fname_data, fname_settings, load=True):
        # Load data and parameters
        self.dir_data = dir_data
        self.fname_data = fname_data
        self.fname_settings = fname_settings
        if load:
            if isinstance(fname_data, str):
                fpath_data = self.combine_dir_fname(dir_data, fname_data)
                fpath_settings = self.combine_dir_fname(dir_data, fname_settings)
                self.data = self.load_data(fpath_data)
                self.params = self.load_params(fpath_settings)
            elif isinstance(fname_data, list):
                fpaths_data = self.combine_dir_fnames_multiple(dir_data, fname_data)
                fpaths_settings = self.combine_dir_fnames_multiple(dir_data, fname_settings)
                self.data = self.load_data_multiple(fpaths_data)
                self.params = self.load_params(fpaths_settings[0])
            
            # Get basic info
            self.n_gen, self.n_pop, self.n_allele = self.data["alleles"].shape
            # self.n_gen = 959
    
    
    def combine_dir_fname(self, dir_data, fname):
        fpath = join(dir_data, fname)
        return fpath

    
    def combine_dir_fnames_multiple(self, dir_data, fnames):
        fpaths = []
        for fname in fnames:
            fpaths.append(self.combine_dir_fname(dir_data, fname))
        
        return fpaths

    
    def load_data(self, fpath):
        with h5py.File(fpath, "r") as dset:
            data = {}
            for key in dset:
                if key not in ["smooth_fitness", "thresh_max", "fname_settings"]:
                    if key != "weights":
                        data[key] = dset[key][:]
        
        return data

    
    def load_data_multiple(self, fpaths):
        data_all = self.load_data(fpaths[0])
        ind_start_runs = [0, data_all["fitness"].size]
        for ii in range(1, len(fpaths)):
            fpath = fpaths[ii]
            data = self.load_data(fpath)
            for key in data_all:
                if key not in ["smooth_fitness", "thresh_max", "fname_settings"]:
                    data_all[key] = np.hstack([data_all[key], data[key]])
            
            ind_start_runs.append(ind_start_runs[-1] + data["fitness"].size)
        
        self.idx_runs = np.zeros(ind_start_runs[-1], dtype=int)
        for ii in range(len(ind_start_runs) - 1):
            self.idx_runs[ind_start_runs[ii] : ind_start_runs[ii+1]] = ii
        
        return data_all

    
    def load_params(self, fpath):
        with open(fpath, "rb") as f:
            params = pickle.load(f)
        
        return params
    
    
    def smooth_histogram(self, data, n_bins, lo, hi, w_gaussian):
        hist, bins = np.histogram(
            data,
            n_bins,
            range=(lo, hi)
        )
        hist_smooth = gaussian_filter(
            hist.astype(float), w_gaussian
        )
        hist_smooth /= hist_smooth.max()
        return hist_smooth, bins

    
    def calc_relative_error(self):
        if "idx_fit" not in dir(self):
            self.total_chromosomes = self.n_gen * self.n_pop
            self.idx_fit = np.ones((self.n_gen, self.n_pop), dtype=bool)
            filter_flag = 0
        else:
            self.total_chromosomes = self.fitness.size
            filter_flag = 1
        ind_subject = 0
        self.relative_error = {}
        cell_order = ["L2/3 PC", "L5 PC", "L6 PC", "L2/3 INT", "L5 INT", "L6 INT"]
        for obj in self.params["objective_names"]:
            if obj in self.params["targets"]:
                if len(self.params["targets"][obj].shape) > 1:
                    if not filter_flag:
                        data_flatten = self.data[obj][:self.n_gen, :, :, :self.params["measured_waves"]].reshape(
                            (self.total_chromosomes, self.params["measured_waves"])
                        )
                    else:
                        data_flatten = self.data[obj][:self.n_gen, :, :, :self.params["measured_waves"]][self.idx_fit][np.arange(self.total_chromosomes), ind_subject]
                    ind_start = 0
                    
                    # Normalize peaks before computing relative error
                    if obj in ["peaks_iwave", "minima_iwave"]:
                        amp_I1 = self.data["peaks_iwave"][:self.n_gen, :, :, :self.params["measured_waves"]][self.idx_fit].reshape(
                            (self.total_chromosomes, self.params["measured_waves"])
                        )[:, self.params["ind_I1"]]
                        for ii in range(self.params["measured_waves"]):
                            data_flatten[:, ii] /= amp_I1
                    
                    # Compute relative error
                    num_targets = self.params["targets"][obj].shape[1]
                    for ind_wave in range(ind_start, num_targets):
                        if obj == "peaks_iwave":
                            if ind_wave == 1:
                                continue
                        if obj == "tpeaks_iwave":
                            if "2020_43" in self.params["fname_root"]:
                                if ind_wave < 2:
                                    continue
                            else:
                                if ind_wave == 1:
                                    continue
                        if obj == "tminima_iwave":
                            if "2020_43" in self.params["fname_root"]:
                                if ind_wave == 0:
                                    continue
                        
                        if ind_wave == 0:
                            wave = "D"
                        else:
                            wave = "I%i" % ind_wave
                        
                        target_name = obj+"-"+wave
                        target = np.zeros(self.total_chromosomes)
                        for ii in range(self.total_chromosomes):
                            target[ii] = self.params["targets"][obj][ind_subject][ind_wave]
                        
                        if target.min() != 0:
                            self.relative_error[target_name] = np.abs(
                                (data_flatten[:, ind_wave] - target) / target
                            )
                        else:
                            self.relative_error[target_name] = np.abs(data_flatten[:, ind_wave])
                
                else:
                    if not filter_flag:
                        data_flatten = self.data[obj][:self.n_gen, :].reshape(
                            (self.total_chromosomes,) + self.params["targets"][obj].shape
                        )
                    else:
                        data_flatten = self.data[obj][:self.n_gen][self.idx_fit]
                    for ii in range(len(self.params["targets"][obj])):
                        if obj != "postmin_iwave":
                            obj_name = obj + "-" + cell_order[ii]
                        else:
                            obj_name = obj
                        if np.sum(np.abs(self.params["targets"][obj])) > 0:
                            self.relative_error[obj_name] = np.abs(
                                (data_flatten[:, ii] - self.params["targets"][obj][ii]) / self.params["targets"][obj][ii]
                            )
                        else:
                            self.relative_error[obj_name] = np.abs((data_flatten[:, ii] - self.params["targets"][obj][ii]))
        
        self.total_error = np.zeros(self.total_chromosomes)
        for obj in self.relative_error:
            if "syn_noise" not in obj:
                self.total_error += self.relative_error[obj]