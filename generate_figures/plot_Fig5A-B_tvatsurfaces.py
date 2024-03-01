# Plots TVAT surfaces and polynomial regressions from Fig 5A-B

from os.path import join
from get_paths import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from colors_paultol import cmap_iridescent

if __name__ == "__main__":
    save_flag = 0
    
    ## Load data
    fname = "data_tvat_surfaces_examples.pickle"
    fpath = join(dir_data, fname)
    with open(fpath, "rb") as f:
        data = pickle.load(f)

    ## Plot
    figs = {}
    for parameter_pair in data:
        figs[parameter_pair] = plt.figure(figsize=(3.25, 3))
        figs[parameter_pair].subplots_adjust(left=0.15, right=0.91, bottom=0.16, top=0.93, wspace=0.4)
        xlabel = data[parameter_pair]["xlabel"]
        ylabel = data[parameter_pair]["ylabel"]
        if xlabel.startswith("TMS"):
            xticks = ["0", "0.5", "1"]
        else:
            xticks = ["0", "5", "10"]
        if ylabel.startswith("TMS"):
            yticks = ["0", "0.5", "1"]
        else:
            yticks = ["0", "5", "10"]
        
        waves = list(data[parameter_pair]["sim_data"].keys())
        for ind_wave, wave in enumerate(waves):
            r2 = data[parameter_pair]["r2"][wave]
            sim_data = data[parameter_pair]["sim_data"][wave]
            regression = data[parameter_pair]["regression"][wave]
            
            ax = plt.subplot(2, 2, ind_wave + 1)
            _=plt.title("$r^{{2}}$: {0:.2f}".format(r2)) 
            IM = plt.imshow(sim_data, origin="lower", cmap=cmap_iridescent)
            _=plt.xticks([])
            if ind_wave == 0:
                _=plt.yticks(np.linspace(0, 20, 3), yticks) 
                _=plt.ylabel(ylabel) 
            else:
                _=plt.yticks([])
            divider = make_axes_locatable(ax)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(IM, cax = cax1)
            cbar.ax.tick_params(labelsize=9)
        
            ax = plt.subplot(2, 2, ind_wave + len(waves) + 1)
            IM = plt.imshow(regression, origin="lower", cmap=cmap_iridescent)
            _=plt.xticks(np.linspace(0, 20, 3), xticks)
            _=plt.xlabel(xlabel)
            if ind_wave == 0:
                _=plt.yticks(np.linspace(0, 20, 3), yticks)
                _=plt.ylabel(ylabel)
            else:
                _=plt.yticks([])
            divider = make_axes_locatable(ax)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(IM, cax = cax1)
            cbar.ax.tick_params(labelsize=9)

    if save_flag:
        for pair in figs:
            figs[pair].savefig("Fig5A-B_tvatsurfaces_{}.tiff".format(pair), dpi=600)
    
    plt.show()
                