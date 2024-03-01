# Plot all effect sizes for all corticospinal waves for S1 Appendix Fig F

# Originally: Analysis/plot_fig_sensitivityanalysis.py
from os.path import join
from get_paths import *

from TVATPlotting import TVATPlotting
import matplotlib.pyplot as plt

if __name__ == "__main__":
    save_flag = 0
    
    ## Load data
    fname_regressions = "data_tvat_polynomialregressions.pickle"
    
    fname_settings = {}
    fname_settings["D+"] = "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle"
    fname_settings["D-"] = "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle"
    
    results = TVATPlotting(dir_data, fname_regressions, fname_settings)
        
    ## Plot data
    # Plot effect size for each parameter for all waves
    fig_effect_size_waves_all = results.plot_effect_size_waves_all(remove_big=3)
    
    if save_flag:
        fig_effect_size_waves_all.savefig("SFigF_effectsizes_all.tiff", dpi=600)
    
    plt.show()