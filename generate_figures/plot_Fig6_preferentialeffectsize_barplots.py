# Plots effect sizes for preferential parameters for Fig 6

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
        
    # Plot effect sizes for top 5 selective parameters per wave
    figs_effect_size_selective_by_wave = results.plot_effect_size_selective_by_wave()

    if save_flag:
        for wave in figs_effect_size_selective_by_wave:
            figs_effect_size_selective_by_wave[wave].savefig("Fig6_preferentialeffectsize_barplots_{}.tiff".format(wave), dpi=600)
    
    plt.show()