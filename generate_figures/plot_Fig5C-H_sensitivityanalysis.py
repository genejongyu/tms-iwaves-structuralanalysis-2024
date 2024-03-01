# Plots effect sizes and sensitivities for Fig 5C-H

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
    
    # Plot total effect size (across all waves)
    figs_effect_size_total = results.plot_effect_size_total()
    
    # Plot relative contributions of each wave to total sum of coefficients, normalized and stacked
    figs_effect_size_by_wave_stacked = results.plot_effect_size_by_wave_stacked()

    # Plot sensitivity comparing different groups
    figs_sensitivities_hierarchical = results.plot_sensitivities_hierarchical()

    if save_flag:
        figs_effect_size_total.savefig("Fig5C_effectsize_ranked.tiff", dpi=600)
        figs_effect_size_by_wave_stacked.savefig("Fig5D_effectsize_waves.tiff", dpi=600)
        letters = ["E", "F", "G", "H"]
        for ind, key in enumerate(figs_sensitivities_hierarchical):
            letter = letters[ind]
            figs_sensitivities_hierarchical[key].savefig("Fig5{}_{}.tiff".format(letter, key), dpi=600)
    
    plt.show()