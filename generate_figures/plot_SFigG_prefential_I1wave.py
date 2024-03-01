# Plot effect sizes for all parameters that preferentially activated the I1-wave
# for S1 Appendix Fig G

from os.path import join
from get_paths import *
from TVATPlotting import TVATPlotting
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ## Load data
    fname_regressions = "data_tvat_polynomialregressions.pickle"
    
    fname_settings = {}
    fname_settings["D+"] = "settings_pso_iwaves_column_restingstate_2013_31_120RMT_seed0.pickle"
    fname_settings["D-"] = "settings_pso_iwaves_column_restingstate_2020_43_120RMT_seed0.pickle"
    
    results = TVATPlotting(dir_data, fname_regressions, fname_settings)
        
    ## Plot data
    # Plot of all selective I1-wave parameters
    figs_effect_size_selective_I1 = results.plot_effect_size_selective_by_wave(
        width = 0.75,
        pad = 6,
        max_plot = 20,
        waves = ["I1"],
        figsize = (5, 6),
    )

    # figs_effect_size_selective_I1["I1"].savefig("..\\..\\Final_Figures\\Original_Figures\\SFigG_prefential_I1wave.tiff", dpi=600)
    
    plt.show()