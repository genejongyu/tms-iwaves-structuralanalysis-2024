# Inherits TVATAnalysis
# Adds functions to generate plots from TVAT analysis

from TVATAnalysis import TVATAnalysis
import matplotlib.pyplot as plt
import numpy as np
from colors_paultol import *
from scipy.stats import shapiro, levene, ttest_ind
import plotly.express as px

class TVATPlotting(TVATAnalysis):
    def __init__(self, dir_data, fname_sensitivity, fname_settings):
        # Initialize using TVATAnalysis
        TVATAnalysis.__init__(self, dir_data, fname_sensitivity, fname_settings)

    # Plot effect size for selective parameters for each wave
    def plot_effect_size_selective_by_wave(
        self,
        width = 0.75,
        pad = 6,
        max_plot = 5,
        waves = [],
        figsize = (0, 0),
        Dtype="D+",
    ):
        if len(waves) == 0:
            waves = self.waves
        if sum(figsize) == 0:
            figsize = (5, 3)
        figs_effect_size_selective_by_wave = {}
        bars_effect_size = {}
        projections_selective = {}
        for wave in waves:
            figs_effect_size_selective_by_wave[wave] = plt.figure(figsize=figsize)
            figs_effect_size_selective_by_wave[wave].subplots_adjust(left=0.52, right=0.93, top=0.9, bottom=0.2)
            _=plt.title("{}-Wave".format(wave), fontsize=16)
            bar_plot = self.total_effectsize_all[Dtype][wave][self.inds_resorted]
            idx = self.selective[Dtype][wave][self.inds_resorted] > 0
            x_bar = np.arange(bar_plot[idx].size)
            ind_sizesort = np.argsort(bar_plot[idx])
            projections_selective[wave] = [self.parameters_resorted[ii] for ii in range(len(self.parameters_resorted)) if self.selective[Dtype][wave][self.inds_resorted][ii] > 0]
            projections_selective[wave] = [projections_selective[wave][ind] for ind in ind_sizesort]
            bars_effect_size[wave] = plt.barh(x_bar[: max_plot], bar_plot[idx][ind_sizesort][-max_plot : ], width, color=colors["waves"][wave], edgecolor="k", linewidth=0.5)
        
            _=plt.yticks(x_bar[: max_plot], projections_selective[wave][-max_plot : ], fontsize=16)
            _=plt.xlabel("Effect Size", fontsize=16)
            _=plt.grid(True, axis="x")
            if 0 < len(ind_sizesort[-max_plot : ]) < pad:
                diff = pad - len(ind_sizesort[-max_plot : ])
                _=plt.ylim(-diff/2, x_bar[: max_plot][-1] + diff/2)

        return figs_effect_size_selective_by_wave

    # Plot total effect_size as sum of effect sizes for each parameter
    def plot_effect_size_total(
        self,
        width = 0.75,
        num_plot = 20,
        Dtype="D+",
    ):
        x_bar = np.arange(len(self.parameters_resorted))
        figs_effect_size_total = plt.figure(figsize=(6.5, 3.5))
        figs_effect_size_total.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.5)
        total = np.zeros(len(self.inds_resorted))
        for wave in self.waves:
            total += np.abs(self.total_effectsize_all[Dtype][wave][self.inds_resorted])
        
        ind_sort = np.argsort(-total)
        _=plt.bar(x_bar[:num_plot], total[ind_sort][:num_plot], width, color=cset_bright[4], edgecolor="k", linewidth=0.5)
        _=plt.xticks(x_bar[:num_plot], np.array(self.parameters_resorted)[ind_sort][:num_plot], rotation=90)
        _=plt.grid(True, axis="y")
        _=plt.ylabel("Effect Size")

        return figs_effect_size_total

    # Plot sensitivity of waves as sum of all effect sizes within wave
    def plot_sensitivity_all_by_wave(
        self,
        width = 0.75,
        d = 0.67,
        Dtype="D+",
    ):
        x_bar = np.arange(len(self.waves))
        figs_sensitivity_all_by_wave, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(4, 5))
        figs_sensitivity_all_by_wave.subplots_adjust(left=0.15, right=0.96, top=0.9, bottom=0.1, hspace=0.05)
        _=ax_top.set_title("Mean Sensitivity Across Waves (All)")
        for ind, wave in enumerate(self.waves):
            total = np.zeros(len(self.waves))
            total[ind] = np.mean(self.total_effectsize_all[Dtype][wave])
            _=ax_top.bar(x_bar, total, width, color=colors["waves"][wave], edgecolor="k", linewidth=0.5)
            _=ax_bottom.bar(x_bar, total, width, color=colors["waves"][wave], edgecolor="k", linewidth=0.5)
        ax_top.set_ylim(2.5, 3.5)
        ax_bottom.set_ylim(0, 1)
        ax_top.spines.bottom.set_visible(False)
        ax_bottom.spines.top.set_visible(False)
        ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False)
        ax_bottom.xaxis.tick_bottom()

        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)
        _=plt.xticks(x_bar, self.waves)
        _=ax_top.grid(True, axis="y")
        _=ax_bottom.grid(True, axis="y")
        _=plt.ylabel("Sensitivity")
        
        return figs_sensitivity_all_by_wave

    # Plot sensitivity of waves as sum of effect sizes for only selective parameters within wave
    def plot_sensitivity_selective_by_wave(
        self,
        width = 0.75,
        d = 0.67,
        Dtype="D+",
    ):
        x_bar = np.arange(len(self.waves))
        figs_sensitivity_selective_by_wave, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(4, 5))
        figs_sensitivity_selective_by_wave.subplots_adjust(left=0.15, right=0.98, top=0.9, bottom=0.1, hspace=0.05)
        _=ax_top.set_title("Mean Sensitivity Across Waves (Selective)")
        for ind, wave in enumerate(self.waves):
            idx = self.selective[Dtype][wave] > 0
            total = np.zeros(len(self.waves))
            total[ind] = np.mean(self.total_effectsize_all[Dtype][wave][idx])
            _=ax_top.bar(x_bar, total, width, color=colors["waves"][wave], edgecolor="k", linewidth=0.5)
            _=ax_bottom.bar(x_bar, total, width, color=colors["waves"][wave], edgecolor="k", linewidth=0.5)
    
        ax_top.set_ylim(2.5, 3.5)
        ax_bottom.set_ylim(0, 1)
        ax_top.spines.bottom.set_visible(False)
        ax_bottom.spines.top.set_visible(False)
        ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False)
        ax_bottom.xaxis.tick_bottom()

        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)
        _=plt.xticks(x_bar, self.waves)
        _=ax_top.grid(True, axis="y")
        _=ax_bottom.grid(True, axis="y")
        _=plt.ylabel("Sensitivity")

        return figs_sensitivity_selective_by_wave

    def plot_effect_size_by_type_stacked(
        self,
        width = 0.2,
        Dtype="D+",
    ):
        x_bar = np.arange(len(self.parameters_resorted))
        figs_effect_size_by_type_stacked = plt.figure(figsize=(15, 4.5))
        figs_effect_size_by_type_stacked.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.4)
        _=plt.title("Effect Size By Type: {}".format(Dtype))
        for ind, wave in enumerate(self.waves):
            bottom = np.zeros(len(self.parameters_resorted))
            bar_first = plt.bar(x_bar + ind * width, self.total_sensitivity_first[Dtype][wave][self.inds_resorted] / self.total_sensitivity_all[Dtype][wave][self.inds_resorted], width, color=colors["light"][0], edgecolor="k", linewidth=1, bottom=bottom, alpha=1)
            bottom += self.total_sensitivity_first[Dtype][wave][self.inds_resorted] / self.total_sensitivity_all[Dtype][wave][self.inds_resorted]
            bar_second = plt.bar(x_bar + ind  * width, self.total_sensitivity_second[Dtype][wave][self.inds_resorted] / self.total_sensitivity_all[Dtype][wave][self.inds_resorted], width, color=colors["light"][1], edgecolor="k", linewidth=1, bottom=bottom, alpha=1)
            bottom += self.total_sensitivity_second[Dtype][wave][self.inds_resorted] / self.total_sensitivity_all[Dtype][wave][self.inds_resorted]
            bar_third = plt.bar(x_bar + ind  * width, self.total_sensitivity_third[Dtype][wave][self.inds_resorted] / self.total_sensitivity_all[Dtype][wave][self.inds_resorted], width, color=colors["light"][2], edgecolor="k", linewidth=1, bottom=bottom, alpha=1)
            bottom += self.total_sensitivity_third[Dtype][wave][self.inds_resorted] / self.total_sensitivity_all[Dtype][wave][self.inds_resorted]
            bar_cross2 = plt.bar(x_bar + ind * width, self.total_sensitivity_cross[Dtype][wave][self.inds_resorted] / self.total_sensitivity_all[Dtype][wave][self.inds_resorted], width, color=colors["light"][3], edgecolor="k", linewidth=1, bottom=bottom, alpha=1)
            bottom += self.total_sensitivity_cross[Dtype][wave][self.inds_resorted] / self.total_sensitivity_all[Dtype][wave][self.inds_resorted]
            cross3 = (self.total_sensitivity_cross1[Dtype][wave][self.inds_resorted] + self.total_sensitivity_cross2[Dtype][wave][self.inds_resorted]) / self.total_sensitivity_all[Dtype][wave][self.inds_resorted]
            bar_cross3 = plt.bar(x_bar + ind * width, cross3, width, color=colors["light"][4], edgecolor="k", linewidth=1, bottom=bottom, alpha=1)

        _=plt.legend([bar_first, bar_second, bar_third, bar_cross2, bar_cross3, bar_cross2], ["1st-Order", "2nd-Order", "3rd-Order", "Interaction-2", "Interaction-3"])
        _=plt.xticks(x_bar + 1.5 * width, self.parameters_resorted, rotation=90)
        _=plt.grid(True, axis="y")
        _=plt.xlim(-2 * width, len(self.parameters_resorted))
        _=plt.ylabel("Normalized Effect Size")
        _=plt.ylim(0, 1.05)
        for ind, wave in enumerate(self.waves):
            _=plt.text(ind * width, 1.02, wave, fontsize=5, ha="center")
    
        return figs_effect_size_by_type_stacked 

    def plot_effect_size_by_wave_stacked(
        self,
        width = 0.75,
        num_plot = 20,
        Dtype="D+",
    ):
        x_bar = np.arange(len(self.parameters_resorted))
        figs_effect_size_by_wave_stacked = plt.figure(figsize=(6.5, 3.5))
        figs_effect_size_by_wave_stacked.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.5)
        _=plt.title("Effect Size by Wave: {}".format(Dtype))
        total = np.zeros(len(self.parameters_resorted))
        for wave in self.waves:
            total += np.abs(self.total_effectsize_all[Dtype][wave][self.inds_resorted])

        ind_sort = np.argsort(-total)
        bottom = np.zeros(len(self.parameters_resorted))[:num_plot]
        bar_stacks = []
        for ind, wave in enumerate(self.waves):
            bar_stack = plt.bar(x_bar[:num_plot], np.abs(self.total_effectsize_all[Dtype][wave][self.inds_resorted])[ind_sort][:num_plot] / total[ind_sort][:num_plot], width, color=colors["waves"][wave], edgecolor="k", linewidth=1, bottom=bottom, alpha=1)
            bar_stacks.append(bar_stack)
            bottom += np.abs(self.total_effectsize_all[Dtype][wave][self.inds_resorted])[ind_sort][:num_plot] / total[ind_sort][:num_plot]

        _=plt.legend(bar_stacks, self.waves, ncol=4)
        _=plt.xticks(x_bar[:num_plot], np.array(self.parameters_resorted)[ind_sort][:num_plot], rotation=90)
        _=plt.grid(True, axis="y")
        _=plt.ylabel("Normalized Effect Size")
        _=plt.ylim(0, 1.3)

        return figs_effect_size_by_wave_stacked
    
    # Plot effect size for each parameter for all waves
    def plot_effect_size_waves_all(
        self,
        width = 0.75,
        log_flag = 0,
        remove_big = 0,
        Dtype="D+",
    ):
        max_val = 65
        x_bar = np.arange(len(self.parameters_resorted))
        fig_effect_size_waves_all = plt.figure(figsize=(10, 10))
        fig_effect_size_waves_all.subplots_adjust(left=0.2, right=0.95, top=0.93, bottom=0.15)
        all_values = np.zeros((len(self.waves), len(self.total_effectsize_all[Dtype]["I1"])))
        for ind, wave in enumerate(self.waves):
            all_values[ind] = self.total_effectsize_all[Dtype][wave]
        
        xmin = all_values.min()
        idx_max = np.argsort(-all_values.flatten())
        xmax = all_values.flatten()[idx_max][remove_big]
        for ii in range(len(self.waves)):
            wave = self.waves[ii]
            _=plt.subplot(1, len(self.waves), ii+1)
            _=plt.title("{}-Wave".format(wave))
            y_flipped = np.flipud(self.total_effectsize_all[Dtype][wave][self.inds_resorted])
            bars = plt.barh(x_bar, y_flipped, width, color=colors["waves"][wave], edgecolor="k", linewidth=0.5)
            if not log_flag:
                for jj in range(len(x_bar)):
                    if y_flipped[jj] > max_val:
                        _=plt.text(0.7 * max_val, x_bar[jj]+width+0.2, "{:d}".format(int(y_flipped[jj])), color="k", va="center")
        
            if ii == 0:
                _=plt.yticks(x_bar, np.flipud(self.parameters_resorted))
            else:
                _=plt.yticks([])

            if not log_flag:
                _=plt.xlim(0, max_val)
            else:
                _=plt.xlim(xmin, xmax)
            _=plt.grid(True, axis="x")
            _=plt.xlabel("Effect Size")
            if log_flag:
                _=plt.xscale("log")

        return fig_effect_size_waves_all    
    
    # Plot effect size for parameters by wave while labeling selective vs non-selective
    def plot_effect_size_by_wave_selective_nonselective(
        self,
        width = 0.75,
        pad = 6,
        max_plot = 10,
        waves = [],
        figsize = (0, 0),
        Dtype="D+",
    ):
        if len(waves) == 0:
            waves = self.waves
        if sum(figsize) == 0:
            figsize = (5, 5)
        
        figs_effect_size_by_wave_selective_nonselective = {}
        projections_selective = {}
        for wave in waves:
            figs_effect_size_by_wave_selective_nonselective[wave] = plt.figure(figsize=figsize)
            figs_effect_size_by_wave_selective_nonselective[wave].subplots_adjust(left=0.52, right=0.93, top=0.9, bottom=0.2)
            _=plt.title("{}-Wave".format(wave), fontsize=16)
            data_all = self.total_effectsize_all[Dtype][wave][self.inds_resorted].copy()
            data_all /= np.max(data_all)
            data_selective = self.selective[Dtype][wave][self.inds_resorted].copy()
            ind_sizesort = np.argsort(-data_all)
            parameters_sorted = [self.parameters_resorted[ind] for ind in ind_sizesort]
            
            x_bar = np.arange(max_plot)
            _=plt.barh(x_bar, np.flipud(data_all[ind_sizesort][:max_plot]), width, color=colors["light"][4], edgecolor="k", linewidth=0.5, alpha=0.75)
            _=plt.barh(x_bar, np.flipud(data_selective[ind_sizesort][:max_plot]), width, color=colors["waves"][wave], edgecolor="k", linewidth=0.5)
            _=plt.yticks(x_bar, np.flipud(parameters_sorted[:max_plot]), fontsize=16)
            _=plt.xlabel("Effect Size", fontsize=16)
            _=plt.grid(True, axis="x")
            if 0 < len(ind_sizesort[-max_plot : ]) < pad:
                diff = pad - len(ind_sizesort[-max_plot : ])
                _=plt.ylim(-diff/2, x_bar[: max_plot][-1] + diff/2)

        return figs_effect_size_by_wave_selective_nonselective
    
    # Divide and plot sensitivity for afferent vs non-afferent
    def plot_sensitivity_wave_afferent_nonafferent(
        self,
        width = 0.33,
        Dtype="D+",
    ):
        x_bar = np.arange(len(self.waves))
        figs_sensitivity_wave_afferent_nonafferent= plt.figure(figsize=(3.25, 2.5))
        figs_sensitivity_wave_afferent_nonafferent.subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.1, hspace=0.05)
        _=plt.title("Sensitivity (Afferent vs Circuit)")
        idx_afferent = np.zeros(len(self.parameters_resorted), dtype=bool)
        for ii in range(len(self.parameters_resorted)):
            if self.parameters_resorted[ii].startswith("TMS"):
                if self.parameters_resorted[ii].endswith("AFF"):
                    idx_afferent[ii] = True
            else:
                pre, post = self.parameters_resorted[ii].split("-")
                if pre.endswith("AFF"):
                    idx_afferent[ii] = True

        idx_circuit = ~idx_afferent
        for ind, wave in enumerate(self.waves):
            sensitivity_afferent = np.zeros(len(self.waves))
            sensitivity_circuit = np.zeros(len(self.waves))
            sensitivity_afferent[ind] = np.mean(self.total_effectsize_all[Dtype][wave][self.inds_resorted][idx_afferent])
            sensitivity_circuit[ind] = np.mean(self.total_effectsize_all[Dtype][wave][self.inds_resorted][idx_circuit])
            if ind == 0:
                _=plt.bar(x_bar, sensitivity_afferent, width, color=colors["popsmooth"]["D+"], edgecolor="k", linewidth=0.5, label="Afferent")
                _=plt.bar(x_bar + width, sensitivity_circuit, width, color=colors["popsmooth"]["D-"], edgecolor="k", linewidth=0.5, label="Circuit")
            else:
                _=plt.bar(x_bar, sensitivity_afferent, width, color=colors["popsmooth"]["D+"], edgecolor="k", linewidth=0.5)
                _=plt.bar(x_bar + width, sensitivity_circuit, width, color=colors["popsmooth"]["D-"], edgecolor="k", linewidth=0.5)

        _=plt.legend()
        _=plt.xticks(x_bar + width / 2, self.waves)
        _=plt.grid(True, axis="y")
        _=plt.ylabel("Sensitivity")
        
        return figs_sensitivity_wave_afferent_nonafferent
    
    # Hierarchically plot sensitivities for pairs of categories
    # Afferent vs non-afferent
    # Excitatory vs inhibitory
    # Preferential vs non-preferential
    def plot_sensitivities_hierarchical(
        self,
        width = 0.33,
        Dtype="D+",
    ):
        x_bar = np.arange(len(self.waves))
        figs_sensitivities_hierarchical = {}
        idx_tms = np.zeros(len(self.parameters_resorted), dtype=bool)
        idx_tms_aff = np.zeros(len(self.parameters_resorted), dtype=bool)
        idx_afferent = np.zeros(len(self.parameters_resorted), dtype=bool)
        for ii in range(len(self.parameters_resorted)):
            if self.parameters_resorted[ii].startswith("TMS"):
                idx_tms[ii] = True
                if self.parameters_resorted[ii].endswith("AFF"):
                    idx_tms_aff[ii] = True
            else:
                if self.parameters_resorted[ii].endswith("AFF"):
                    idx_afferent[ii] = True
                else:
                    pre, post = self.parameters_resorted[ii].split("-")
                    if pre.endswith("AFF"):
                        idx_afferent[ii] = True

        idx_circuit = ~idx_tms
        
        idx_selective = np.zeros(len(self.parameters_resorted), dtype=bool)
        for wave in self.selective[Dtype]:
            idx_selective[self.selective[Dtype][wave][self.inds_resorted] > 0] = True

        parameters_selective = [self.parameters_resorted[ii] for ii in range(len(self.parameters_resorted)) if idx_selective[ii]]
        idx_nonselective = ~idx_selective
        
        idx_inhibitory = np.zeros(len(self.parameters_resorted), dtype=bool)
        for ii in range(len(self.parameters_resorted)):
            pre, post = self.parameters_resorted[ii].split("-")
            if pre.endswith("BC"):
                idx_inhibitory[ii] = True
            if pre.startswith("TMS"):
                if "BC" in post:
                    idx_inhibitory[ii] = True

        idx_excitatory = ~idx_inhibitory
        
        sensitivity_tms = np.zeros(len(self.waves))
        sensitivity_tms_aff = np.zeros(len(self.waves))
        sensitivity_tms_col = np.zeros(len(self.waves))
        sensitivity_projection = np.zeros(len(self.waves))
        sensitivity_afferent = np.zeros(len(self.waves))
        sensitivity_circuit = np.zeros(len(self.waves))
        sensitivity_circuit_excitatory = np.zeros(len(self.waves))
        sensitivity_circuit_inhibitory = np.zeros(len(self.waves))
        for ind, wave in enumerate(self.waves):
            sensitivity_tms[ind] = np.mean(self.total_effectsize_all[Dtype][wave][self.inds_resorted][idx_tms])
            sensitivity_tms_aff[ind] = np.mean(self.total_effectsize_all[Dtype][wave][self.inds_resorted][idx_tms * idx_tms_aff])
            sensitivity_tms_col[ind] = np.mean(self.total_effectsize_all[Dtype][wave][self.inds_resorted][idx_tms * ~idx_tms_aff])
            sensitivity_projection[ind] = np.mean(self.total_effectsize_all[Dtype][wave][self.inds_resorted][~idx_tms])
            sensitivity_afferent[ind] = np.mean(self.total_effectsize_all[Dtype][wave][self.inds_resorted][idx_afferent])
            sensitivity_circuit[ind] = np.mean(self.total_effectsize_all[Dtype][wave][self.inds_resorted][idx_circuit])
            sensitivity_circuit_excitatory[ind] = np.mean(self.total_effectsize_all[Dtype][wave][self.inds_resorted][idx_circuit * idx_excitatory])
            sensitivity_circuit_inhibitory[ind] = np.mean(self.total_effectsize_all[Dtype][wave][self.inds_resorted][idx_circuit * idx_inhibitory])
        
        ymax = 40
        figsize = (3.25, 2.25)
        figs_sensitivities_hierarchical["tms_projection"] = plt.figure(figsize=figsize)
        figs_sensitivities_hierarchical["tms_projection"].subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.11, hspace=0.05)
        _=plt.title("Sensitivity (TMS vs Projection")
        _=plt.bar(x_bar, sensitivity_tms, width, color=colors["popsmooth"]["D+"], edgecolor="k", linewidth=0.5, label="Activation")
        _=plt.bar(x_bar + width, sensitivity_projection, width, color=colors["popsmooth"]["D-"], edgecolor="k", linewidth=0.5, label="Synapse")
        _=plt.legend()
        _=plt.xticks(x_bar + width / 2, self.waves)
        _=plt.grid(True, axis="y")
        _=plt.ylabel("Sensitivity")
        _=plt.ylim(0, ymax)
        figs_sensitivities_hierarchical["tms_aff_col"] = plt.figure(figsize=figsize)
        figs_sensitivities_hierarchical["tms_aff_col"].subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.11, hspace=0.05)
        _=plt.title("Sensitivity (TMS vs Projection")
        _=plt.bar(x_bar, sensitivity_tms_aff, width, color=colors["popsmooth"]["D+"], edgecolor="k", linewidth=0.5, label="Afferent")
        _=plt.bar(x_bar + width, sensitivity_tms_col, width, color=colors["popsmooth"]["D-"], edgecolor="k", linewidth=0.5, label="Column")
        _=plt.legend()
        _=plt.xticks(x_bar + width / 2, self.waves)
        _=plt.grid(True, axis="y")
        _=plt.ylabel("Sensitivity")
        _=plt.ylim(0, ymax)
        figs_sensitivities_hierarchical["afferent_circuit"] = plt.figure(figsize=figsize)
        figs_sensitivities_hierarchical["afferent_circuit"].subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.11, hspace=0.05)
        _=plt.title("Sensitivity (Afferent vs Circuit)")
        _=plt.bar(x_bar, sensitivity_afferent, width, color=colors["popsmooth"]["D+"], edgecolor="k", linewidth=0.5, label="Afferent")
        _=plt.bar(x_bar + width, sensitivity_circuit, width, color=colors["popsmooth"]["D-"], edgecolor="k", linewidth=0.5, label="Column")
        _=plt.legend()
        _=plt.xticks(x_bar + width / 2, self.waves)
        _=plt.grid(True, axis="y")
        _=plt.ylabel("Sensitivity")
        _=plt.ylim(0, ymax)
        figs_sensitivities_hierarchical["circuit_excitatory"] = plt.figure(figsize=figsize)
        figs_sensitivities_hierarchical["circuit_excitatory"].subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.11, hspace=0.05)
        _=plt.title("Sensitivity Circuit (Exc vs Inh)")
        _=plt.bar(x_bar, sensitivity_circuit_excitatory, width, color=colors["popsmooth"]["D+"], edgecolor="k", linewidth=0.5, label="Excitatory")
        _=plt.bar(x_bar + width, sensitivity_circuit_inhibitory, width, color=colors["popsmooth"]["D-"], edgecolor="k", linewidth=0.5, label="Inhibitory")
        _=plt.legend()
        _=plt.xticks(x_bar + width / 2, self.waves)
        _=plt.grid(True, axis="y")
        _=plt.ylabel("Sensitivity")
        _=plt.ylim(0, ymax)
        
        return figs_sensitivities_hierarchical
