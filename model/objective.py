# Functions used to evaluate particles to match corticospinal response

import numpy as np
from scipy import signal


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    y = signal.sosfiltfilt(sos, data)
    return y


def butter_lowpass_filter(data, lowcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = signal.butter(order, low, btype="lowpass", output="sos")
    y = signal.sosfiltfilt(sos, data)
    return y


def get_stim_times(stim_delay, num_subjects, stim_amps, num_trials, trial_interval):
    stim_times = {}
    for state in stim_amps:
        tstim_total = stim_delay
        stim_times[state] = {}
        for subj in range(num_subjects):
            stim_times[state][subj] = np.zeros((len(stim_amps[state][subj]), num_trials))
            for amp in range(len(stim_amps[state][subj])):
                for ii in range(num_trials):
                    stim_times[state][subj][amp][ii] = tstim_total
                    tstim_total += trial_interval[state]
    
    return stim_times


def get_tstops(stim_times, num_subjects, trial_interval, stim_delay):
    tstops = {}
    for state in stim_times:
        if stim_times[state][num_subjects-1].shape[1] > 0:
            tstops[state] = stim_times[state][num_subjects-1][-1][-1] + trial_interval[state]
            if state == "resting":
                tstops[state] -= trial_interval[state]
                tstops[state] += 100
        else:
            tstops[state] = stim_delay
    
    return tstops


def check_current_nan(currents):
    bad_flag = 0
    for state in currents:
        idx_nan = np.zeros(currents[state].shape)[
            np.isnan(currents[state])
        ]
        if idx_nan.size > 0:
            bad_flag = 1
    
    return bad_flag


def get_poprates_all(spikes, state, params):
    celltype = "L5"
    if state == "resting":
        t_window = params["trange_iwaves"]
        t_pre = 0
    elif state == "isometric":
        t_window = params["tpre_csp"] + params["trial_interval"][state]
        t_pre = params["tpre_csp"]
    
    poprates = np.zeros(
        (
            np.sum(params["num_amps"][state]),
            params["num_trials"],
            int(t_window / params["tres"])
        )
    )
    ind = 0
    for subj in range(params["num_subjects"]):
        for amp in range(params["num_amps"][state][subj]):
            for trial in range(params["num_trials"]):
                tstart = params["stim_times"][state][subj][amp][trial] - t_pre
                poprates[ind][trial] = get_poprate(
                    spikes[celltype],
                    tstart,
                    t_window,
                    params["tres"]
                )
            ind += 1
    
    return poprates


def get_poprate(spikes, tstim, tstop, tres):
    tbins = int(tstop / tres)
    poprate = np.zeros(tbins)
    for ID in spikes:
        idx_poststim = np.all([spikes[ID] >= tstim, spikes[ID] < tstim + tstop], axis=0)
        idx_spikes = ((spikes[ID][idx_poststim] - tstim) / tres).astype(int)
        for idx in idx_spikes:
            poprate[idx] += 1
    
    return poprate


def get_filtered_poprates_all(poprates, tres, bandpass_cutoff_lo, bandpass_cutoff_hi):
    popsmooths = {}
    for state in poprates:
        popsmooths[state] = np.zeros(poprates[state].shape)
        n_amps, n_trials = poprates[state].shape[:2]
        for amp in range(n_amps):
            for trial in range(n_trials):
                popsmooths[state][amp][trial] = filter_poprate(
                    poprates[state][amp][trial], 
                    tres, 
                    bandpass_cutoff_lo, 
                    bandpass_cutoff_hi
                )

    return popsmooths


def filter_poprate(poprate, tres, cutoff_lo, cutoff_hi):
    Fs = 1000 / tres
    filtered = butter_bandpass_filter(poprate, cutoff_lo, cutoff_hi, Fs)
    return filtered


def get_firingrates_all(spikes, analysis_tstart, stim_delay, stim_times, trial_interval, poststim_tstart):
    firingrates = {}
    for state in spikes:
        # Calculate the total time over which the spikes are counted
        # Include the time before stimulation is first applied
        t_total = stim_delay - analysis_tstart
        # Include the time between stimuli during which spikes are counted
        t_interval = trial_interval[state] - poststim_tstart
        for subj in stim_times[state]:
            t_total += stim_times[state][subj].size * t_interval
        # Convert to seconds
        t_total /= 1000
        firingrates[state] = {}
        for celltype in spikes[state]:
            if "input" not in celltype:
                n_spikes = np.zeros(len(spikes[state][celltype]))
                for ind, ID in enumerate(spikes[state][celltype]):
                    # Counting spikes that occur before first stimulus
                    idx = np.all(
                        [
                            spikes[state][celltype][ID] >= analysis_tstart,
                            spikes[state][celltype][ID] < stim_delay,
                        ],
                        axis=0,
                    )
                    n_spikes[ind] = len(spikes[state][celltype][ID][idx])
                    
                    # Counting spikes that occur between stimuli
                    for subj in stim_times[state]:
                        for stim_time in stim_times[state][subj].flatten():
                            tstart = stim_time + poststim_tstart
                            tstop = stim_time + trial_interval[state]
                            idx = np.all(
                                [
                                    spikes[state][celltype][ID] >= tstart,
                                    spikes[state][celltype][ID] < tstop,
                                ],
                                axis=0,
                            )
                            n_spikes[ind] += len(spikes[state][celltype][ID][idx])
                
                firingrates[state][celltype] = n_spikes / t_total
    
    return firingrates


def get_time_to_I1_wave(popsmooths, tres, ind_subj_start, num_amps, delay_direct, tstop_dwave, tstop_I1wave):
    state = "resting"
    t_I1wave_pop = {}
    for state in popsmooths:
        t_I1wave_pop[state] = []
        for subj in range(len(ind_subj_start)):
            istart = ind_subj_start[subj]
            istop = istart + num_amps[state][subj]
            
            # Compute the mean response across trials
            tms_response_mean = popsmooths[state][istart:istop].mean(axis=1)
            
            # Find first peak in response
            t = np.arange(tms_response_mean.shape[1]) * tres
            t_I1wave_pop[state].append(np.zeros(num_amps[state][subj]))
            for amp in range(num_amps[state][subj]):
                # Look for I1-wave based on typical inter-peak-intervals for D/I-waves
                tstart_I1 = delay_direct + tstop_dwave
                tstop_I1 = tstart_I1 + tstop_I1wave
                
                idx_t = np.all([t >= tstart_I1, t < tstop_I1], axis=0)
                if tms_response_mean[amp][idx_t].max() > 0:
                    idx_dwave = np.argmax(tms_response_mean[amp][idx_t])
                    t_I1wave_pop[state][subj][amp] = t[idx_t][idx_dwave]
                else:
                    t_I1wave_pop[state][subj][amp] = tstart_I1
    
    return t_I1wave_pop


def calc_props_tms_acute(sim_results, opt_args, props):
    state = "resting"
    
    # Index for storing I-wave properties 
    for subj in range(opt_args["num_subjects"]):
        istart = opt_args["ind_subj_start"][subj]
        n_amps = opt_args["num_amps"][state][subj]
        for amp in range(n_amps):
            t_I1wave = sim_results["t_I1wave"][state][subj][amp]
            popsmooth_mean = np.mean(sim_results["popsmooths"][state][istart + amp], axis=0)
            t = opt_args["tres"] * np.arange(popsmooth_mean.size)
            
            # Find peaks and troughs
            peaks, tpeaks, minima, tminima = get_peaks_iwaves(
                popsmooth_mean,
                t,
                t_I1wave,
                opt_args["I1wave_offset"],
                opt_args["Dwave_offset"]
            )
            
            # Make sure activity after I-waves has small peaks
            if (
                len(tpeaks)
                >= opt_args["total_waves"]
            ):
                t_poststart = (
                    t_I1wave
                    + opt_args["targets"]["tpeaks_iwave"][istart + amp][-1]
                    + 0.5 # TODO: Move this parameter into a settings file
                )
                t_poststop = t_poststart + 10 # TODO: Move this parameter into a settings file
                idx_post = np.all([
                    t >= t_poststart,
                    t < t_poststop
                ], axis=0)
                props["postmin_iwave"][istart + amp] = np.max(popsmooth_mean[idx_post])
            
            # Remove time to first wave from timings
            tpeaks -= t_I1wave
            tminima -= t_I1wave
            
            # Limit data to desired number of I-waves
            if (len(peaks) < opt_args["total_waves"]):
                props["peaks_iwave"][istart + amp][:len(peaks)] = peaks
            else:
                props["peaks_iwave"][istart + amp] = peaks[:opt_args["total_waves"]]

            if (len(minima) < opt_args["total_waves"]):
                props["minima_iwave"][istart + amp][:len(minima)] = minima
            else:
                props["minima_iwave"][istart + amp] = minima[:opt_args["total_waves"]]
            
            if (len(tpeaks) < opt_args["total_waves"]):
                props["tpeaks_iwave"][istart + amp][:len(tpeaks)] = tpeaks
            else:
                props["tpeaks_iwave"][istart + amp] = tpeaks[:opt_args["total_waves"]]
            if "tminima_iwave" in props:
                if (len(tminima) < opt_args["total_waves"]):
                    props["tminima_iwave"][istart + amp][:len(tminima)] = tminima
                else:
                    props["tminima_iwave"][istart + amp] = tminima[:opt_args["total_waves"]]
        
        # Subject normalization
        # Get largest I1 amplitude based on experimental data
        ind_I1max_amp = istart + opt_args["ind_I1max_amp"][subj]
        peak_I1wave = props["peaks_iwave"][ind_I1max_amp][opt_args["ind_I1"]]
        
        # Normalize measurements
        props["peaks_iwave"][istart:istart + n_amps] /= peak_I1wave
        props["minima_iwave"][istart:istart + n_amps] /= peak_I1wave
        props["postmin_iwave"][istart:istart + n_amps] /= peak_I1wave
        
        # Save peak_I1wave to un-normalize before saving the data
        # for future sensitivity analysi
        props["amp_I1"][subj] = peak_I1wave
    
    return props


def get_peaks_iwaves(popsmooth_mean, t, t_I1wave, I1wave_offset, Dwave_offset):
    # Identify magnitudes and timings of peaks and troughs
    peaks, tpeaks = get_localmax(
        popsmooth_mean, t
    )
    minima, tminima = get_localmax(
        -popsmooth_mean, t
    )
    
    # Find timings for peaks for the I-wave response
    idx_postI1wave = tpeaks >= (
        t_I1wave - I1wave_offset
    )
    peaks_i = peaks[idx_postI1wave]
    tpeaks_i = tpeaks[idx_postI1wave]
    
    # Find all minima starting from just after the first peak
    if len(tpeaks_i) > 0:
        # Get indices of minima that occur after the first peak
        idx_peak1 = np.arange(len(tminima))[
            tminima > tpeaks_i[0]
        ]
        if len(idx_peak1) > 0:
            minima_i = minima[idx_peak1]
            tminima_i = tminima[idx_peak1]
        else:
            minima_i = 1000 * np.ones(minima.size)
            tminima_i = 1000 * np.ones(tminima.size)
    else:
        minima_i = 1000 * np.ones(minima.size)
        tminima_i = 1000 * np.ones(tminima.size)
    
    idx_preI1wave = t < (t_I1wave - 0.5)
    idx_max = np.argmax(popsmooth_mean[idx_preI1wave])
    peaks_d = [np.max(popsmooth_mean[idx_preI1wave])]
    tpeaks_d = [t[idx_preI1wave][idx_max]]
    idx_between_D_and_I1 = np.all(
        [
            tminima > tpeaks_d[0],
            tminima < t_I1wave
        ],
        axis=0
    )
    if len(tminima[idx_between_D_and_I1]) > 0:
        tminima_d = [tminima[idx_between_D_and_I1][0]]
    else:
        tminima_d = [1000]
    idx_preI1wave_minima = np.all(
        [
            t >= tpeaks_d[0],
            t < (t_I1wave - 0.5)
        ],
        axis=0
    )
    minima_d = [-np.min(popsmooth_mean[idx_preI1wave_minima])]
    
    # Combine D-wave and I-waves
    peaks_total = np.hstack([peaks_d, peaks_i])
    tpeaks_total = np.hstack([tpeaks_d, tpeaks_i])
    minima_total = np.hstack([minima_d, minima_i])
    tminima_total = np.hstack([tminima_d, tminima_i])
    return peaks_total, tpeaks_total, minima_total, tminima_total


def calc_syn_noise_weight(sim_results, opt_args, props):
    # Get syn_noise_weight parameters
    ind_save = 0
    for ind_param, param in enumerate(opt_args["parameters"]):
        if "syn_noise_weight" in param:
            props["syn_noise_weight"][ind_save] = sim_results["chromosome"][ind_param]
            ind_save += 1

    return props


def calc_iclamp_mean(sim_results, opt_args, props):
    # Get iclamp_means
    ind_save = 0
    for ind_param, param in enumerate(opt_args["parameters"]):
        if "iclamp_mean" in param[0]:
            props["iclamp_mean"][ind_save] = np.abs(sim_results["chromosome"][ind_param])
            ind_save += 1

    return props


def calc_ISIs_mean(sim_results, opt_args, props):
    celltypes = [celltype for celltype in opt_args['cell_types'] if 'input' not in celltype]
    n_celltypes = len(celltypes)
    ISIs_mean = np.zeros(n_celltypes)
    if "spikes_resting" in sim_results:
        spikes = sim_results["spikes_resting"]
    else:
        spikes = sim_results["spikes"]["resting"]
    
    t_window = opt_args["stim_delay"] - opt_args["analysis_tstart"]
    ISIs_mean = np.zeros(n_celltypes)
    for ind, celltype in enumerate(celltypes):
        for ID in spikes[celltype]:
            if len(spikes[celltype][ID]) >= 2:
                ISIs = np.diff(spikes[celltype][ID])
                ISIs_mean[ind] += np.mean(ISIs)
            else:
                ISIs_mean[ind] += t_window
        
        ISIs_mean[ind] /= len(spikes[celltype])
    
    props['ISI_mean'] = ISIs_mean
    return props


def calc_ISIs_std(sim_results, opt_args, props):
    celltypes = [celltype for celltype in opt_args['cell_types'] if 'input' not in celltype]
    n_celltypes = len(celltypes)
    if "spikes_resting" in sim_results:
        spikes = sim_results["spikes_resting"]
    else:
        spikes = sim_results["spikes"]["resting"]
    
    t_window = opt_args["stim_delay"] - opt_args["analysis_tstart"]
    ISIs_std = np.zeros(n_celltypes)
    for ind, celltype in enumerate(celltypes):
        for ID in spikes[celltype]:
            if len(spikes[celltype][ID]) > 3:
                ISIs = np.diff(spikes[celltype][ID])
                ISIs_std[ind] += np.std(ISIs)
            else:
                ISIs_std[ind] += t_window
        
        ISIs_std[ind] /= len(spikes[celltype])
        if ISIs_std[ind] == 0:
            ISIs_std[ind] = 1000
    
    props['ISI_stdev'] = ISIs_std
    return props


def calc_across_cell_ISI_std(sim_results, opt_args, props):
    celltypes = [celltype for celltype in opt_args['cell_types'] if 'input' not in celltype]
    n_celltypes = len(celltypes)
    if "spikes_resting" in sim_results:
        spikes = sim_results["spikes_resting"]
    else:
        spikes = sim_results["spikes"]["resting"]
    t_window = opt_args["stim_delay"] - opt_args["analysis_tstart"]
    across_cell_ISI_std = np.zeros(n_celltypes)
    for ind, celltype in enumerate(celltypes):
        ISIs_celltype = []
        for ID in spikes[celltype]:
            if len(spikes[celltype][ID]) >= 2:
                ISIs = np.diff(spikes[celltype][ID])
                ISIs_celltype.append(np.mean(ISIs))
            else:
                ISIs_celltype.append(t_window)

        across_cell_ISI_std[ind] = np.std(ISIs_celltype)

    props["ISI_std_across_cell"] = across_cell_ISI_std
    return props


def calc_firingrates_mean(sim_results, opt_args, props):
    celltypes = [celltype for celltype in opt_args['cell_types'] if 'input' not in celltype]
    n_celltypes = len(celltypes)
    firingrates_mean = np.zeros(n_celltypes)
    if "spikes_resting" in sim_results:
        spikes = sim_results["spikes_resting"]
    else:
        spikes = sim_results["spikes"]["resting"]
    t_window = opt_args["stim_delay"] - opt_args["analysis_tstart"]
    firingrates_mean = np.zeros(n_celltypes)
    for ind, celltype in enumerate(celltypes):
        for ID in spikes[celltype]:
            firingrate = len(spikes[celltype][ID]) / t_window * 1000
            firingrates_mean[ind] += firingrate
        
        firingrates_mean[ind] /= len(spikes[celltype])
    
    props['firingrate_mean'] = firingrates_mean
    return props


def calc_firingrates_std(sim_results, opt_args, props):
    celltypes = [celltype for celltype in opt_args['cell_types'] if 'input' not in celltype]
    n_celltypes = len(celltypes)
    spikes = sim_results["spikes"]["resting"]
    firingrates_std = np.zeros(n_celltypes)
    for ind, celltype in enumerate(celltypes):
        for ID in spikes[celltype]:
            t_idx = np.all(
                [
                    spikes[celltype][ID] >= opt_args["analysis_tstart"],
                    spikes[celltype][ID] < opt_args["stim_delay"],
                ],
                axis=0
            )
            if len(spikes[celltype][ID][t_idx]) > 3:
                firingrates = np.diff(spikes[celltype][ID][t_idx])
                firingrates_std[ind] += np.std(firingrates)
        
        firingrates_std[ind] /= len(spikes[celltype])
    
    props['firingrate_stdev'] = firingrates_std
    return props


def calc_ISI_firingrate(sim_results, opt_args):
    celltypes = [celltype for celltype in opt_args['cell_types'] if 'input' not in celltype]
    n_celltypes = len(celltypes)
    firingrate_mean = np.zeros(n_celltypes)
    if "spikes_resting" in sim_results:
        spikes = sim_results["spikes_resting"]
    else:
        spikes = sim_results["spikes"]["resting"]
    t_window = opt_args["stim_delay"] - opt_args["analysis_tstart"]
    for ind, celltype in enumerate(celltypes):
        for ID in spikes[celltype]:
            if len(spikes[celltype][ID]) >= 2:
                ISIs = np.diff(spikes[celltype][ID])
                ISI_mean = np.mean(ISIs)
            else:
                ISI_mean = t_window
            firingrate_mean[ind] += 1000 / ISI_mean

        firingrate_mean[ind] /= len(spikes[celltype])

    props['ISI_firingrate_mean'] = ISIs_mean
    return props


def get_restingstate_spikes(sim_results, opt_args):
    celltypes = [celltype for celltype in opt_args['cell_types'] if 'input' not in celltype]
    spikes = sim_results["spikes"]["resting"]
    spikes_resting = {}
    for ind, celltype in enumerate(celltypes):
        spikes_resting[celltype] = {}
        for ID in spikes[celltype]:
            t_idx = np.all(
                [
                    spikes[celltype][ID] >= opt_args["analysis_tstart"],
                    spikes[celltype][ID] < opt_args["stim_delay"],
                ],
                axis=0
            )
            spikes_resting[celltype][ID] = spikes[celltype][ID][t_idx]

    return spikes_resting


def get_localmax(popsmooth_shifted, t_shifted):
    # Find all positive and positive-going indices
    idx_pos = []
    flag_pos = 1
    diff = np.diff(popsmooth_shifted)
    for ii in range(len(diff)):
        if (diff[ii] > 0) and (popsmooth_shifted[ii + 1] > 0) and flag_pos:
            idx_pos.append(ii)
            flag_pos = 0
        if diff[ii] < 0:
            flag_pos = 1

    # Compute peak
    peaks = []
    idx_peak = []
    for ii in range(len(idx_pos)):
        if ii < (len(idx_pos) - 1):
            wave = popsmooth_shifted[idx_pos[ii] : idx_pos[ii + 1]]
        else:
            wave = popsmooth_shifted[idx_pos[ii] :]

        peaks.append(np.max(wave))
        idx_peak.append(np.argmax(wave) + idx_pos[ii])

    peaks = np.array(peaks)
    idx_peak = np.array(idx_peak).astype(int)

    # Compute time to peaks
    tpeaks = t_shifted[idx_peak]

    return peaks, tpeaks


def calc_response_cv(sim_results, opt_args, props):
    popsmooth = sim_results["popsmooths"]
    t_window = 10 #opt_args["trange_iwaves"]
    num_samples = int(t_window / opt_args["tres"])
    std = popsmooth["resting"][0].std(axis=0)[:num_samples]
    mean = popsmooth["resting"][0].mean(axis=0)[:num_samples]
    # Dealing with zeros in the mean
    mean[mean == 0] = 1
    cv = std / np.abs(mean)
    props["response_cv"] = np.array([np.mean(cv)])
    return props


def calc_synchrony_baseline(sim_results, opt_args, props):
    celltypes = [celltype for celltype in opt_args['cell_types'] if 'input' not in celltype]
    spikes = sim_results["spikes"]
    tstart = opt_args["analysis_tstart"]
    tstop = opt_args["stim_delay"]
    tres = opt_args["tres"]
    cutoff_lo = opt_args["bandpass_cutoff_lo"]
    cutoff_hi = opt_args["bandpass_cutoff_hi"]
    # Lowcut set to regime at which oscillations should be maximal (alpha)
    lowcut = 20 # TODO: Move to a settings file
    tbins = int((tstop - tstart) / tres)
    Fs = 1000 / tres
    for state in spikes:
        synchrony_baseline = np.ones(len(celltypes))
        for ind_celltype, celltype in enumerate(celltypes):
            filtered = np.zeros((len(spikes[state][celltype]), tbins))
            for ind_cell, ID in enumerate(spikes[state][celltype]):
                poprate = np.zeros(tbins)
                idx_poststim = np.all([spikes[state][celltype][ID] >= tstart, spikes[state][celltype][ID] <  tstop], axis=0)
                idx_spikes = ((spikes[state][celltype][ID][idx_poststim] - tstart) / tres).astype(int)
                for idx in idx_spikes:
                    poprate[idx] += 1
                
                filtered[ind_cell] = butter_lowpass_filter(poprate, lowcut, Fs)
            
            synchrony_baseline[ind_celltype] = get_synchrony(filtered)[0]
        
        name_prop = "synchrony_baseline_" + state
        if name_prop in props:
            props[name_prop] = synchrony_baseline
    
    return props


def butter_lowpass_filter(data, lowcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = signal.butter(order, low, btype="lowpass", output="sos")
    y = signal.sosfiltfilt(sos, data)
    return y


def get_synchrony(popsmooths):
    phase = np.zeros(popsmooths.shape)
    for ii in range(len(popsmooths)):
        ht = signal.hilbert(popsmooths[ii])
        phase[ii][: len(ht)] = np.angle(ht)

    phase_group = np.mean(np.exp(1j * phase), axis=0)
    phase_group = np.angle(phase_group)
    phase_rel = phase - phase_group
    phase_rel_mean = np.mean(np.exp(1j * phase_rel), axis=1)
    return np.array([np.abs(phase_rel_mean).mean()])
