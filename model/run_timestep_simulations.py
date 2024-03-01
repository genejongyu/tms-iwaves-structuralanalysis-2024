# Script to run simulations to evaluate effect of time-step
# on simulation accuracy and speed

from os.path import join
from get_paths import *

import matplotlib.pyplot as plt
import numpy as np
import cell_esser
import time as cookie
import lib_neuronsim
import multiprocessing
import multiprocessing.connection as conn
import pickle
# If larger data are being placed in the queue,
# then BUFSIZE will need to be increased
conn.BUFSIZE = 2 ** 24
plt.rcParams["font.family"] = "Georgia"

def Laplace(x,tau):
    return np.exp(-np.abs(x)/tau)

def vanRossum_distance(spikes1, spikes2, tau):
    allspikes1,allspikes2 = np.meshgrid(spikes1,spikes2)
    diff = allspikes1.flatten()-allspikes2.flatten()
    sum1 = np.sum(Laplace(diff,tau))
    
    spikes1_self1,spikes1_self2 = np.meshgrid(spikes1,spikes1)
    diff = spikes1_self1.flatten()-spikes1_self2.flatten()
    sum2 = np.sum(Laplace(diff,tau))
    
    spikes2_self1,spikes2_self2 = np.meshgrid(spikes2,spikes2)
    diff = spikes2_self1.flatten()-spikes2_self2.flatten()
    sum3 = np.sum(Laplace(diff,tau))
    
    return np.sqrt(-2*sum1 + (sum2+sum3))

def worker(queue, cell_func, celltype, rangen, FR, delay, tstop, tres, cvode_flag=0):
    t, v, spikes = lib_neuronsim.sim_poisson_esser(cell_func, celltype, rangen, FR, delay, tstop, tres, cvode_flag)
    queue.put(t)
    queue.put(v)
    queue.put(spikes)

if __name__ == "__main__":
    save_flag = 1
    print("save_flag: {}".format(save_flag))
    fpath = join(dir_data, "data_timestep.pickle")
    
    dur = 20000
    delay = 50
    num_trials = 1
    FR = 1000
    tstop = dur + delay
    celltype = 'L5'
    treses = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    queue = multiprocessing.Queue()
    
    ts = {}
    vs = {}
    spiketimes = {}
    tfin = {}
    for tres in treses:
        ts[tres] = []
        vs[tres] = []
        spiketimes[tres] = []
        tfin[tres] = np.zeros(num_trials)
        for ii in range(num_trials):
            print(tres, ii)
            rangen = np.random.default_rng(seed=ii)
            ST = cookie.time()
            if tres == 0:
                p = multiprocessing.Process(target=worker, args=(queue, cell_esser.cell_esser, celltype, rangen, FR, delay, tstop, tres, 1))
            else:
                p = multiprocessing.Process(target=worker, args=(queue, cell_esser.cell_esser, celltype, rangen, FR, delay, tstop, tres, 0))
            
            p.start()
            
            ts[tres].append(queue.get())
            vs[tres].append(queue.get())
            spiketimes[tres].append(queue.get())
            
            p.join()
            
            ET = cookie.time()
            tfin[tres][ii] = ET-ST
        
        print("Took {} minutes".format(tfin[tres].sum() / 60))
    
    print('Total time: minutes', sum([tfin[tres].sum() for tres in tfin]) / 60)
    
    tau_vr = 500
    num_spikes = np.zeros((len(treses), num_trials))
    mean_ISI = np.zeros((len(treses), num_trials))
    stdev_ISI = np.zeros((len(treses), num_trials))
    vdiff = np.zeros((len(treses)-1, num_trials))
    vdiff0 = np.zeros((len(treses)-1, num_trials))
    vr_dist = np.zeros((len(treses)-1, num_trials))
    vr_dist0 = np.zeros((len(treses)-1, num_trials))
    for ii in range(num_trials):
        for ind, tres in enumerate(treses):
            num_spikes[ind][ii] = spiketimes[tres][ii].size
            mean_ISI[ind][ii] = np.mean(np.diff(spiketimes[tres][ii]))
            stdev_ISI[ind][ii] = np.std(np.diff(spiketimes[tres][ii]))
            if ind == 0:
                idx = np.all([ts[tres][ii] >= delay, ts[tres][ii] < delay+dur], axis=0)
                v0 = vs[tres][ii][idx]
                t0 = ts[tres][ii][idx]
                idx_spike0 = np.all([spiketimes[tres][ii] >= delay, spiketimes[tres][ii] < delay+dur], axis=0)
                spikes0 = spiketimes[tres][ii][idx_spike0]
            else:
                idx_prev = np.all([ts[treses[ind-1]][ii] >= delay, ts[treses[ind-1]][ii] < delay+dur], axis=0)
                v_prev = vs[treses[ind-1]][ii][idx_prev]
                idx = np.all([ts[tres][ii] >= delay, ts[tres][ii] < delay+dur], axis=0)
                
                idx = np.all([ts[tres][ii] >= delay, ts[tres][ii] < delay+dur], axis=0)
                v = vs[tres][ii][idx]
                
                v_interp_prev = np.interp(ts[tres][ii][idx], ts[treses[ind-1]][ii][idx_prev], v_prev)
                
                vdiff[ind-1][ii] = np.sqrt(np.mean((v - v_interp_prev)**2)/np.mean(v_interp_prev**2))
                
                v_interp0 = np.interp(ts[tres][ii][idx], t0, v0)
                vdiff0[ind-1][ii] = np.sqrt(np.mean((v - v_interp0)**2)/np.mean(v_interp0**2))
                
                idx_spike_prev = np.all([spiketimes[treses[ind-1]][ii] >= delay, spiketimes[treses[ind-1]][ii] < delay+dur], axis=0)
                idx_spike = np.all([spiketimes[tres][ii] >= delay, spiketimes[tres][ii] < delay+dur], axis=0)
                
                spikes_prev = spiketimes[treses[ind-1]][ii][idx_spike_prev]
                spikes_current = spiketimes[tres][ii][idx_spike]
                
                vr_dist[ind-1][ii] = vanRossum_distance(spikes_prev, spikes_current, tau_vr)
                vr_dist0[ind-1][ii] = vanRossum_distance(spikes0, spikes_current, tau_vr)
    
    cv_ISI = stdev_ISI / mean_ISI
    
    # Save data
    if save_flag:
        save_data = {}
        save_data["treses"] = treses
        save_data["spiketimes"] = spiketimes
        save_data["num_spikes"] = num_spikes
        save_data["mean_ISI"] = mean_ISI
        save_data["cv_ISI"] = cv_ISI
        save_data["nrmse"] = vdiff0
        save_data["vr_dist"] = vr_dist0
        save_data["vs"] = {} # Save one instance of voltages for each time-step
        save_data["ts"] = {}
        save_data["v_tstart"] = 6370
        save_data["v_tstop"] = 6400
        for tres in treses:
            idx = np.all([
                ts[tres][0] >= save_data["v_tstart"],
                ts[tres][0] < save_data["v_tstop"]
            ], axis=0)
            save_data["vs"][tres] = vs[tres][0][idx]
            save_data["ts"][tres] = ts[tres][0][idx]
        with open(fpath, "wb") as f:
            pickle.dump(save_data, f)

    plt.show()