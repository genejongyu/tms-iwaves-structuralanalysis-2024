# Save neuron positions in motor cortical macrocolumn model

from os.path import join
import sys
dir_data = join("..", "data")
dir_model_parameters = "settings_2013_31_120RMT"
sys.path.append(dir_model_parameters)

from net_esser import net_esser
import settings_main
import pickle

# Initialize dummy delay_jitter to instantiate network
delay_jitter = {}
for pre in settings_main.params["con"]["p"]:
    delay_jitter[pre] = {}
    for post in settings_main.params["con"]["p"][pre]:
        delay_jitter[pre][post] = 1

# Instantiate network
net = net_esser(
    settings_main.params, 
    seed_con=0, 
    seed_tms=0, 
    seed_input=0, 
    tstop=1, 
    delay_jitter=delay_jitter, 
    tstart=0
)

# Save data
fpath = join(dir_data, "data_network_locations2.pickle")
save_data = {}
save_data["x"] = net.x_locs
save_data["y"] = net.y_locs
save_data["z"] = net.z_locs
with open(fpath, "wb") as f:
    pickle.dump(save_data, f)
