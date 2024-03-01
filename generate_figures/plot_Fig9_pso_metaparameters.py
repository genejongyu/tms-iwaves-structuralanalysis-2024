# Plot evolution of PSO metaparameters for Fig 9

from get_paths import *

import numpy as np
import matplotlib.pyplot as plt
from colors_paultol import *

save_flag = 0

def sigmoid(gens, n_gen, a, b, amp, offset):
    return amp/(1+np.exp((a*gens-b*n_gen)/n_gen))+offset

# Set parameters
n_gen = 300
gens = np.arange(n_gen)

weights_cog = sigmoid(gens, n_gen, 20, 7.2, 2.4, 0.1)
weights_soc = sigmoid(gens, n_gen, 20, 7.2, -2.4, 2.5)
weights_inert = sigmoid(gens, n_gen, 10, 2.4, 2.15, 0.5)
weights_gain = sigmoid(gens, n_gen, 10, 2.4, 1.5, 0.5)
weights_noise = sigmoid(gens, n_gen, 15, 4.2, 0.195, 0.005)

fig = plt.figure(figsize=(6, 5))
_=plt.title('Evolution of Weights')
_=plt.plot(gens, weights_cog, linewidth=4, label='Cognitive Weights', color=cset_bright[0])
_=plt.plot(gens, weights_soc, linewidth=4, label='Social Weights', color=cset_bright[1])
_=plt.plot(gens, weights_inert, linewidth=4, label='Inertia Weights', color=cset_bright[2])
_=plt.plot(gens, weights_gain, linewidth=4, label='Gain Weights', color=cset_bright[3])
_=plt.plot(gens, weights_noise, linewidth=4, label='Noise Weights', color=cset_bright[4])
_=plt.ylabel("Weight")
_=plt.xlabel("Generation")
_=plt.xlim(gens[0], gens[-1])
_=plt.legend()
_=plt.grid()

if save_flag:
    fig.savefig("Fig9_pso_metaparameters.tiff", dpi=600)

plt.show()