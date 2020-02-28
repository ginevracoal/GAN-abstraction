import numpy as np
from numpy.random import randint, random
import os
import shutil
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

filename = "../../data/Repressilator/Repressilator_training_set_full.pickle"
with open(filename, 'rb') as handle:
    dataset_dict = pickle.load(handle)

Y_par = dataset_dict["Y_par"]
Y_s0_full = dataset_dict["Y_s0"]
X_full = dataset_dict["X"]
n_training_points, n_steps, n_species_glob = X_full.shape
n_indip_species = n_species_glob-3
print("N STEPS: ", n_steps)

X = np.zeros((n_training_points, n_steps, n_indip_species))

X[:,:, 0] = X_full[:,:,0]
X[:,:, 1] = X_full[:,:,2]
X[:,:, 2] = X_full[:,:,4]
X[:,:, 3] = X_full[:,:,1]
X[:,:, 4] = X_full[:,:,3]
X[:,:, 5] = X_full[:,:,5]

init_states = np.vstack((Y_s0_full[:,1], Y_s0_full[:,3], Y_s0_full[:,5], Y_s0_full[:,6], Y_s0_full[:,7], Y_s0_full[:,8])).T #G1_on, G2_on, G3_on, P1, P2, P3
print("SHAPE init_states: ", init_states.shape)

indip_filename = "../../data/Repressilator/Repressilator_training_set.pickle"

ord_dataset_dict = {"X": X, "Y_par": Y_par, "Y_s0": init_states}
with open(indip_filename, 'wb') as handle:
	pickle.dump(ord_dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
'''
_, p = Y_par.shape


timeline = np.linspace(0,n_steps,n_steps)

fig1, ax1 = plt.subplots(1, 3, figsize=(12, 8))
for ggg in range(0,1000,100):
	ax1[0].plot(timeline, X_ord[ggg,:,3], "r")
	ax1[1].plot(timeline, X_ord[ggg,:,4], "b")
	ax1[2].plot(timeline, X_ord[ggg,:,5], "g")
	
ax1[0].set_xlabel("time")	
ax1[0].set_title("P1")
ax1[1].set_xlabel("time")
ax1[1].set_title("P2")
ax1[2].set_xlabel("time")
ax1[2].set_title("P3")
plt.tight_layout()
string_name_1 = "Plots/proteins_oscillating.png"
fig1.savefig(string_name_1)

for ttt in range(0,1000,50):
	print("PARAMETERS for ttt: ", Y_par[ttt])
	fig, ax = plt.subplots(1, 2, figsize=(12, 8))
	ax[0].plot(timeline, X_ord[ttt,:,0], "r", label = "G1")
	ax[0].plot(timeline, X_ord[ttt,:,1], "b", label = "G2")
	ax[0].plot(timeline, X_ord[ttt,:,2], "g", label = "G3")
	ax[0].legend()
	ax[0].set_xlabel("time")
	ax[0].set_ylabel("genes")
	ax[0].set_title("Genes")
	ax[1].plot(timeline, X_ord[ttt,:,3], "r", label = "P1")
	ax[1].plot(timeline, X_ord[ttt,:,4], "b", label = "P2")
	ax[1].plot(timeline, X_ord[ttt,:,5], "g", label = "P3")
	ax[1].set_xlabel("time")
	ax[1].set_ylabel("proteins")
	ax[1].legend()
	plt.tight_layout()
	string_name = "Plots/genes_and_proteins_{}.png".format(ttt)
	fig.savefig(string_name)
'''