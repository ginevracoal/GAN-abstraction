import numpy as np
from numpy.random import randint, random
import os
import shutil
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

TRAINING = True

if TRAINING:
	filename = "../../data/GRN/GRN_fixed_training_set.pickle"
	with open(filename, 'rb') as handle:
	    dataset_dict = pickle.load(handle)

	#Y_par = dataset_dict["Y_par"]
	Y_s0 = dataset_dict["Y_s0"]
	X = dataset_dict["X"]
	n_training_points, n_steps, n_species_glob = X.shape
	n_indip_species = n_species_glob-3
	print("N STEPS: ", n_steps)

else:
	filename = "../../data/GRN/GRN_fixed_validation_set.pickle"
	with open(filename, 'rb') as handle:
	    dataset_dict = pickle.load(handle)

	#Y_par = dataset_dict["Y_par"]
	Y_s0 = dataset_dict["Y_s0"]
	X = dataset_dict["X"]
	n_val_points, n_trajs, n_steps, n_species_glob = X.shape
	n_indip_species = n_species_glob-3

#_, p = Y_par.shape

print(X)
print("X SHAPE: ", X.shape)
timeline = np.linspace(0,n_steps,n_steps)

fig1, ax1 = plt.subplots(1, 3, figsize=(12, 8))
for ggg in range(6):
	ax1[0].plot(timeline, X[ggg,:,0], "r")
	ax1[1].plot(timeline, X[ggg,:,1], "b")
	ax1[2].plot(timeline, X[ggg,:,2], "g")
	
ax1[0].set_xlabel("time")	
ax1[0].set_title("P1")
ax1[1].set_xlabel("time")
ax1[1].set_title("P2")
ax1[2].set_xlabel("time")
ax1[2].set_title("P3")
plt.tight_layout()
string_name_1 = "Plots/gnr_traj.png"
fig1.savefig(string_name_1)

'''
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