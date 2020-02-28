import numpy as np
from numpy.random import randint, random
import stochpy
import pandas as pd
import os
import shutil
from tqdm import tqdm
import pickle


N = 100 # popul size

filename = "../../data/SIR/SIR_training_set.pickle"
with open(filename, 'rb') as handle:
    dataset_dict = pickle.load(handle)

Y_par = dataset_dict["Y_par"]
Y_s0 = dataset_dict["Y_s0"]
X = dataset_dict["X"]

n_training_points, T, m = X.shape
_, p = Y_par.shape
print("N PARAM: ", Y_par)

input_trajs =  np.zeros((n_training_points, T, m+1))
init_states = np.zeros((n_training_points, m+1))

init_states[:,:m] = Y_s0
init_states[:,m] = N-Y_s0[:,0]-Y_s0[:,1]

input_trajs[:,:,:m] = X
for i in range(n_training_points):
	input_trajs[i,:,m] = N - X[i,:,0]-X[i,:,1]

