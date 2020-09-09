import numpy as np
from numpy.random import randint, random
import stochpy
import pandas as pd
import os
import shutil
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

N = 100 # popul size

train_filename = "../../data/eSIR/eSIR_training_set_kernel_fixed_param_test.pickle"
val_filename = "../../data/eSIR/eSIR_validation_set_kernel_fixed_param.pickle"

with open(train_filename, 'rb') as handle:
    train_set_dict = pickle.load(handle)
with open(val_filename, 'rb') as handle:
    val_set_dict = pickle.load(handle)


Y_s0_train = train_set_dict["Y_s0"]
X_train = train_set_dict["X"]

Y_s0_val = val_set_dict["Y_s0"]
X_val = val_set_dict["X"]

xx_tr = np.squeeze(X_train, 1)
T = 500


for i in range(10):
	print(Y_s0_train[i*T])
	plt.hist(xx_tr[i*T:(i+1)*T], bins = 100)
	name = "train_{}.png".format(i)
	plt.savefig(name)
	plt.close()
