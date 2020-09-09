# example of a wgan for generating handwritten digits
from numpy import mean, ones
from numpy.random import randn, rand, randint
import numpy as np
from tensorflow.keras.backend import expand_dims, squeeze
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Concatenate, \
                                    Embedding, Flatten, Reshape, RepeatVector, Permute, \
                                    SeparableConv1D, Lambda, BatchNormalization, UpSampling1D
                              
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import RMSprop
import pickle
from keras import backend
from matplotlib import pyplot

from scipy.stats import wasserstein_distance
from conv_1d_trans import Conv1DTranspose




COND_SIZE = 1
TRAJ_LEN = 32
LATENT_DIM = 240
N_SPECIES = 2

HMAX = 50.0

CLIP_CONST = 0.01

C_LR = 0.00005
G_LR = 0.00005

Q = 4
N_CH = 256

embedding = "REPEAT" #"REPEAT" or "DENSE"

intermediate_plots = True

N_EPOCHS = 200
BATCH_SIZE = 256
N_CRITIC = 10
N_GEN = 1

import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import os

PLOTS_PATH = "Plots/Emb_"+embedding+"_Plots_EP={}_BAT={}_LRs={}_{}_N_CR={}_N_CH={}_Q={}_CC={}".format(N_EPOCHS,BATCH_SIZE,C_LR,G_LR, N_CRITIC, N_CH, Q, CLIP_CONST)
try:
    os.mkdir(PLOTS_PATH)
except OSError:
    print ("Creation of the directory %s failed" % PLOTS_PATH)
else:
    print ("Successfully created the directory %s " % PLOTS_PATH)

MODELS_PATH = "Models/LARGE_SavedModels_EP={}_BAT={}_LRs={}_{}_N_CR={}_N_CH={}_Q={}_CC={}".format(N_EPOCHS,BATCH_SIZE,C_LR,G_LR, N_CRITIC, N_CH, Q, CLIP_CONST)
try:
    os.mkdir(MODELS_PATH)
except OSError:
    print ("Creation of the directory %s failed" % MODELS_PATH)
else:
    print ("Successfully created the directory %s " % MODELS_PATH)

print_settings = True
if print_settings:
	print("NUM EPOCHS: ", N_EPOCHS)
	print("BATCH SIZE: ", BATCH_SIZE)
	print("N GEN: ", N_GEN)
	print("N CRIT", N_CRITIC)
	print("LATENT_DIM: ", LATENT_DIM)
	print("TRAJ LEN: ", TRAJ_LEN)
	print("CLIP CONST: ", CLIP_CONST)
	print("Initial state embedding: ", embedding)
	print("CRITIC LR: ", C_LR)
	print("GEN LR: ", G_LR)
	print("GEN Q: ", Q, ", N_CH: ", N_CH)

	print("PLOTS_PATH: ", PLOTS_PATH)
	print("MODELS_PATH: ", MODELS_PATH)
 
# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
 
# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)
 
# define the standalone critic model
def define_critic():
	
	traj = Input(shape=(TRAJ_LEN+1, N_SPECIES)) 
		

	HC = [64, 64]
	KC = [4, 4]
	SC = [2,2]
	# weight constraint
	const = ClipConstraint(CLIP_CONST)
	# downsample 
	x = Conv1D(HC[0], KC[0], strides=SC[0], padding='same', kernel_constraint=const)(traj)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	# downsample 
	x = Conv1D(HC[1], KC[1], strides=SC[1], padding='same', kernel_constraint=const)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	# downsample 
	#x = Conv1D(128, 4, strides=2, padding='same', kernel_constraint=const)(x)
	#x = BatchNormalization()(x)
	#x = LeakyReLU(alpha=0.2)(x)

	# scoring, linear activation
	x = Flatten()(x)
	outputs = Dense(1)(x)
	x = Dropout(0.4)(x)

	model = Model(inputs=traj, outputs=outputs)

	# compile model
	opt = RMSprop(lr=C_LR)
	model.compile(loss=wasserstein_loss, optimizer=opt)

	arch_critic = 'C_ARCH: H={}, K={}, S={}+LeakyRelu02'.format(HC, KC, SC)
	print(arch_critic)

	return model
 
def define_generator(latent_dim):
	#init = RandomNormal(stddev=0.02)

	noise = Input(shape=(latent_dim)) 
	n_nodes_n = N_CH * Q
	nv = Dense(n_nodes_n)(noise)
	#nv = LeakyReLU(alpha=0.2)(nv)
	nv = Reshape((Q, N_CH))(nv)

	init_states = Input(shape=(N_SPECIES))
	
	if embedding== "DENSE":
		n_nodes_i = Q * 1
		iv = Dense(n_nodes_i)(init_states)
		iv = Reshape((Q, 1))(iv)
	else:
		iv = RepeatVector(Q)(init_states)
		#iv = Reshape((Q, N_SPECIES))(iv)
	#iv = LeakyReLU(alpha=0.2)(iv)

	merge = Concatenate(axis=2)([iv,nv])

	HG = [128, 256, 128]
	KG = [4, 4, 4, 4]
	SG = [2, 2, 2]
	# upsample to 2*Q = 8
	x = Conv1DTranspose(HG[0], KG[0], padding = "same", strides = SG[0])(merge)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	
	# upsample to 4*Q = 16
	x = Conv1DTranspose(HG[1], KG[1], padding = "same", strides = SG[1])(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	
	# upsample to 8*Q = 32
	x = Conv1DTranspose(HG[2], KG[2], padding = "same", strides = SG[2])(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	
	# output
	outputs = Conv1D(N_SPECIES, KG[-1], activation='tanh', padding='same')(x)
	print("GEN OUTPUT: ", outputs)

	model = Model(inputs=[noise,init_states], outputs=outputs)

	arch_gen = 'G_ARCH: H={}, K={}, S={}+LeakyRelu02'.format(HG, KG, SG)
	print(arch_gen)

	return model

 
# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
	# make weights in the critic not trainable
	critic.trainable = False
	noise, init_states = generator.input
	gen_traj = generator.output

	in_st = Reshape((1,N_SPECIES))(init_states)
	merged_traj = Concatenate(axis=1)([in_st, gen_traj])

	gan_output = critic(merged_traj)

	model = Model(inputs=[noise, init_states], outputs=gan_output)

	# compile model
	opt = RMSprop(lr=G_LR)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model

# load images
def load_real_samples(filename):

	# load dataset
	file = open(filename, 'rb')
	# dump information to that file
	data = pickle.load(file)
	# close the file
	file.close()

	# select all of the examples for a given class
	X = data["X"]
	T = data["Y_s0"]

	# convert from ints to floats
	X = X.astype('float32')
	T = T.astype('float32')

	# scale to [-1,1]
	X = (X-HMAX)/HMAX
	T = (T-HMAX)/HMAX


	return X, T
 
# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	trajectories, initial_states = dataset 
	ix = randint(0, trajectories.shape[0], n_samples)
	# select images
	X = trajectories[ix]
	T = initial_states[ix]
	# generate class labels, -1 for 'real'
	y = -ones((n_samples, 1))
	return X, T, y, ix
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, initial_states, n_samples,  phase, ix = []):
	if phase == "D":
		t_input = initial_states[ix]
	elif phase == "G":	
		t_input = (rand(int(n_samples),N_SPECIES)-0.5)*2	
	else:
		print("ERROR!!")

	# generate points in the latent space
	z_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = z_input.reshape(n_samples, latent_dim)
	return z_input, t_input, ix

def generate_noise(latent_dim, n_samples):
	
	# generate points in the latent space
	z_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = z_input.reshape(n_samples, latent_dim)
	return z_input

def generate_cond_fake_samples(generator, latent_dim, initial_state, n_samples):
	# generate points in latent space
	z_input = generate_noise(latent_dim, n_samples)
	# predict outputs
	initial_state_rep = initial_state*np.ones((n_samples,N_SPECIES))
	
	#initial_state_rep = np.expand_dims(initial_state_rep, axis=1)
	
	X_gen = generator.predict([z_input, initial_state_rep])
	
	return X_gen 
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, initial_states, latent_dim, n_samples, phase, ret_ind = False, ix = []):
	# generate points in latent space
	z_input, t_input, idx = generate_latent_points(latent_dim, initial_states, n_samples, phase, ix = ix)
	# predict outputs
	X = generator.predict([z_input, t_input])
	# create class labels with 1.0 for 'fake'
	y = ones((n_samples, 1))
	if ret_ind:
		return X, t_input, idx 
	else:
		return X, t_input, y
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, dataset, n_samples=4):
	trajectories, initial_states = dataset
	# prepare fake examples
	X_fake, t_input, idx = generate_fake_samples(g_model, initial_states, latent_dim, n_samples, phase="D", ret_ind = True, ix = randint(0, trajectories.shape[0], n_samples))
	X_real = trajectories[idx]

	# plot images
	GR = 2
	if intermediate_plots:
		for i in range(GR * GR):
			# define subplot
			pyplot.subplot(GR, GR, 1 + i)
			complete_traj_fake = np.vstack((t_input[i], X_fake[i]))
			complete_traj_real = np.vstack((t_input[i], X_real[i]))
			
			xxx = np.linspace(0,TRAJ_LEN,TRAJ_LEN+1)
			pyplot.plot(xxx, complete_traj_fake[:, 0], label="S", color = "r")
			pyplot.plot(xxx, complete_traj_fake[:, 1], label="I", color = "b")

			pyplot.plot(xxx, complete_traj_real[:, 0], '--', label="S", color = "r")
			pyplot.plot(xxx, complete_traj_real[:, 1], '--', label="I", color = "b")
			
	# save plot to file
		filename1 = PLOTS_PATH+'/generated_plot_%04d.png' % (step+1)
		pyplot.legend()
		pyplot.savefig(filename1)
		pyplot.close()

		filename2 = MODELS_PATH+'/gen_model_%04d.h5' % (step+1)
		g_model.save(filename2)

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig(PLOTS_PATH+'/losses.png')
	pyplot.close()
 
# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64, n_critic=5, n_gen=1):
	trajectories, initial_states = dataset

	# calculate the number of batches per training epoch
	bat_per_epo = int(trajectories.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# lists for keeping track of loss
	c1_hist, c2_hist, g_hist = list(), list(), list()
	# manually enumerate epochs
	for i in range(n_steps):

		if i % bat_per_epo == 0:
			print("Epoch ", int(i / bat_per_epo)+1, " of ", n_epochs)

		# update the critic more than the generator
		c1_tmp, c2_tmp = list(), list()
		for _ in range(n_critic):
			# get randomly selected 'real' samples
			X_real, t_real, y_real, idx = generate_real_samples(dataset, half_batch)
			#print(X_real.shape, t_real.shape, y_real.shape)
			# update critic model weights
			t_real = tf.reshape(t_real, (half_batch,1, N_SPECIES))
			real_traj = tf.concat([t_real, X_real], axis = 1)
			c_loss1 = c_model.train_on_batch(real_traj, y_real)
			c1_tmp.append(c_loss1)
			# generate 'fake' examples
			X_fake, t_fake, y_fake = generate_fake_samples(g_model, initial_states, latent_dim, half_batch, phase="D", ix = idx)
			# update critic model weights
			t_fake = tf.reshape(t_fake, (half_batch,1, N_SPECIES))
			fake_traj = tf.concat([t_fake, X_fake], axis = 1)
			c_loss2 = c_model.train_on_batch(fake_traj, y_fake)
			c2_tmp.append(c_loss2)
		# store critic loss
		c1_hist.append(mean(c1_tmp))
		c2_hist.append(mean(c2_tmp))

		g_tmp = list()
		for _ in range(n_gen):
			# prepare points in latent space as input for the generator
			X_gan, t_gan, _ = generate_latent_points(latent_dim, initial_states, n_batch, phase="G")
			# create inverted labels for the fake samples
			y_gan = -ones((n_batch, 1))
			# update the generator via the critic's error
			g_loss = gan_model.train_on_batch([X_gan, t_gan], y_gan)
			g_tmp.append(g_loss)

		g_hist.append(mean(g_tmp))
		# summarize loss on this batch
		print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_hist[-1]))
		# evaluate the model performance every 'epoch'
		if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim, dataset)

	# line plots of loss
	plot_history(c1_hist, c2_hist, g_hist)

	return g_model
 

def load_test_data(filename):
		
	file = open(filename, 'rb')
	# dump information to that file
	traj_simulations = pickle.load(file)
	# close the file
	file.close()

	trajectories = traj_simulations["X"]
	initial_states = traj_simulations["Y_s0"]
	trajectories = trajectories.astype('float32')
	initial_states = initial_states.astype('float32')
	# scale to [-1,1]
	trajectories = (trajectories-HMAX)/HMAX
	initial_states = (initial_states-HMAX)/HMAX
	
	
	return trajectories, initial_states

def compute_trajectories(generator, valid_data):

	noise_timesteps = LATENT_DIM

	
	ssa_trajectories, initial_states = valid_data
	traj_per_state = ssa_trajectories.shape[1]

	print(f"\nComputing trajectories on {len(initial_states)} initial states")
	gen_trajectories = np.empty(shape=(len(initial_states), traj_per_state, TRAJ_LEN, N_SPECIES))

	for s, init_state in enumerate(initial_states):
		print("\tinit_state = ", init_state)
		
		gen_traj = generate_cond_fake_samples(generator, noise_timesteps, init_state, traj_per_state)

		gen_trajectories[s, :, :, :] = gen_traj			
		
		
	valid_dict = {"ssa": ssa_trajectories, "gen": gen_trajectories}
	file = open('FixedParam_wgan_conv1d_results.pickle', 'wb')
	# dump information to that file
	pickle.dump(valid_dict, file)
	# close the file
	file.close()
	return gen_trajectories

def plot_trajectories(ssa_trajectories, gen_trajectories):#valid_dict
	import seaborn as sns 
	
	n_init_states, traj_per_state, n_timesteps, n_species = ssa_trajectories.shape
	gen_trajectories_unscaled = np.round((gen_trajectories+1)*HMAX)
	ssa_trajectories_unscaled = np.round((ssa_trajectories+1)*HMAX)
	for init_state in range(n_init_states):

		fig, ax = pyplot.subplots(2,1,figsize=(12,6))

		ssa_fixed_init = ssa_trajectories_unscaled[init_state]
		gen_fixed_init = gen_trajectories_unscaled[init_state]

		for s in range(n_species):

			for traj_idx in range(5):#(traj_per_state):


				sns.lineplot(range(n_timesteps), ssa_fixed_init[traj_idx,:,s], ax=ax[s], color="blue")
				sns.lineplot(range(n_timesteps), gen_fixed_init[traj_idx,:,s], ax=ax[s], color="orange")
				ax[s].set_xlabel("timesteps")
				#ax[s].legend()

		fig.savefig(PLOTS_PATH+"/Trajectories"+str(init_state)+".png")
		pyplot.close()

def compute_distances(valid_data, gen_trajectories):#valid_dict,initial_states
	
	ssa_trajectories, initial_states = valid_data
	print(ssa_trajectories.shape, gen_trajectories.shape)
	n_init_states, traj_per_state, n_timesteps, n_species = ssa_trajectories.shape
	traj_per_state = ssa_trajectories.shape[1]

	gen_trajectories_unscaled = np.round((gen_trajectories+1)*HMAX)
	ssa_trajectories_unscaled = np.round((ssa_trajectories+1)*HMAX)
	
	print(f"\nComputing histograms on {len(initial_states)} initial states")
	bins = 100
	ssa_histograms_count = np.zeros(shape=(len(initial_states), TRAJ_LEN, N_SPECIES, bins))
	ssa_histograms_edg = np.zeros(shape=(len(initial_states), TRAJ_LEN, N_SPECIES, bins+1))
	gen_histograms_count = np.zeros(shape=(len(initial_states), TRAJ_LEN, N_SPECIES, bins))
	gen_histograms_edg = np.zeros(shape=(len(initial_states), TRAJ_LEN, N_SPECIES, bins+1))
	dist = np.zeros(shape=(len(initial_states), TRAJ_LEN, N_SPECIES))

	for s, init_state in enumerate(initial_states):
		print("\tinit_state = ", init_state)
		for t in range(TRAJ_LEN):
			for m in range(N_SPECIES): 
				hist = np.histogram(ssa_trajectories[s,:,t,m], bins=bins)
				ssa_histograms_count[s, t, m, :] = hist[0]
				ssa_histograms_edg[s, t, m, :] = hist[1]

				hist = np.histogram(gen_trajectories[s,:,t,m], bins=bins)
				gen_histograms_count[s, t, m, :] = hist[0]
				gen_histograms_edg[s, t, m, :] = hist[1]

				dist[s, t, m] = wasserstein_distance(ssa_histograms_count[s,t,m,:], 
				                                     gen_histograms_count[s,t,m,:]) 
	avg_dist = np.mean(dist, axis=0)
	fig = pyplot.figure()
	for spec in range(n_species):
		pyplot.plot(np.arange(TRAJ_LEN), avg_dist[:, spec])
	pyplot.xlabel("step")
	pyplot.ylabel("wass dist")
	pyplot.tight_layout()

	figname = "Results/LARGE_Traj_avg_wass_distance_{}epochs.png".format(N_EPOCHS)
	fig.savefig(figname)
	distances_dict = {"ssa_count":ssa_histograms_count, "ssa_edg":ssa_histograms_edg,
	              "gen_count":gen_histograms_count, "gen_edg":gen_histograms_edg,
	              "wass_dist":dist}
	file = open('LARGE_FixedParam_wgan_conv1d_distances.pickle', 'wb')
	# dump information to that file
	pickle.dump(distances_dict, file)
	# close the file
	file.close()

	colors = ['red', 'tan']
	leg = ['real', 'gen']
	bins = 50
	
	for s, init_state in enumerate(initial_states):
		fig, ax = pyplot.subplots(2,1,figsize=(12,6))
		for spec in range(N_SPECIES):
			XXX = np.vstack((ssa_trajectories_unscaled[s,:,-1,spec], gen_trajectories_unscaled[s,:,-1,spec])).T
			
			ax[spec].hist(XXX, bins = bins, stacked=False, density=False, color=colors, label=leg)
			ax[spec].legend()

		figname = PLOTS_PATH+"/hist_comparison_susc_last_timestep_"+str(init_state)+".png"
		fig.savefig(figname)
		pyplot.tight_layout()

		pyplot.close()

	return distances_dict


trainset_file = 'Datasets/eSIR_training_set_fixed_param_32steps.pickle'
valset_file = 'Datasets/eSIR_validation_set_fixed_param_32steps.pickle'

# create the critic
critic = define_critic()
# create the generator
generator = define_generator(LATENT_DIM)
# create the gan
gan_model = define_gan(generator, critic)
# load image data
dataset = load_real_samples(trainset_file)

# train model
DO_TRAIN = True
if DO_TRAIN:
	trained_generator = train(generator, critic, gan_model, dataset, LATENT_DIM, n_epochs = N_EPOCHS, n_batch = BATCH_SIZE, n_critic = N_CRITIC, n_gen = N_GEN)
	filename_gen = 'cwgan_fixed_param_traj_generator_model_{}_epochs.h5'.format(N_EPOCHS)
	trained_generator.save(filename_gen)
else:
	print("Loading trained generator...")
	trained_generator = load_model('cwgan_fixed_param_generator_model_final.h5')

valid_data = load_test_data(valset_file)
ssa_valid_trajectories, valid_init_states = valid_data

print(":::::::::GENERATE VALIDATION TRAJECTORIES")
gen_valid_trajectories = compute_trajectories(trained_generator, valid_data)


print(":::::::::COMPUTE AND PLOT HISTOGRAMS")
d = compute_distances(valid_data, gen_valid_trajectories)

print(":::::::::PLOT TRAJECTORIES: SSA vs GEN")
plot_trajectories(ssa_valid_trajectories, gen_valid_trajectories)
