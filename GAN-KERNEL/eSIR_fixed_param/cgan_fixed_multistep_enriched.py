from scipy.stats import wasserstein_distance

# example of a wgan for generating handwritten digits
from numpy import mean, ones, zeros
from numpy.random import randn, rand, randint
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import expand_dims, squeeze
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Concatenate, \
                                    Embedding, Flatten, Reshape, RepeatVector, Permute, \
                                    SeparableConv1D, Lambda, BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import pickle
from keras import backend
from matplotlib import pyplot

from scipy.stats import wasserstein_distance
from sklearn import preprocessing
import os

pyplot.rcParams.update({'font.size': 22})
pyplot.tight_layout()

ZZ = 100

Ngen = 5

HMAX = 50.0
N_EPOCHS = 50
Nsteps = 32

SCALING_FLAG = True

PLOT_PATH = "Plots/ENRICHED_{}-Step_FIXED_FF_CGAN_ngen_{}_nepochs_{}_ZZ ={}".format(Nsteps, Ngen, N_EPOCHS, ZZ)
try:
    os.mkdir(PLOT_PATH)
except OSError:
    print ("Creation of the directory %s failed" % PLOT_PATH)
else:
    print ("Successfully created the directory %s " % PLOT_PATH)


class CGAN_KERN(object):

	def __init__(self, noise_dim, state_dim):
		#self.timesteps = timesteps
		self.noise_dim = noise_dim
		self.state_dim = state_dim
		self.generator = None
		self.discriminator = None
		self.gan = None
		self.x0 = None
		self.x1 = None
		self.n_points_dataset = None
		self.normalization = True

	def set_T(self, T):
		self.T = T
	 
	# define the standalone critic model
	def define_discriminator(self):

		x1 = Input(shape=(self.state_dim,))
		x0 = Input(shape=(self.state_dim,)) 

		inputs = Concatenate(axis=1)([x1, x0])

		# define model
		x = Dense(64, activation='tanh')(inputs)
		
		x = Dense(128, activation='tanh')(x)

		#x = Dense(256, activation='tanh')(x)

		#x = Dense(256, activation='tanh')(x)

		x = Dense(128, activation='tanh')(x)

		x = Dense(64, activation='tanh')(x)

		output_l = Dense(1, activation='sigmoid')(x)
		
		model = Model(inputs=[x1, x0], outputs=output_l)

		# compile model
		opt = Adam(lr=0.0001, beta_1=0.5)

		model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
		
		self.discriminator = model
	 
	def define_generator(self):
		
		noise = Input(shape=(self.noise_dim,)) 
		x0 = Input(shape=(self.state_dim,))

		merge = Concatenate(axis=1)([noise, x0])

		print("GEN INPUT: ", merge)
		
		x = Dense(32, activation='tanh')(merge)
		
		#x = Dense(128, activation='tanh')(x)

		x = Dense(32, activation='tanh')(x)

		## output
		if self.normalization:
			output_x1 = Dense(self.state_dim, activation='tanh')(x)
		else:
			output_x1 = Dense(self.state_dim)(x)

		print("GEN OUTPUT: ", output_x1)

		model = Model(inputs=[noise, x0], outputs=output_x1)

		self.generator = model


	 
	# define the combined generator and critic model, for updating the generator
	def define_gan(self):

		# make weights in the critic not trainable
		self.discriminator.trainable = False
		noise, x0 = self.generator.input
		gen_x1 = self.generator.output

		gan_output = self.discriminator([gen_x1, x0])

		model = Model(inputs=[noise, x0], outputs=gan_output)

		# compile model
		opt = Adam(lr=0.001, beta_1=0.5)

		model.compile(loss='binary_crossentropy', optimizer=opt)
		
		self.gan = model
	

	# load images
	def load_real_samples(self, dataset_filename):

		# load dataset
		file = open(dataset_filename, 'rb')
		# dump information to that file
		data = pickle.load(file)
		# close the file
		file.close()

		# select all of the examples for a given class
		x1 = data["X"]
		x0 = data["Y_s0"]

		x1 = x1.astype('float32')
		x0 = x0.astype('float32')

		if self.normalization:
			self.x1 = (x1-HMAX)/HMAX
			self.x0 = (x0-HMAX)/HMAX
			self.n_points_dataset = x1.shape[0]			
		else:
			self.x1 = x1
			self.x0 = x0
			self.n_points_dataset = states.shape[0]
	 
	# select real samples
	def generate_real_samples(self, n_samples):
		# choose random instances
		ix = randint(0, self.n_points_dataset, n_samples)
		# select images
		x0 = self.x0[ix]
		x1 = self.x1[ix]
		# generate class labels, -1 for 'real'
		lab = ones((n_samples, 1))
		return x1, x0, lab, ix
	 
	# generate points in latent space as input for the generator
	def generate_latent_points(self, n_samples, phase, ix = []):
		
		if phase == "D":
			x0_input = self.x0[ix]
		elif phase == "G":	
			if self.normalization:
				x0_input = (rand(int(n_samples),self.state_dim)-0.5)*2
			else:
				x0_input = 2*HMAX*rand(int(n_samples),self.state_dim)
		else:
			print("ERROR!!")
		
		# generate points in the latent space
		z_input = randn(self.noise_dim*n_samples)

		# reshape into a batch of inputs for the network
		z_input = z_input.reshape(n_samples, self.noise_dim)
		
		return z_input, x0_input, ix

	def generate_noise(self, n_samples):
		
		# generate points in the latent space
		z_input = randn(self.noise_dim*n_samples)
		# reshape into a batch of inputs for the network
		z_input = z_input.reshape(n_samples, self.noise_dim)
		return z_input

	def generate_cond_fake_samples(self, selected_x0, n_samples):
		# generate points in latent space
		z_input = self.generate_noise(n_samples)
		# predict outputs		
		x0_rep = selected_x0*np.ones((n_samples,self.state_dim))
		gen_x1 = self.generator.predict([z_input, x0_rep])
		
		return gen_x1 


	def generate_cond_fake_trajs(self, selected_x0, n_samples):
		# generate points in latent space
		gen_trajs = np.zeros((n_samples, self.T, self.state_dim))
		init_states = selected_x0*np.ones((n_samples,self.state_dim))
		for t in range(self.T):
			z_input = self.generate_noise(n_samples)
			# predict outputs		
			gen_x1 = self.generator.predict([z_input, init_states])
			gen_trajs[:,t,:] = gen_x1
			init_states = gen_x1

		return gen_trajs

	def generate_cond_fake_unscaled_trajs(self, selected_x0, n_samples):
		# generate points in latent space
		unscaled_gen_trajs = np.zeros((n_samples, self.T, self.state_dim))
		init_states = selected_x0*np.ones((n_samples,self.state_dim))
		for t in range(self.T):
			z_input = self.generate_noise(n_samples)
			# predict outputs		
			gen_x1 = self.generator.predict([z_input, init_states])
			unscaled_gen_x1 = np.round((gen_x1+1)*HMAX)
			unscaled_gen_trajs[:,t,:] = unscaled_gen_x1
			init_states = (unscaled_gen_x1-HMAX)/HMAX

		return unscaled_gen_trajs 

	# use the generator to generate n fake examples, with class labels
	def generate_fake_samples(self, n_samples, phase, ret_ind = False, ix = []):
		# generate points in latent space
		z_input, x0_input, idx = self.generate_latent_points(n_samples, phase, ix = ix)
		# predict outputs
		gen_x1 = self.generator.predict([z_input, x0_input])
		# create class labels with 1.0 for 'fake'
		lab = zeros((n_samples, 1))
		return gen_x1, x0_input, lab
	 
	# generate samples and save as a plot and save the model
	def summarize_performance(self, step, n_samples=4):
		# prepare fake examples
		hist_size = 10
		#ix = randint(0, self.n_points_dataset, n_samples)
		ID = randint(0, 10, n_samples)
		#x0_inputs = self.x0[ix]
		x0_inputs = self.x0[ID*hist_size]
		gen_x1 = np.empty((n_samples, hist_size, self.state_dim))
		for j in range(n_samples):
			gen_x1[j] = self.generate_cond_fake_samples(x0_inputs[j], hist_size)
		
		colors = ['green', 'red']
		leg = ['real', 'gen']
		
		bins = 25
		# plot images
		GR = 2
		count = 0
		for i in range(GR * GR):
			# define subplot
			#real_x1 = self.x1[ix[i]]*ones((hist_size, self.state_dim))
			real_x1 = self.x1[ID[i]*hist_size:(ID[i]+1)*hist_size]
			
			for s in range(self.state_dim):
				XX = np.vstack((real_x1[:,s],gen_x1[i,:,s])).T	
		
				pyplot.subplot(GR*GR, self.state_dim, 1 + count)
				pyplot.hist(XX, bins = bins, stacked=False, density=True, color=colors, label=leg, alpha = 0.5)
				count += 1

		pyplot.tight_layout()
		filename1a = PLOT_PATH+'/generated_states_%04d.png' % (step+1)
		pyplot.legend()
		pyplot.savefig(filename1a)
		pyplot.close()
		
		
	# create a line plot of loss for the gan and save to file
	def plot_history(self,d1_hist, d2_hist, g_hist, d_acc_hist):
		# plot history
		pyplot.plot(d1_hist, label='discr_real')
		pyplot.plot(d2_hist, label='discr_fake')
		pyplot.plot(g_hist, label='gen')
		pyplot.title("Losses")
		pyplot.legend()
		pyplot.savefig(PLOT_PATH+'/losses.png')
		pyplot.close()

		pyplot.plot(d_acc_hist)
		pyplot.title("Discr. Acc.")
		pyplot.legend()
		pyplot.savefig(PLOT_PATH+'/accuracy.png')
		pyplot.close()
	 
	# train the generator and critic
	def train(self, n_epochs=10, n_batch=64, n_gen=5):
		
		# calculate the number of batches per training epoch
		bat_per_epo = int(self.n_points_dataset / n_batch)
		# calculate the number of training iterations
		n_steps = bat_per_epo * n_epochs
		# calculate the size of half a batch of samples
		half_batch = int(n_batch / 2)
		# lists for keeping track of loss
		d1_hist, d2_hist, g_hist, d_acc_hist = list(), list(), list(), list()
		# manually enumerate epochs
		for i in range(n_steps):
			# update the critic more than the generator
			
			# get randomly selected 'real' samples
			x1_real, x0_real, l_real, idx = self.generate_real_samples(half_batch)
			#print(X_real.shape, t_real.shape, y_real.shape)
			# update critic model weights
			d_loss1, d_acc1 = self.discriminator.train_on_batch([x1_real, x0_real], l_real)
			# generate 'fake' examples
			x1_fake, x0_fake, l_fake = self.generate_fake_samples(half_batch, phase="D", ix = idx)
			# update critic model weights
			
			d_loss2, d_acc2 = self.discriminator.train_on_batch([x1_fake, x0_fake], l_fake)
			
			d_acc = (d_acc1+d_acc2)/2

			# store critic loss
			d1_hist.append(d_loss1)
			d2_hist.append(d_loss2)
			d_acc_hist.append(d_acc)

			g_tmp = list()
			for _ in range(n_gen):
				# prepare points in latent space as input for the generator
				x1_gan, x0_gan, _ = self.generate_latent_points(n_batch, phase="G")
				# create inverted labels for the fake samples
				l_gan = ones((n_batch, 1))
				# update the generator via the critic's error
				g_loss = self.gan.train_on_batch([x1_gan, x0_gan], l_gan)
				g_tmp.append(g_loss)
			g_hist.append(mean(g_tmp))
			# summarize loss on this batch
			print('>%d, d1=%.3f, d2=%.3f, d_acc=%.3f, g=%.3f' % (i+1, d1_hist[-1], d2_hist[-1], d_acc_hist[-1], g_hist[-1]))
			# evaluate the model performance every 'epoch'
			if (i+1) % bat_per_epo == 0:
				self.summarize_performance(i)
		# line plots of loss
		self.plot_history(d1_hist, d2_hist, g_hist, d_acc_hist)

		filename2 = 'Models/enriched_trained_gen_model_final.h5'
		self.generator.save(filename2)


	def load_test_data(self, val_data_filename):
			
		# load dataset
		file = open(val_data_filename, 'rb')
		# dump information to that file
		data = pickle.load(file)
		# close the file
		file.close()

		

		x1_val = data["X"]
		x1_val = x1_val[:, :,:self.T,:]
		x0_val = data["Y_s0"]
		x1_val = x1_val.astype('float32')
		x0_val = x0_val.astype('float32')

		print(data["Y_par"][0])
		# scale from [0,255] to [-1,1]
		if self.normalization:
			self.x1_val_unscaled = x1_val
			self.x1_val = (x1_val-HMAX)/HMAX
			self.x0_val_unscaled = x0_val
			self.x0_val = (x0_val-HMAX)/HMAX
			self.n_val_points = x0_val.shape[0]
			self.traj_per_state = x1_val.shape[1]			
		else:
			self.x1_val = x1_val
			self.x0_val = x0_val
			self.n_val_points = x0_val.shape[0]
			self.traj_per_state = x1_val.shape[1]
		

	def compute_test_results(self, scaled_flag):

		gen_trajs = np.empty(shape=(self.n_val_points, self.traj_per_state, self.T, self.state_dim))

		for i, init_state in enumerate(self.x0_val):
			print("\tinit_state = ", init_state)
			
			if scaled_flag:
				gen_traj = self.generate_cond_fake_trajs(init_state, self.traj_per_state)
			else:
				gen_traj = self.generate_cond_fake_unscaled_trajs(init_state, self.traj_per_state)
			
			gen_trajs[i] = gen_traj			
			
		return gen_trajs

	def plot_test_hist(self, gen_trajs, scaled_flag):

		if scaled_flag:
			gen_traj_unscaled = np.round((gen_trajs+1)*HMAX)
		else:
			gen_traj_unscaled = gen_trajs

		colors = ['green', 'red']
		leg = ['real', 'gen']
		bins = 25

		dist = np.zeros((self.n_val_points, self.T, self.state_dim))
		for i, init_state in enumerate(self.x0_val):
			for t in range(self.T):
				
				fig, ax = pyplot.subplots(self.state_dim,1,figsize=(12,6))
				for spec in range(self.state_dim):
					XXX = np.vstack((self.x1_val_unscaled[i,:, t, spec], gen_traj_unscaled[i,:, t, spec])).T
					
					ax[spec].hist(XXX, bins = bins, stacked=False, density=True, color=colors, label=leg)
					ax[spec].legend()
					dist[i, t, spec] = wasserstein_distance(self.x1_val_unscaled[i,:, t, spec], gen_traj_unscaled[i,:, t, spec])
				figname = PLOT_PATH+"/hist_comparison_"+str(init_state)+"_t_"+str(t)+".png"
				fig.savefig(figname)
				pyplot.tight_layout()
				pyplot.close()
		fname = 'Results/ENRICHED_Scaled={}_20ep_wass_dist_{}steps.npy'.format(scaled_flag,Nsteps)
		with open(fname, 'wb') as f:
			np.save(f, dist)

		avg_dist = np.mean(dist, axis=0)
		fig = pyplot.figure()
		for spec in range(self.state_dim):
			pyplot.plot(np.arange(self.T), avg_dist[:, spec])
		pyplot.xlabel("step")
		pyplot.ylabel("wass dist")
		pyplot.tight_layout()
		figname = "Results/ENRICHED_Scaled={}_20ep_avg_wass_distance_{}steps.png".format(scaled_flag,Nsteps)
		fig.savefig(figname)

DO_TRAIN = True
# size of the latent space
state_dim = 2
latent_dim = state_dim*ZZ

cgan = CGAN_KERN(latent_dim, state_dim)

cgan.set_T(Nsteps)
# create the critic
cgan.define_discriminator()
# create the generator
cgan.define_generator()
# create the gan
cgan.define_gan()
# load image data
cgan.load_real_samples('Datasets/eSIR_enriched_training_set_fixed_param_32steps.pickle')

if DO_TRAIN:
	# train model
	cgan.train(n_epochs = N_EPOCHS, n_batch = 1024, n_gen=Ngen)
else:
	cgan.generator = load_model('trained_gen_model_final_enriched.h5')


cgan.load_test_data('Datasets/eSIR_validation_set_fixed_param_32steps.pickle')


print(":::::::::Scaling flag: ", SCALING_FLAG)

gen_valid_x1s = cgan.compute_test_results(SCALING_FLAG)

#print(":::::::::PLOT TEST HISTOGRAMS")
cgan.plot_test_hist(gen_valid_x1s, SCALING_FLAG)