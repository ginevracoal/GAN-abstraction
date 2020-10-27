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
import os

pyplot.rcParams.update({'font.size': 22})
pyplot.tight_layout()

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


class WGAN_MA(object):

	def __init__(self, model_name, noise_dim, state_dim, param_dim, traj_len, labels, colors):
		self.noise_dim = noise_dim
		self.state_dim = state_dim
		self.param_dim = param_dim
		self.traj_len = traj_len
		self.labels = labels
		self.colors = colors
		self.MODEL_NAME = model_name
		self.generator = None
		self.critic = None
		self.gan = None
		self.T_train = None
		self.X_train = None
		self.T_train = None
		self.X_train = None
		self.n_points_dataset = None
		self.n_epochs = 100
		self.batch_size = 64
		self.n_critic = 1
		self.n_gen = 1
		self.HMAX = 0
		self.clip_const = 0.01
		self.c_lr = 0.00005
		self.g_lr = 0.00005
		self.q = int(traj_len/(2**4))
		self.n_ch = 512
		self.embedding = "REPEAT" #"REPEAT" or "DENSE"
		self.intermediate_plots_flag = True



	def generate_directories(self, ID = str(randint(0,100000))):
		self.ID = ID
		self.PLOTS_PATH = self.MODEL_NAME + "/Plots/ID_" +self.ID
		self.MODELS_PATH = self.MODEL_NAME + "/Models/ID_" +self.ID
		self.RESULTS_PATH = self.MODEL_NAME + "/Results/ID_" +self.ID
		os.makedirs(self.PLOTS_PATH, exist_ok=True)
		os.makedirs(self.MODELS_PATH, exist_ok=True)
		os.makedirs(self.RESULTS_PATH, exist_ok=True)


	def print_log(self):
		f = open(self.RESULTS_PATH+"/log.txt", "w")
		f.write(self.MODEL_NAME+ " MODEL----------"+str(self.labels)+str(self.colors))
		f.write("n_epochs={}, batch_size={}, n_critic={}, n_gen={}, noise_dim={}, traj_len={}, state_dim={}".format(self.n_epochs,self.batch_size,self.n_gen, self.n_critic,self.noise_dim,self.traj_len,self.state_dim))
		f.write("Q={}, N_CH={}, LR=({},{}), embedding={}".format(self.q,self.n_ch, self.c_lr,self.g_lr,self.embedding))
		f.write(self.arch_critic+self.arch_gen)
		f.write('../Dataset_Generation/data/'+self.MODEL_NAME+'/'+self.MODEL_NAME+self.trainset_filename)
		f.write('../Dataset_Generation/data/'+self.MODEL_NAME+'/'+self.MODEL_NAME+self.validset_filename)
		f.close()

	def wasserstein_loss(self, y_true, y_pred):
		return backend.mean(y_true * y_pred)


	def define_critic(self):
		
		traj = Input(shape=(self.traj_len+1, self.state_dim)) 
		par = Input(shape=(self.param_dim,))
		
		if self.embedding == "DENSE":
			p = Dense(self.traj_len+1)(par)
			p = Reshape((self.traj_len+1,1))(p)
		else:
			p = RepeatVector(self.traj_len+1)(par)

		inputs = Concatenate(axis=2)([traj, p])

		HC = [64, 64]
		KC = [4, 4]
		SC = [2,2]
		# weight constraint
		const = ClipConstraint(self.clip_const)
		# downsample 
		x = Conv1D(HC[0], KC[0], strides=SC[0], padding='same', kernel_constraint=const)(inputs)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)
		# downsample 
		x = Conv1D(HC[1], KC[1], strides=SC[1], padding='same', kernel_constraint=const)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)

		# scoring, linear activation
		x = Flatten()(x)
		outputs = Dense(1)(x)

		model = Model(inputs=[traj, par], outputs=outputs)

		# compile model
		opt = RMSprop(lr=self.c_lr)
		model.compile(loss=self.wasserstein_loss, optimizer=opt)

		self.arch_critic = 'C_ARCH: H={}, K={}, S={}+LeakyRelu02'.format(HC, KC, SC)
		print(self.arch_critic)

		self.critic = model


	def define_generator(self):

		noise = Input(shape=(self.noise_dim)) 
		n_nodes_n = self.n_ch * self.q
		nv = Dense(n_nodes_n)(noise)
		nv = Reshape((self.q, self.n_ch))(nv)

		init_states = Input(shape=(self.state_dim))
		
		par = Input(shape=(self.param_dim,))

		if self.embedding== "DENSE":
			n_nodes_i = self.q * 1
			iv = Dense(n_nodes_i)(init_states)
			iv = Reshape((self.q, 1))(iv)
			pv = Dense(n_nodes_i)(par)
			pv = Reshape((self.q, 1))(pv)
		else:
			iv = RepeatVector(self.q)(init_states)
			pv = RepeatVector(self.q)(par)

		merge = Concatenate(axis=2)([iv, pv, nv])

		HG = [128, 256, 256, 128]
		KG = [4, 4, 4, 4, 4]
		SG = [2, 2, 2, 2]
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

		x = Conv1DTranspose(HG[3], KG[3], padding = "same", strides = SG[3])(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)
		
		# output
		outputs = Conv1D(self.state_dim, KG[-1], activation='tanh', padding='same')(x)
		print("GEN OUTPUT: ", outputs)

		model = Model(inputs=[noise,init_states,par], outputs=outputs)

		self.arch_gen = 'G_ARCH: H={}, K={}, S={}+LeakyRelu02'.format(HG, KG, SG)
		print(self.arch_gen)

		self.generator = model

	 
	# define the combined generator and critic model, for updating the generator
	def define_gan(self):
		# make weights in the critic not trainable
		self.critic.trainable = False
		noise, init_states, par = self.generator.input
		gen_traj = self.generator.output

		in_st = Reshape((1,self.state_dim))(init_states)
		merged_traj = Concatenate(axis=1)([in_st, gen_traj])

		gan_output = self.critic([merged_traj,par])

		model = Model(inputs=[noise, init_states, par], outputs=gan_output)

		# compile model
		opt = RMSprop(lr=self.g_lr)
		model.compile(loss=self.wasserstein_loss, optimizer=opt)
		self.gan = model


	def set_dataset_location(self, trainset_filename, valset_filename):
		self.trainset_filename = trainset_filename
		self.validset_filename = valset_filename



	def load_real_data(self):

		# load dataset
		file = open('../Dataset_Generation/data/'+self.MODEL_NAME+'/'+self.MODEL_NAME+self.trainset_filename, 'rb')
		# dump information to that file
		data = pickle.load(file)
		# close the file
		file.close()

		# select all of the examples for a given class
		X = data["X"][:,:self.traj_len,:]
		T = data["Y_s0"]
		par = data["Y_par"][:,:1]

		# convert from ints to floats
		X = X.astype('float32')
		T = T.astype('float32')
		par = par.astype('float32')

		self.HMAX = np.max(np.max(X, axis = 0),axis=0)/2
		self.PMAX = np.max(par, axis = 0)/2

		self.n_points_dataset = X.shape[0]
		# scale to [-1,1]
		self.X_train = (X-self.HMAX)/self.HMAX
		self.T_train = (T-self.HMAX)/self.HMAX
		self.P_train = (par-self.PMAX)/self.PMAX

	def load_test_data(self):
		
		file = open('../Dataset_Generation/data/'+self.MODEL_NAME+'/'+self.MODEL_NAME+self.validset_filename, 'rb')
		# dump information to that file
		val_data = pickle.load(file)
		# close the file
		file.close()

		X_val = val_data["X"][:,:,:self.traj_len,:]
		T_val = val_data["Y_s0"]
		par_val = val_data["Y_par"][:,:1]

		X_val = X_val.astype('float32')
		T_val = T_val.astype('float32')
		par_val = par_val.astype('float32')
		# scale to [-1,1]
		self.X_val = (X_val-self.HMAX)/self.HMAX
		self.T_val = (T_val-self.HMAX)/self.HMAX
		self.P_val = (par_val-self.PMAX)/self.PMAX

	# select real samples
	def generate_real_samples(self, n_samples):
		
		ix = randint(0, self.X_train.shape[0], n_samples)
		# select datas
		Xb = self.X_train[ix]
		Tb = self.T_train[ix]
		Pb = self.P_train[ix]
		# generate class labels, -1 for 'real'
		yb = -ones((n_samples, 1))
		return Xb, Tb, Pb, yb, ix

	# generate points in latent space as input for the generator
	def generate_latent_points(self, n_samples,  phase, ix = []):
		if phase == "D":
			t_input = self.T_train[ix]
			p_input = self.P_train[ix]
		elif phase == "G":	
			t_input = (rand(int(n_samples),self.state_dim)-0.5)*2
			p_input = (rand(int(n_samples),self.param_dim)-0.5)*2	
		else:
			print("ERROR!!")

		# generate points in the latent space
		z_input = randn(self.noise_dim * n_samples)
		# reshape into a batch of inputs for the network
		z_input = z_input.reshape(n_samples, self.noise_dim)
		return z_input, t_input, p_input, ix

	def generate_noise(self, n_samples):
		# generate points in the latent space
		z_input = randn(self.noise_dim * n_samples)
		# reshape into a batch of inputs for the network
		z_input = z_input.reshape(n_samples, self.noise_dim)
		return z_input

	def generate_cond_fake_samples(self, selected_initial_state, selected_par, n_samples):
		# generate points in latent space
		z_input = self.generate_noise(n_samples)
		# predict outputs
		initial_state_rep = selected_initial_state*np.ones((n_samples,self.state_dim))
		par_rep = selected_par*np.ones((n_samples,self.param_dim))

		X_gen = self.generator.predict([z_input, initial_state_rep, par_rep])
		
		return X_gen

	# use the generator to generate n fake examples, with class labels
	def generate_fake_samples(self, n_samples, phase, ret_ind = False, ix = []):
		# generate points in latent space
		z_input, t_input, p_input, idx = self.generate_latent_points(n_samples, phase, ix = ix)
		# predict outputs
		X = self.generator.predict([z_input, t_input, p_input])
		# create class labels with 1.0 for 'fake'
		y = ones((n_samples, 1))
		if ret_ind:
			return X, t_input, p_input, idx 
		else:
			return X, t_input, p_input, y

	# generate samples and save as a plot and save the model
	def summarize_performance(self, step, n_samples=4):

		# prepare fake examples
		X_fake, t_input, p_input, idx = self.generate_fake_samples(n_samples, phase="D", ret_ind = True, ix = randint(0, self.X_train.shape[0], n_samples))
		X_real = self.X_train[idx]

		# plot images
		GR = 2
		if self.intermediate_plots_flag:
			for i in range(GR * GR):
				# define subplot
				pyplot.subplot(GR, GR, 1 + i)
				complete_traj_fake = np.vstack((t_input[i], X_fake[i]))
				complete_traj_real = np.vstack((t_input[i], X_real[i]))
				
				xxx = np.linspace(0,self.traj_len,self.traj_len+1)
				for spec in range(self.state_dim):
					pyplot.plot(xxx, complete_traj_fake[:, spec], label=self.labels[spec], color = self.colors[spec])
					pyplot.plot(xxx, complete_traj_real[:, spec], '--', label=self.labels[spec], color = self.colors[spec])
					
			# save plot to file
			filename1 = self.PLOTS_PATH+'/generated_plot_%04d.png' % (step+1)
			pyplot.legend()
			pyplot.savefig(filename1)
			pyplot.close()
			
			filename2 = self.MODELS_PATH+'/gen_model_%04d.h5' % (step+1)
			self.generator.save(filename2)

	# create a line plot of loss for the gan and save to file
	def plot_history(self,d1_hist, d2_hist, g_hist):
		# plot history
		pyplot.plot(d1_hist, label='crit_real')
		pyplot.plot(d2_hist, label='crit_fake')
		pyplot.plot(g_hist, label='gen')
		pyplot.legend()
		pyplot.savefig(self.PLOTS_PATH+'/losses.png')
		pyplot.close()
	

	def set_training_options(self, n_epochs, batch_size, n_critic, n_gen):
		self.n_epochs=n_epochs
		self.batch_size=batch_size
		self.n_critic=n_critic
		self.n_gen=n_gen


	def define_wgan_model(self):
		self.define_critic()
		self.define_generator()
		self.define_gan()


	# train the generator and critic
	def train(self):
		# calculate the number of batches per training epoch
		bat_per_epo = int(self.n_points_dataset / self.batch_size)
		# calculate the number of training iterations
		n_steps = bat_per_epo * self.n_epochs
		# calculate the size of half a batch of samples
		half_batch = int(self.batch_size / 2)
		# lists for keeping track of loss
		c1_hist, c2_hist, g_hist = list(), list(), list()
		# manually enumerate epochs
		for i in range(n_steps):

			if i % bat_per_epo == 0:
				print("Epoch ", int(i / bat_per_epo)+1, " of ", self.n_epochs)

			# update the critic more than the generator
			c1_tmp, c2_tmp = list(), list()
			for _ in range(self.n_critic):
				# get randomly selected 'real' samples
				X_real, t_real, p_real, y_real, idx = self.generate_real_samples(half_batch)
				# update critic model weights
				t_real = tf.reshape(t_real, (half_batch,1, self.state_dim))
				real_traj = tf.concat([t_real, X_real], axis = 1)
				c_loss1 = self.critic.train_on_batch([real_traj, p_real], y_real)
				c1_tmp.append(c_loss1)
				# generate 'fake' examples
				X_fake, t_fake, p_fake, y_fake = self.generate_fake_samples(half_batch, phase="D", ix = idx)
				# update critic model weights
				t_fake = tf.reshape(t_fake, (half_batch,1, self.state_dim))
				fake_traj = tf.concat([t_fake, X_fake], axis = 1)
				c_loss2 = self.critic.train_on_batch([fake_traj, p_fake], y_fake)
				c2_tmp.append(c_loss2)
			# store critic loss
			c1_hist.append(mean(c1_tmp))
			c2_hist.append(mean(c2_tmp))

			g_tmp = list()
			for _ in range(self.n_gen):
				# prepare points in latent space as input for the generator
				Z_gan, t_gan, p_gan, _ = self.generate_latent_points(self.batch_size, phase="G")
				# create inverted labels for the fake samples
				y_gan = -ones((self.batch_size, 1))
				# update the generator via the critic's error
				g_loss = self.gan.train_on_batch([Z_gan, t_gan, p_gan], y_gan)
				g_tmp.append(g_loss)

			g_hist.append(mean(g_tmp))
			# summarize loss on this batch
			print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_hist[-1]))
			# evaluate the model performance every 'epoch'
			if (i+1) % bat_per_epo == 0:
				self.summarize_performance(i)

		# line plots of loss
		self.plot_history(c1_hist, c2_hist, g_hist)
		filename = self.MODELS_PATH+'/final_generator_{}_epochs.h5'.format(self.n_epochs)
		self.generator.save(filename)

	def generate_validation_trajectories(self):
		
		traj_per_state = self.X_val.shape[1]

		print(f"\nComputing trajectories on {len(self.T_val)} initial settings")
		gen_trajectories = np.empty(shape=(len(self.T_val), traj_per_state, self.traj_len, self.state_dim))

		for s in range(len(self.T_val)):
			print("\tinit_state = ", self.T_val[s]," param = ", self.P_val[s])
			
			gen_traj = self.generate_cond_fake_samples(self.T_val[s], self.P_val[s], traj_per_state)

			gen_trajectories[s, :, :, :] = gen_traj			
			
			
		valid_dict = {"ssa": self.X_val, "gen": gen_trajectories}
		file = open(self.RESULTS_PATH+'/validation_trajectories_GAN_vs_SSA.pickle', 'wb')
		# dump information to that file
		pickle.dump(valid_dict, file)
		# close the file
		file.close()
		self.gen_trajectories = gen_trajectories


	def eval_times(self):

		traj_per_state = 20000	
		
		for s in range(len(self.T_val)):
			print("\tinit_state = ", self.T_val[s]," param = ", self.P_val[s])
			start_time = time.time()
			gen_traj = self.generate_cond_fake_samples(self.T_val[s], self.P_val[s], traj_per_state)
			print(time.time()-start_time)
	

	def plot_validation_trajectories(self):
		import seaborn as sns 
		
		n_val_settings, traj_per_state, n_timesteps, n_species = self.X_val.shape
		gen_trajectories_unscaled = np.round((self.gen_trajectories+1)*self.HMAX)
		ssa_trajectories_unscaled = np.round((self.X_val+1)*self.HMAX)
		
		for ii in range(n_val_settings):

			fig, ax = pyplot.subplots(self.state_dim,1,figsize=(12,self.state_dim*3))

			ssa_fixed_init = ssa_trajectories_unscaled[ii]
			gen_fixed_init = gen_trajectories_unscaled[ii]
			for s in range(self.state_dim):

				for traj_idx in range(5):
					sns.lineplot(range(n_timesteps), ssa_fixed_init[traj_idx,:,s], ax=ax[s], color="blue")
					sns.lineplot(range(n_timesteps), gen_fixed_init[traj_idx,:,s], ax=ax[s], color="orange")
					ax[s].set_xlabel("timesteps")
					ax[s].set_ylabel(self.labels[s])


			fig.savefig(self.PLOTS_PATH+"/"+self.MODEL_NAME+"_Trajectories"+str(ii)+".png")
			pyplot.close()


	def plot_last_step_histogram(self):
		
		n_val_settings, traj_per_state, n_timesteps, n_species = self.X_val.shape
		
		gen_trajectories_unscaled = np.round((self.gen_trajectories+1)*self.HMAX)
		ssa_trajectories_unscaled = np.round((self.X_val+1)*self.HMAX)
		
		colors = ['blue', 'orange']
		leg = ['real', 'gen']
		bins = 50
		
		for s in range(n_val_settings):
			fig, ax = pyplot.subplots(self.state_dim,1, figsize = (12,self.state_dim*3))
			for spec in range(self.state_dim):
				XXX = np.vstack((ssa_trajectories_unscaled[s,:,-1,spec], gen_trajectories_unscaled[s,:,-1,spec])).T
				
				ax[spec].hist(XXX, bins = bins, stacked=False, density=False, color=colors, label=leg)
				ax[spec].legend()
				ax[spec].set_ylabel(self.labels[spec])

			figname = self.PLOTS_PATH+"/"+self.MODEL_NAME+"_hist_comparison_last_timestep_"+str(s)+".png"
			fig.savefig(figname)

			pyplot.close()


	def compute_distances(self):
		
		n_val_settings, traj_per_state, n_timesteps, n_species = self.X_val.shape

		gen_trajectories_unscaled = np.round((self.gen_trajectories+1)*self.HMAX)
		ssa_trajectories_unscaled = np.round((self.X_val+1)*self.HMAX)
		
		print(f"\nComputing histograms on {n_val_settings} val settings")
		
		dist = np.zeros(shape=(n_val_settings, self.traj_len, self.state_dim))
		XMAX = np.max(self.HMAX*2)
		for s in range(n_val_settings):
			print("\tinit_state = ", self.T_val[s]," param = ", self.P_val[s])
			for t in range(self.traj_len):
				for m in range(self.state_dim):
					A = ssa_trajectories_unscaled[s,:,t,m]
					B = gen_trajectories_unscaled[s,:,t,m]

					histA, edgA = np.histogram(A, bins=np.arange(XMAX))

					histB, edGB = np.histogram(B, bins=np.arange(XMAX))
					
					dist[s, t, m] = wasserstein_distance(histA, histB)
					

		avg_dist = np.mean(dist, axis=0)
		markers = ['--','-.',':']
		fig = pyplot.figure()
		for spec in range(self.state_dim):
			pyplot.plot(np.arange(self.traj_len), avg_dist[:, spec], markers[spec], label=self.labels[spec])
		pyplot.legend()
		pyplot.xlabel("time")
		pyplot.ylabel("wass dist")
		pyplot.tight_layout()

		figname = self.RESULTS_PATH+"/Traj_avg_wass_distance_{}epochs_{}steps.png".format(self.n_epochs, self.traj_len)
		fig.savefig(figname)
		distances_dict = {"gen_hist":histB, "ssa_hist":histA, "wass_dist":dist}
		file = open(self.RESULTS_PATH+'/wgan_conv1d_distances.pickle', 'wb')
		# dump information to that file
		pickle.dump(distances_dict, file)
		# close the file
		file.close()


