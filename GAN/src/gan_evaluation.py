import os
import argparse
from directories import *
from gan_abstraction import GAN_abstraction
from utils import load_from_pickle, generate_noise, save_to_pickle, load_and_rescale
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from tqdm import tqdm
import os
from keras import backend as K
import itertools


class GAN_evaluator(GAN_abstraction):

	def __init__(self, model, n_traj, timesteps, noise_timesteps, n_epochs, gen_epochs, embed, 
		         fixed_params, lr):
		super(GAN_evaluator, self).__init__(model=model, timesteps=timesteps, n_epochs=n_epochs,
											noise_timesteps=noise_timesteps, embed=embed, 
											fixed_params=fixed_params, lr=lr, gen_epochs=gen_epochs)
		self.n_traj = n_traj
		self.path = "evaluations/"+self.model+"_t="+str(self.timesteps)+"_tNoise="+\
		            str(self.noise_timesteps)+"_ep="+str(n_epochs)+"_epG="+str(gen_epochs)+\
		            "_lr="+str(self.lr)+"/"

	def load_gan(self, rel_path):
		return super(GAN_evaluator, self).load(rel_path=rel_path, n_epochs=self.n_epochs, 
			                                   gen_epochs=self.gen_epochs)

	def load_test_data(self, model, path="../../SSA/data/test/"):
		
		if model == "SIR":
			filename = "SIR_test_set"
		elif model == "eSIR":
			filename = "eSIR_test_set"
		elif model == "Repress":
			filename = "Repressilator_test_set"
		elif model == "Toggle":
			filename = "ToggleSwitch_test_set"

		if self.fixed_params==1:
			filename = filename+"_oneparam"

		traj_simulations = load_from_pickle(path=path+filename+".pickle")
		print("traj_simulations: ", [print(key,val.shape) for key,val in traj_simulations.items()])

		trajectories = traj_simulations["X"][:,:self.n_traj,:,:]
		initial_states = traj_simulations["Y_s0"]
		params = traj_simulations["Y_par"]
		initial_states = np.expand_dims(initial_states, axis=1)

		# [print(initial_states[i], params[i]) for i in range(50)]
		# exit()

		n_species = initial_states.shape[-1]
		n_params = params.shape[1]

		print("\ntrajectories.shape = ", trajectories.shape)
		print("initial_states.shape = ", initial_states.shape)
		print("params.shape = ", params.shape)
		print("n_species = ", n_species)
		print("n_params = ", n_params)


		# path = RESULTS+"trained_models/"+self.filename
		# trajectories = load_and_rescale(trajectories, path+"_traj")	
		# initial_states = load_and_rescale(initial_states, path+"_states")	
		# params = load_and_rescale(params, path+"_params")	

		return trajectories, initial_states, params

	# === DISTANCES ===

	def compute_distances(self, discriminator, generator, test_data):
		timesteps = self.timesteps
		noise_timesteps = self.noise_timesteps

		trajectories, initial_states, params = test_data 
		traj_per_state = trajectories.shape[1]
		n_species = trajectories.shape[-1]
		max_n_traj = int(traj_per_state/timesteps)*timesteps

		print(f"\nComputing histograms on {len(initial_states)} initial states")
		bins = 100
		gen_traj = np.empty(shape=(len(initial_states), max_n_traj, timesteps, n_species))
		ssa_histograms_count = np.empty(shape=(len(initial_states), timesteps, n_species, bins))
		ssa_histograms_x = np.empty(shape=(len(initial_states), timesteps, n_species, bins+1))
		gen_histograms_count = np.empty(shape=(len(initial_states), timesteps, n_species, bins))
		gen_histograms_x = np.empty(shape=(len(initial_states), timesteps, n_species, bins+1))
		dist = np.empty(shape=(len(initial_states), timesteps, n_species))

		for s, init_state in tqdm(enumerate(initial_states)):
			print("\tinit_state = ", init_state)
			ssa_trajectories = trajectories[s,:max_n_traj,:timesteps,:]
			init_state = np.expand_dims(init_state, 1)
			par = np.expand_dims(params[s,:], 0)
		
			for traj_idx, traj in enumerate(ssa_trajectories):			
				# print(noise.shape, init_state.shape, par.shape)
				noise = generate_noise(batch_size=1, noise_timesteps=noise_timesteps, 
					                   n_species=n_species)
				generated_trajectories = np.squeeze(generator.predict([noise, init_state, par]))
				gen_traj[s, traj_idx, :, :] = generated_trajectories[:max_n_traj,:]
			
			# print("\ntraj shapes:", ssa_trajectories.shape, gen_traj.shape)
			for t in range(timesteps):
				for m in range(n_species): 
					hist = np.histogram(ssa_trajectories[:,t,m], bins=bins)
					ssa_histograms_count[s, t, m, :] = hist[0]
					ssa_histograms_x[s, t, m, :] = hist[1]

					hist = np.histogram(gen_traj[:,t,m], bins=bins)
					gen_histograms_count[s, t, m, :] = hist[0]
					gen_histograms_x[s, t, m, :] = hist[1]

					dist[s, t, m] = wasserstein_distance(ssa_histograms_count[s,t,m,:], 
					                                     gen_histograms_count[s,t,m,:]) 

		print("\nhistograms shapes =", ssa_histograms_count.shape, ssa_histograms_x.shape)
		print("distances shape =", dist.shape)

		distances = {"ssa_count":ssa_histograms_count, "ssa_x":ssa_histograms_x,
		              "gen_count":gen_histograms_count, "gen_x":gen_histograms_x,
		              "wass_dist":dist}

		save_to_pickle(data=distances, relative_path=RESULTS+self.path, 
			           filename=self.filename+"_"+str(bins)+"_distances.pkl")	
		return distances

	def load_distances(self, rel_path, bins=100):
		distances = load_from_pickle(path=rel_path+self.path+self.filename+"_"+str(bins)+"_distances.pkl")
		print("\ndistances.shape = ", distances.shape)

	def plot_evolving_distances(self, distances, labels):
		import seaborn as sns 
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(1,1,figsize=(12,6))

		n_init_samples, n_timesteps, n_species = list(distances.shape)

		for s in range(n_species):
			timesteps = []
			traj_values = []
			for t in range(n_timesteps):
				[timesteps.append(t) for i in range(n_init_samples)]
				[traj_values.append(v) for v in distances[:,t,s]]
			# print(len(timesteps), len(traj_values))
			sns.lineplot(x=timesteps, y=traj_values, label=labels[s])

		ax.set_xlabel("timesteps")
		plt.tight_layout()
		os.makedirs(os.path.dirname(RESULTS+self.path), exist_ok=True)
		plt.savefig(RESULTS+self.path+self.filename+"_evolving_dist.png")
		plt.close()

	# === TRAJECTORIES ===

	def compute_trajectories(self, discriminator, generator, test_data):
		timesteps = self.timesteps
		noise_timesteps = self.noise_timesteps

		trajectories, initial_states, params = test_data 
		traj_per_state = trajectories.shape[1]
		n_species = trajectories.shape[-1]

		print(f"\nComputing trajectories on {len(initial_states)} initial states")
		gen_traj = np.empty(shape=(len(initial_states), traj_per_state, timesteps, n_species))

		for s, init_state in tqdm(enumerate(initial_states)):
			print("\tinit_state = ", init_state)
			ssa_trajectories = trajectories[s,:,:timesteps,:]
			init_state = np.expand_dims(init_state, 1)
			par = np.expand_dims(params[s,:], 0)
		
			for traj_idx, traj in enumerate(ssa_trajectories):			
				noise = generate_noise(batch_size=1, noise_timesteps=noise_timesteps, 
					                   n_species=n_species)
				# print(noise.shape, init_state.shape, par.shape)
				generated_trajectories = generator.predict([noise, init_state, par])
				generated_trajectories = np.round(generated_trajectories)
				gen_traj[s, traj_idx, :, :] = np.squeeze(generated_trajectories)
			
		trajectories = {"ssa":trajectories[:,:,:timesteps,:], "gen": gen_traj}
		save_to_pickle(data=trajectories, relative_path=RESULTS+"evaluations/", 
                       filename=self.filename+"_trajectories.pkl")	
		return trajectories

	def load_trajectories(self, rel_path):
		trajectories = load_from_pickle(path=rel_path+"evaluations/"+self.filename+"_trajectories.pkl")
		print("trajectories['ssa'].shape = ", trajectories["ssa"].shape)
		print("trajectories['gen'].shape = ", trajectories["gen"].shape)
		return trajectories

	def plot_trajectories(self, trajectories):
		import seaborn as sns 
		import matplotlib.pyplot as plt

		n_init_states, traj_per_state, n_timesteps, n_species = trajectories["ssa"].shape
		
		for init_state in range(n_init_states):

			fig, ax = plt.subplots(2,1,figsize=(12,6))

			ssa_fixed_init = trajectories["ssa"][init_state]
			gen_fixed_init = trajectories["gen"][init_state]

			for s in range(n_species):
				# print("\nssa: ", ssa_fixed_init[:10,:10,s])
				# print("gen: ", gen_fixed_init[:10,:10,s])

				for traj_idx in range(20):#(traj_per_state):
					sns.lineplot(range(n_timesteps), ssa_fixed_init[traj_idx,:,s], ax=ax[s], 
						         color="blue")
					sns.lineplot(range(n_timesteps), gen_fixed_init[traj_idx,:,s], ax=ax[s], 
						         color="orange")
					ax[s].set_xlabel("timesteps")

			os.makedirs(os.path.dirname(RESULTS+self.path), exist_ok=True)
			plt.savefig(RESULTS+self.path+self.filename+"_trajStateIdx="+str(init_state)+".png")
			plt.close()

	def plot_trajectories_dist(self, trajectories, bins=20):
		import seaborn as sns 
		import matplotlib.pyplot as plt

		n_init_states, traj_per_state, n_timesteps, n_species = trajectories["ssa"].shape
		ssa_flat = trajectories["ssa"].reshape((n_init_states*traj_per_state, n_timesteps, 
			                                    n_species))
		gen_flat = trajectories["gen"].reshape((n_init_states*traj_per_state, n_timesteps,
		                                        n_species))
		chosen_timesteps = [0, int(n_timesteps/2), n_timesteps-1]
		
		for init_state in range(n_init_states):

			fig, ax = plt.subplots(2,3,figsize=(12,6))

			ssa_fixed_init = trajectories["ssa"][init_state]
			gen_fixed_init = trajectories["gen"][init_state]

			for s in range(n_species):
				for t_idx, t in enumerate(chosen_timesteps):
					# print("\nssa: ", ssa_fixed_init[:100,t,s])
					# print("gen: ", gen_fixed_init[:100,t,s])

					sns.distplot(ssa_fixed_init[:,t,s], label="SSA", ax=ax[s,t_idx], kde=False, 
						         bins=bins)
					sns.distplot(gen_fixed_init[:,t,s], label="GEN", ax=ax[s,t_idx], kde=False, 
						         bins=bins)
					ax[s,t_idx].set_xlabel("timestep "+str(t))

			ax[0,0].legend()
			ax[1,0].legend()

			os.makedirs(os.path.dirname(RESULTS+self.path), exist_ok=True)
			plt.savefig(RESULTS+self.path+self.filename+"_traj_distr"+"_"+str(init_state)+".png")
			plt.close()


def main(args):

	# epochs_list = [50]
	# gen_epochs_list = [10]
	# noise_timesteps_list = [4,8]
	# combinations = list(itertools.product(epochs_list, gen_epochs_list, noise_timesteps_list))

	combinations = [(args.epochs, args.gen_epochs, args.noise_timesteps)]

	for (n_epochs, gen_epochs, noise_timesteps) in combinations:
		gan_eval = GAN_evaluator(model=args.model, timesteps=args.timesteps, 
			                     noise_timesteps=noise_timesteps, n_epochs=n_epochs,
			                     gen_epochs=gen_epochs, n_traj=args.n_traj, embed=args.embed,
			                     fixed_params=args.fixed_params, lr=args.lr)

		data = gan_eval.load_test_data(model=args.model)
		d, g = gan_eval.load_gan(rel_path=RESULTS)
		
		traj = gan_eval.compute_trajectories(discriminator=d, generator=g, test_data=data)
		# traj = gan_eval.load_trajectories(rel_path=RESULTS)
		gan_eval.plot_trajectories(trajectories=traj)
		# gan_eval.plot_trajectories_dist(trajectories=traj)

		# distances = gan_eval.compute_distances(discriminator=d, generator=g, test_data=data)
		# # distances = gan_eval.load_distances(rel_path=RESULTS)
		# gan_eval.plot_evolving_distances(distances["wass_dist"], labels=["S","I"])
		# # gan_eval.plot_distances_distr(histograms["wass_dist"])
		# exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("--model", default="eSIR", type=str)
    parser.add_argument("-n", "--n_traj", default=1000, type=int)
    parser.add_argument("-t", "--timesteps", default=128, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--gen_epochs", default=10, type=int)
    parser.add_argument("--noise_timesteps", default=128, type=int)
    parser.add_argument("--embed", default=0, type=int)
    parser.add_argument("--fixed_params", default=1, type=int)
    parser.add_argument("--lr", default="0.001", type=float)

    main(args=parser.parse_args())