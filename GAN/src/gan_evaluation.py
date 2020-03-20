import os
import argparse
from directories import *
from gan_abstraction import GAN_abstraction
from utils import load_from_pickle, generate_noise, save_to_pickle
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from tqdm import tqdm
import os
from keras import backend as K
import itertools


class GAN_evaluator(GAN_abstraction):

	def __init__(self, model, timesteps, noise_timesteps, fixed_params, gen_lr, discr_lr, n_epochs, 
                 gen_epochs, n_traj):
		super(GAN_evaluator, self).__init__(model=model, fixed_params=fixed_params,
			                                timesteps=timesteps, noise_timesteps=noise_timesteps,
			                                n_epochs=n_epochs, gen_epochs=gen_epochs,
											gen_lr=gen_lr, discr_lr=discr_lr)
		self.n_traj=n_traj

	def load_gan(self, rel_path):
		return super(GAN_evaluator, self).load(rel_path=rel_path)

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
		params = np.expand_dims(params, axis=-1)
		params = np.concatenate((params,params),axis=-1)

		self.n_species = initial_states.shape[-1]
		self.n_params = params.shape[1]

		print("\ntrajectories.shape = ", trajectories.shape)
		print("initial_states.shape = ", initial_states.shape)
		print("params.shape = ", params.shape)
		print("n_species = ", self.n_species)
		print("n_params = ", self.n_params)
		return trajectories, initial_states, params

	# === DISTANCES ===

	def compute_distances(self, trajectories, bins=100):

		n_init_states = trajectories["ssa"].shape[0]
		traj_per_state = trajectories["ssa"].shape[1]

		print(f"\nComputing histograms on {n_init_states} initial states")
		ssa_histograms_count = np.empty(shape=(n_init_states, self.timesteps, self.n_species, bins))
		gen_histograms_count = np.empty(shape=(n_init_states, self.timesteps, self.n_species, bins))
		ssa_histograms_x = np.empty(shape=(n_init_states, self.timesteps, self.n_species, bins+1))
		gen_histograms_x = np.empty(shape=(n_init_states, self.timesteps, self.n_species, bins+1))
		dist = np.empty(shape=(n_init_states, self.timesteps, self.n_species))

		for s in range(n_init_states):
			for t in range(self.timesteps):
				for m in range(self.n_species): 

					ssa_traj = trajectories["ssa"][s, :, t, m]
					gen_traj = trajectories["gen"][s, :, t, m]
					dist[s, t, m] = wasserstein_distance(ssa_traj, gen_traj)

		print("\nhistograms shapes =", ssa_histograms_count.shape, ssa_histograms_x.shape)
		print("distances shape =", dist.shape)

		distances = {"ssa_count":ssa_histograms_count, "ssa_x":ssa_histograms_x,
		              "gen_count":gen_histograms_count, "gen_x":gen_histograms_x,
		              "wass_dist":dist}

		return distances["wass_dist"]

	def plot_evolving_distances(self, distances, labels=None):
		import seaborn as sns 
		import matplotlib.pyplot as plt

		n_init_states = distances.shape[0]

		for s in range(n_init_states):

			fig, ax = plt.subplots(1,1,figsize=(12,6))	
			for m in range(self.n_species):
				sns.lineplot(x=range(self.timesteps), y=distances[s,:,m])
			ax.set_xlabel("timesteps")

			path=RESULTS+self.path+"distances/"
			os.makedirs(os.path.dirname(path), exist_ok=True)
			plt.savefig(path+"evolving_dist_stateIdx="+str(s)+".png")
			plt.close()

		fig, ax = plt.subplots(1,1,figsize=(12,6))	
		for m in range(self.n_species):
			distances_sum_over_init_states = np.sum(distances[:,:,m], axis=0)
			sns.lineplot(x=range(self.timesteps), y=distances_sum_over_init_states)
		ax.set_xlabel("timesteps")

		path=RESULTS+self.path
		os.makedirs(os.path.dirname(path), exist_ok=True)
		plt.savefig(path+"evolving_dist_sumOverStates.png")
		plt.close()


	# def plot_cumulative_distances():


	# === TRAJECTORIES ===

	def compute_trajectories(self, discriminator, generator, test_data):
		timesteps = self.timesteps
		noise_timesteps = self.noise_timesteps

		trajectories, initial_states, params = test_data 
		traj_per_state = trajectories.shape[1]
		n_species = trajectories.shape[-1]

		print(f"\nComputing trajectories on {len(initial_states)} initial states")
		gen_traj = np.empty(shape=(len(initial_states), traj_per_state, timesteps, n_species))

		for s_idx, s in tqdm(enumerate(initial_states)):
			print("\tinit_state = ", s)
			ssa_trajectories = trajectories[s_idx,:,:timesteps,:]
			s = np.expand_dims(s, 1)
			p = np.expand_dims(params[s_idx],0)
			for t_idx, t in enumerate(ssa_trajectories):		
				n = generate_noise(batch_size=1, noise_timesteps=noise_timesteps, n_species=n_species)
				latent_data = [n, s] if self.fixed_params==1 else [n, s, p]
				generated_trajectories = np.round(generator.predict(latent_data))
				gen_traj[s_idx, t_idx, :, :] = np.squeeze(generated_trajectories)

		trajectories = {"ssa":trajectories[:,:,:timesteps,:], "gen": gen_traj}
		save_to_pickle(data=trajectories, relative_path=RESULTS+self.path+"trajectories/", 
                       filename="trajectories_"+str(self.n_traj)+".pkl")	
		return trajectories

	def load_trajectories(self, rel_path):
		path = rel_path+self.path+"trajectories/trajectories_"+str(self.n_traj)+".pkl"
		trajectories = load_from_pickle(path=path)
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

				for traj_idx in range(self.n_traj):
					sns.lineplot(range(n_timesteps), ssa_fixed_init[traj_idx,:,s], ax=ax[s], 
						         color="blue")
					sns.lineplot(range(n_timesteps), gen_fixed_init[traj_idx,:,s], ax=ax[s], 
						         color="orange")
					ax[s].set_xlabel("timesteps")

			path=RESULTS+self.path+"trajectories/"
			os.makedirs(os.path.dirname(path), exist_ok=True)
			plt.savefig(path+"trajectories_stateIdx="+str(init_state)+".png")
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

					sns.distplot(ssa_fixed_init[:,t,s], label="SSA", ax=ax[s,t_idx], kde=False, 
						         bins=bins)
					sns.distplot(gen_fixed_init[:,t,s], label="GEN", ax=ax[s,t_idx], kde=False, 
						         bins=bins)
					ax[s,t_idx].set_xlabel("timestep "+str(t))

			ax[0,0].legend()
			ax[1,0].legend()

			path=RESULTS+self.path+"trajectories_distributions/"
			os.makedirs(os.path.dirname(path), exist_ok=True)
			plt.savefig(path+"traj_distr_stateIdx="+str(init_state)+".png")
			plt.close()


def main(args):

	# epochs_list = [50]
	# gen_epochs_list = [10]
	# noise_timesteps_list = [4,8]
	# combinations = list(itertools.product(epochs_list, gen_epochs_list, noise_timesteps_list))

	combinations = [(args.epochs, args.gen_epochs, args.noise_timesteps)]

	for (n_epochs, gen_epochs, noise_timesteps) in combinations:
		gan_eval = GAN_evaluator(model=args.model, fixed_params=args.fixed_params, n_traj=args.traj,
			                     timesteps=args.timesteps, noise_timesteps=noise_timesteps, 
			                     n_epochs=n_epochs, gen_epochs=gen_epochs, 
			                     discr_lr=args.discr_lr, gen_lr=args.gen_lr)

		data = gan_eval.load_test_data(model=args.model)
		# d, g = gan_eval.load_gan(rel_path=RESULTS)
		d, g = gan_eval.load_gan(rel_path=DATA)
		
		traj = gan_eval.compute_trajectories(discriminator=d, generator=g, test_data=data)
		# traj = gan_eval.load_trajectories(rel_path=RESULTS)

		gan_eval.plot_trajectories(trajectories=traj)
		# # gan_eval.plot_trajectories_dist(trajectories=traj)

		distances = gan_eval.compute_distances(trajectories=traj)
		gan_eval.plot_evolving_distances(distances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("--model", default="eSIR", type=str)
    parser.add_argument("--traj", default=20, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--timesteps", default=128, type=int)
    parser.add_argument("--noise_timesteps", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--gen_epochs", default=10, type=int)
    parser.add_argument("--fixed_params", default=1, type=int)
    parser.add_argument("--gen_lr", default="0.0001", type=float)
    parser.add_argument("--discr_lr", default="0.0001", type=float)

    main(args=parser.parse_args())