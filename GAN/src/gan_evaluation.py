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
import itertools


class GAN_evaluator(GAN_abstraction):

	def __init__(self, model, timesteps, noise_timesteps, epochs, gen_epochs):
		super(GAN_evaluator, self).__init__(model, timesteps, noise_timesteps)
		self.epochs = epochs
		self.gen_epochs = gen_epochs
		self.filename = self.model+"_t="+str(self.timesteps)+"_tNoise="+\
		                str(self.noise_timesteps)+"_epochs="+str(self.epochs)+\
		                "_epochsGen="+str(self.gen_epochs)

	def load_gan(self, rel_path):
		return super(GAN_evaluator, self).load(rel_path=rel_path, n_epochs=self.epochs, 
			                                   gen_epochs=self.gen_epochs)

	def load_test_data(self, model, path="../../SSA/data/test/"):
		
		if model == "SIR":
			filename = "SIR_test_set.pickle"
		elif model == "eSIR":
			filename = "eSIR_test_set.pickle"
		elif model == "Repress":
			filename = "Repressilator_test_set_indip_vars.pickle"
		elif model == "Toggle":
			filename = "ToggleSwitch_test_set_indip_vars.pickle"

		traj_simulations = load_from_pickle(path=path+filename)
		print("traj_simulations: ", [print(key,val.shape) for key,val in traj_simulations.items()])

		trajectories = traj_simulations["X"]
		initial_states = traj_simulations["Y_s0"]
		params = traj_simulations["Y_par"]
		initial_states = np.expand_dims(initial_states, axis=1)

		n_species = initial_states.shape[-1]
		n_params = params.shape[1]

		print("\ntrajectories.shape = ", trajectories.shape)
		print("initial_states.shape = ", initial_states.shape)
		print("params.shape = ", params.shape)
		print("n_species = ", n_species)
		print("n_params = ", n_params)

		return trajectories, initial_states, params

	def evaluate(self, discriminator, generator, test_data):
		timesteps = self.timesteps
		noise_timesteps = self.noise_timesteps

		trajectories, initial_states, params = test_data 
		traj_per_state = 200 #trajectories.shape[1]
		n_species = trajectories.shape[-1]
		max_n_traj = int(traj_per_state/timesteps)*timesteps

		print(f"\nComputing histograms on {len(initial_states)} initial states")
		bins = 10
		gen_traj = np.empty(shape=(len(initial_states), traj_per_state, timesteps, n_species))
		ssa_histograms_count = np.empty(shape=(len(initial_states), timesteps, n_species, bins))
		ssa_histograms_x = np.empty(shape=(len(initial_states), timesteps, n_species, bins+1))
		gen_histograms_count = np.empty(shape=(len(initial_states), timesteps, n_species, bins))
		gen_histograms_x = np.empty(shape=(len(initial_states), timesteps, n_species, bins+1))
		dist = np.empty(shape=(len(initial_states), timesteps, n_species))

		for s, init_state in tqdm(enumerate(initial_states)):
			print("\tinit_state = ", init_state)
			state_trajectories = trajectories[s,:max_n_traj,:timesteps,:]
			init_state = np.expand_dims(init_state, 1)
			par = np.expand_dims(params[s,:], 0)
		
			for traj_idx, traj in enumerate(state_trajectories):			
				# print(noise.shape, init_state.shape, par.shape)
				noise = generate_noise(batch_size=1, noise_timesteps=noise_timesteps, 
					                   n_species=n_species)
				generated_trajectories = np.squeeze(generator.predict([noise, init_state, par]))
				gen_traj[s, traj_idx, :, :] = generated_trajectories
						
			for t in range(timesteps):
				for m in range(n_species): 
					hist = np.histogram(state_trajectories[:,t,m], bins=bins)
					ssa_histograms_count[s, t, m, :] = hist[0]
					ssa_histograms_x[s, t, m, :] = hist[1]

					hist = np.histogram(gen_traj[:,t,m], bins=bins)
					gen_histograms_count[s, t, m, :] = hist[0]
					gen_histograms_x[s, t, m, :] = hist[1]

					dist[s, t, m] = wasserstein_distance(ssa_histograms_count[s,t,m,:], 
					                                     gen_histograms_count[s,t,m,:]) 

		print("\nhistograms shapes =", ssa_histograms_count.shape, ssa_histograms_x.shape)
		print("distances shape =", dist.shape)

		save_to_pickle(data=dist, relative_path=RESULTS, filename=self.filename+"_distances.pkl")
		return dist

	def load_distances(self, rel_path):
		distances = load_from_pickle(path=rel_path+self.filename+"_distances.pkl")
		print("\ndistances.shape =", distances.shape)
		return distances

	def plot_evolving_distributions(self, distances):
		import seaborn as sns 
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(1,1,figsize=(12,6))

		n_init_samples, n_timesteps, n_species = list(distances.shape)

		labels = ["S","I"]
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
		os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
		plt.savefig(RESULTS+self.filename+"_evolving_dist.png")

	def plot_distplots(self, distances):
		import seaborn as sns 
		import matplotlib.pyplot as plt

		n_init_samples, n_timesteps, n_species = list(distances.shape)
		chosen_timesteps = [0,int(n_timesteps/2),n_timesteps-1]
		
		fig, ax = plt.subplots(1,3,figsize=(12,6))

		labels = ["S","I"]
		for s in range(n_species):
			timesteps = []
			traj_values = []
			for t_idx, t in enumerate(chosen_timesteps):
				# [timesteps.append(t) for i in range(n_init_samples)]
				# [traj_values.append(v) for v in distances[:,t,s]]
			# print(len(timesteps), len(traj_values))
				sns.distplot(distances[:,t,s], label=labels[s], ax=ax[t_idx], bins=10)
				ax[t_idx].set_xlabel("timestep "+str(t))

		plt.tight_layout()
		os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
		plt.savefig(RESULTS+self.filename+"_distplots.png")



def main(args):

	epochs_list = [150,200,300]
	gen_epochs_list = [5,10]
	noise_timesteps_list = [4,8,12]
	combinations = list(itertools.product(epochs_list, gen_epochs_list, noise_timesteps_list))

	for (epochs, gen_epochs, noise_timesteps) in combinations:
		gan_eval = GAN_evaluator(model=args.model, timesteps=args.timesteps, 
			                     noise_timesteps=noise_timesteps, epochs=epochs,
			                     gen_epochs=gen_epochs)
		discriminator, generator = gan_eval.load_gan(rel_path=RESULTS)
		test_data = gan_eval.load_test_data(model=args.model)

		distances = gan_eval.evaluate(discriminator=discriminator, generator=generator, 
			                     test_data=test_data)
		# distances = gan_eval.load_distances(rel_path=RESULTS)
		
		gan_eval.plot_evolving_distributions(distances)
		gan_eval.plot_distplots(distances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("-n", "--n_traj", default=1000, type=int)
    parser.add_argument("-t", "--timesteps", default=128, type=int)
    parser.add_argument("--noise_timesteps", default=5, type=int)
    parser.add_argument("--model", default="eSIR", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--gen_epochs", default=5, type=int)

    main(args=parser.parse_args())