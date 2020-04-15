import os
import argparse
from directories import *
from gan_ensemble import GAN_ensemble
from utils import load_from_pickle, generate_noise, save_to_pickle
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from tqdm import tqdm
import os
from keras import backend as K
import itertools


class EnsembleEvaluator(GAN_ensemble):

	def __init__(self, model, timesteps, noise_timesteps, fixed_params, gen_lr, discr_lr, n_epochs, 
                 gen_epochs, n_networks, n_traj):
		super(EnsembleEvaluator, self).__init__(model=model, fixed_params=fixed_params,
			                                timesteps=timesteps, noise_timesteps=noise_timesteps,
			                                n_epochs=n_epochs, gen_epochs=gen_epochs,
											gen_lr=gen_lr, discr_lr=discr_lr, n_networks=n_networks)
		self.n_traj=n_traj

	def load_gan(self, rel_path):
		return super(EnsembleEvaluator, self).load(rel_path=rel_path)

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

		idxs = np.random.randint(0, len(traj_simulations["X"]), self.n_traj*self.n_networks)
		trajectories = traj_simulations["X"][:,idxs]
		initial_states = traj_simulations["Y_s0"]
		params = traj_simulations["Y_par"]
		initial_states = np.expand_dims(initial_states, axis=1)
		
		self.n_species = initial_states.shape[-1]
		self.n_params = params.shape[1]

		params = np.expand_dims(params, axis=-1)
		params = np.repeat(params, self.n_species, axis=-1)

		print("\ntrajectories.shape = ", trajectories.shape)
		print("initial_states.shape = ", initial_states.shape)
		print("params.shape = ", params.shape)
		print("n_species = ", self.n_species)
		print("n_params = ", self.n_params)

		return trajectories, initial_states, params

	# === TRAJECTORIES ===

	def compute_trajectories(self, discriminators, generators, test_data):
		timesteps = self.timesteps
		noise_timesteps = self.noise_timesteps

		trajectories, initial_states, params = test_data 
		traj_per_state = int(trajectories.shape[1]/self.n_networks)
		n_species = trajectories.shape[-1]

		print(f"\nComputing trajectories on {len(initial_states)} initial states")

		gen_traj = []#np.empty(shape=(len(initial_states), traj_per_state, timesteps, n_species))

		for s_idx, s in tqdm(enumerate(initial_states)):
			print("\tinit_state = ", s)
			ssa_trajectories = trajectories[s_idx,:,:timesteps,:]
			s = np.expand_dims(s, 1)
			p = np.expand_dims(params[s_idx],0)

			generated_trajectories = []
			for _ in range(traj_per_state):		
				n = generate_noise(batch_size=1, noise_timesteps=noise_timesteps, n_species=n_species)
				latent_data = [n, s] if self.fixed_params==1 else [n, s, p]
				for generator in generators:
					generated_trajectories.append(np.round(generator.predict(latent_data)))
			
			gen_traj.append(np.squeeze(generated_trajectories))

		trajectories = {"ssa":trajectories[:,:,:timesteps,:], "gen": np.array(gen_traj)}
		save_to_pickle(data=trajectories, relative_path=RESULTS+self.name+"/trajectories/", 
                       filename="trajectories_"+str(self.n_traj)+".pkl")	
		return trajectories

	def load_trajectories(self, rel_path):
		path = rel_path+self.name+"/trajectories/trajectories_"+str(self.n_traj)+".pkl"
		trajectories = load_from_pickle(path=path)
		print("trajectories['ssa'].shape = ", trajectories["ssa"].shape)
		print("trajectories['gen'].shape = ", trajectories["gen"].shape)
		return trajectories

	def plot_trajectories(self, trajectories):
		import seaborn as sns 
		import matplotlib.pyplot as plt

		n_init_states, traj_per_state, n_timesteps, n_species = trajectories["ssa"].shape

		for s in range(n_init_states):

			fig, ax = plt.subplots(n_species,1,figsize=(12,6))

			# randomly choose n_traj trajectories for each init state
			idxs = np.random.randint(0, traj_per_state, self.n_traj)

			for m in range(n_species):
				for idx in idxs:
					sns.lineplot(range(n_timesteps), trajectories["ssa"][s,idx,:,m], ax=ax[m], 
						         color="blue")
					sns.lineplot(range(n_timesteps), trajectories["gen"][s,idx,:,m], ax=ax[m], 
						         color="orange")
					ax[m].set_xlabel("timesteps")

			path=RESULTS+self.name+"/trajectories/"
			os.makedirs(os.path.dirname(path), exist_ok=True)
			plt.savefig(path+"trajectories_stateIdx="+str(s)+".png")
			plt.close()


def main(args):

	combinations = [(args.epochs, args.gen_epochs, args.noise_timesteps)]

	for (n_epochs, gen_epochs, noise_timesteps) in combinations:
		gan_eval = EnsembleEvaluator(model=args.model, fixed_params=args.fixed_params, 
			                         n_traj=args.traj, n_networks=args.networks,
				                     timesteps=args.timesteps, noise_timesteps=noise_timesteps, 
				                     n_epochs=n_epochs, gen_epochs=gen_epochs, 
				                     discr_lr=args.discr_lr, gen_lr=args.gen_lr)

		data = gan_eval.load_test_data(model=args.model)
		d, g = gan_eval.load_gan(rel_path=RESULTS)
		
		traj = gan_eval.compute_trajectories(discriminators=d, generators=g, test_data=data)
		gan_eval.plot_trajectories(trajectories=traj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("--model", default="eSIR", type=str)
    parser.add_argument("--networks", default=5, type=int)
    parser.add_argument("--traj", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--timesteps", default=32, type=int)
    parser.add_argument("--noise_timesteps", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--gen_epochs", default=2, type=int)
    parser.add_argument("--fixed_params", default=1, type=int)
    parser.add_argument("--gen_lr", default="0.0001", type=float)
    parser.add_argument("--discr_lr", default="0.0001", type=float)

    main(args=parser.parse_args())