import os
import argparse
from directories import *
from gan_abstraction import GAN_abstraction
from utils import load_from_pickle, generate_noise
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from tqdm import tqdm

# todo: classe che estenda GAN abstraction


def load_test_data(model, path="../../SSA/data/test/"):
	
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


def distplot(real_dist, fake_dist):
	import seaborn as sns 
	import matplotlib.pyplot as plt

	pass


def evaluate(discriminator, generator, test_data, timesteps, noise_timesteps):

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
	print("distances shape =", dist.shape, "\n", dist)

	return dist


def main(args):

	gan = GAN_abstraction(args.model, args.timesteps, args.noise_timesteps)
	discriminator, generator = gan.load(rel_path=RESULTS, n_epochs=args.epochs, 
		                                gen_epochs=args.gen_epochs)

	test_data = load_test_data(model=args.model)
	evaluate(discriminator=discriminator, generator=generator, test_data=test_data, 
			 timesteps=args.timesteps, noise_timesteps=args.noise_timesteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("-n", "--n_traj", default=1000, type=int)
    parser.add_argument("-t", "--timesteps", default=118, type=int)
    parser.add_argument("--noise_timesteps", default=5, type=int)
    parser.add_argument("--model", default="eSIR", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--gen_epochs", default=5, type=int)

    main(args=parser.parse_args())