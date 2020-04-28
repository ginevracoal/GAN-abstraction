from gan_abstraction import GAN_abstraction
import argparse
from tensorflow import keras
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
import numpy as np
import tensorflow as tf
from utils import load_from_pickle, execution_time, generate_noise, save_to_pickle, rescale
import time
from directories import *
import os
import itertools
from gan_evaluation import GAN_evaluator


class SpeciesGANs(GAN_abstraction):
    def __init__(self, model, timesteps, noise_timesteps, fixed_params, gen_lr, discr_lr, n_epochs, 
                 gen_epochs):
        super(SpeciesGANs, self).__init__(model, timesteps, noise_timesteps, fixed_params, gen_lr, 
                                          discr_lr, n_epochs, gen_epochs)

        self.architecture = "conv1D"
        name = str(model)+"_t="+str(timesteps)+"_tNoise="+str(noise_timesteps)+\
                    "_ep="+str(n_epochs)+"_epG="+str(gen_epochs)+\
                    "_lrD="+str(discr_lr)+"_lrG="+str(gen_lr)
        name = name+"_fixPar" if fixed_params else name
        self.name = name+"_species"

    def train(self, training_data, batch_size):
        species_gans = {}

        for idx in range(self.n_species):
            print("\n === Training on specie ", idx, "===")
            specie_data = [np.expand_dims(data[:,:,idx],2) for data in training_data]

            classifier = GAN_abstraction(model=self.model, fixed_params=self.fixed_params,
                          timesteps=self.timesteps, noise_timesteps=self.noise_timesteps, 
                          discr_lr=self.discr_lr, gen_lr=self.gen_lr,
                          n_epochs=self.n_epochs, gen_epochs=self.gen_epochs)
            classifier.n_species = 1
            classifier.n_params = self.n_params
            classifier.architecture = self.architecture

            discr, gen = classifier.train(training_data=specie_data, batch_size=batch_size, seed=idx)
            self.save(discr, gen, idx, classifier.fig)
            species_gans.update({str(idx):[discr,gen]})

        return species_gans

    def save(self, discriminator, generator, idx, fig):
        print("\nSaving ", self.name, idx)
        path = RESULTS+self.name+"/trained_models/"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        discriminator.save(path+"discriminator_specie="+str(idx)+".h5")
        generator.save(path+"generator_specie="+str(idx)+".h5")
        fig.savefig(path+"training_specie="+str(idx)+".png")

    def load(self, rel_path):
        path = RESULTS+self.name+"/trained_models/"

        species_gans = {}
        for idx in range(self.n_species):
            classifier = GAN_abstraction(model=self.model, fixed_params=self.fixed_params,
                          timesteps=self.timesteps, noise_timesteps=self.noise_timesteps, 
                          discr_lr=self.discr_lr, gen_lr=self.gen_lr,
                          n_epochs=self.n_epochs, gen_epochs=self.gen_epochs)
            classifier.n_species = 1
            classifier.n_params = self.n_params
            classifier.architecture = "conv1D"
            discr = keras.models.load_model(path+"discriminator_specie="+str(idx)+".h5")
            gen = keras.models.load_model(path+"generator_specie="+str(idx)+".h5") 
            
            species_gans.update({str(idx):[discr,gen]})

        return species_gans


class SpeciesEval(GAN_evaluator):

    def __init__(self, model, timesteps, noise_timesteps, fixed_params, gen_lr, discr_lr, n_epochs, 
                 gen_epochs, n_traj):
        super(SpeciesEval, self).__init__(model, timesteps, noise_timesteps, fixed_params, gen_lr,
                                          discr_lr, n_epochs, gen_epochs, n_traj)
        name = str(model)+"_t="+str(timesteps)+"_tNoise="+str(noise_timesteps)+\
                    "_ep="+str(n_epochs)+"_epG="+str(gen_epochs)+\
                    "_lrD="+str(discr_lr)+"_lrG="+str(gen_lr)
        name = name+"_fixPar" if fixed_params else name
        self.path = name+"_species/"

    def compute_trajectories(self, species_gans, test_data):
        species_traj = {}
        for specie_idx, specie_gan in species_gans.items():
            trajectories, initial_states, params = test_data 
            specie_data = (np.expand_dims(trajectories[:,:,:,int(specie_idx)],3),
                           np.expand_dims(initial_states[:,:,int(specie_idx)],2),
                           np.expand_dims(params[:,:,int(specie_idx)],2))
            discriminator, generator = specie_gan
            traj = super(SpeciesEval, self).compute_trajectories(discriminator, generator, 
                                            specie_data, int(specie_idx))
            species_traj.update({str(specie_idx):traj})
        
        return species_traj

    def save_trajectories(self, data):

        for specie_idx, specie_gan in data.items():
            filename = "trajectories_"+str(self.n_traj)+"_"+str(specie_idx)+".pkl"
            save_to_pickle(data=data, relative_path=RESULTS+self.path+"trajectories/", 
                           filename=filename)

    def load_trajectories(self, rel_path):

        path = rel_path+self.path+"trajectories/"
        species_traj = {}
        
        for specie_idx in range(self.n_species):

            filename = "trajectories_"+str(self.n_traj)+"_"+str(specie_idx)+".pkl"
            trajectories = load_from_pickle(path=path+filename)
            species_traj.update({str(specie_idx):trajectories})

        return species_traj

    def plot_trajectories(self, trajectories, n_traj):
        import seaborn as sns 
        import matplotlib.pyplot as plt

        for m, specie_traj in trajectories.items():
            n_init_states, traj_per_state, n_timesteps, n_species = specie_traj["ssa"].shape
            
            # randomly choose n_traj trajectories to plot for each init state
            idxs = np.random.randint(0, self.n_traj, n_traj)

            # 20 init states grid
            fig, axs = plt.subplots(4,5,figsize=(12,6), dpi=300)

            for s, ax in enumerate(axs.reshape(-1)): 
                for idx in idxs:

                    sns.lineplot(range(n_timesteps), specie_traj["ssa"][s,idx,:,0], 
                                 ax=ax, color="blue")
                    sns.lineplot(range(n_timesteps), specie_traj["gen"][s,idx,:,0], 
                                 ax=ax, color="orange")
                    ax.set_xlabel("")
                    # ax.set_title(f"init_state = {specie_traj["ssa"][s,0,:,0]}")

            # x label = timesteps

            path=RESULTS+self.path+"trajectories/"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path+"trajectories_specie="+m+".png")
            plt.close()


def main(args):

    # === train ===
    gan = SpeciesGANs(model=args.model, fixed_params=args.fixed_params,
                      timesteps=args.timesteps, noise_timesteps=args.noise_timesteps, 
                      discr_lr=args.discr_lr, gen_lr=args.gen_lr,
                      n_epochs=args.epochs, gen_epochs=args.gen_epochs)
    training_data = gan.load_data(n_traj=args.traj, model=args.model, timesteps=args.timesteps)
    
    species_gans = gan.train(training_data=training_data,  batch_size=args.batch_size)
    # species_gans = gan.load(rel_path=RESULTS)

    # === eval ===
    species_eval = SpeciesEval(model=args.model, fixed_params=args.fixed_params, 
                             n_traj=args.eval_traj,
                             timesteps=args.timesteps, noise_timesteps=args.noise_timesteps, 
                             n_epochs=args.epochs, gen_epochs=args.gen_epochs, 
                             discr_lr=args.discr_lr, gen_lr=args.gen_lr)

    test_data = species_eval.load_test_data(model=args.model)
    traj = species_eval.compute_trajectories(species_gans=species_gans, test_data=test_data)
    species_eval.save_trajectories(data=traj)
    # traj = species_eval.load_trajectories(rel_path=RESULTS)
    species_eval.plot_trajectories(trajectories=traj, n_traj=args.eval_traj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("--model", default="eSIR", type=str)
    parser.add_argument("--traj", default=1000, type=int)
    parser.add_argument("--eval_traj", default=10, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--timesteps", default=32, type=int)
    parser.add_argument("--noise_timesteps", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--gen_epochs", default=2, type=int)
    parser.add_argument("--fixed_params", default=0, type=int)
    parser.add_argument("--gen_lr", default="0.0001", type=float)
    parser.add_argument("--discr_lr", default="0.0001", type=float)

    main(args=parser.parse_args())