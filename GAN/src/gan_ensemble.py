from gan_abstraction import GAN_abstraction
import argparse
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv1D, LeakyReLU, Dropout, Concatenate, \
                                    Embedding, Flatten, Reshape, RepeatVector, Permute, \
                                    Lambda, BatchNormalization, Conv2D
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
import numpy as np
import tensorflow as tf
from utils import load_from_pickle, execution_time, generate_noise, save_to_pickle, rescale
import time
from directories import *
import os
import itertools


class GAN_ensemble(GAN_abstraction):

    def __init__(self, model, timesteps, noise_timesteps, fixed_params, gen_lr, discr_lr, n_epochs, 
                 gen_epochs, n_networks):
        super(GAN_ensemble, self).__init__(model, timesteps, noise_timesteps, fixed_params, gen_lr, 
                                           discr_lr, n_epochs, gen_epochs)
        self.n_networks = n_networks
        self.classifiers = None

        name = str(model)+"_t="+str(timesteps)+"_tNoise="+str(noise_timesteps)+\
                    "_ep="+str(n_epochs)+"_epG="+str(gen_epochs)+\
                    "_lrD="+str(discr_lr)+"_lrG="+str(gen_lr)+"_"+str(self.architecture)
        name = name+"_fixPar" if fixed_params else name
        self.name = name+"_GANens"

    def train(self, training_data, batch_size):
        classifiers = []
        for idx in range(self.n_networks):
            print("\n === Training GAN ", idx, "===")
            classifier = GAN_abstraction(model=self.model, fixed_params=self.fixed_params,
                          timesteps=self.timesteps, noise_timesteps=self.noise_timesteps, 
                          discr_lr=self.discr_lr, gen_lr=self.gen_lr,
                          n_epochs=self.n_epochs, gen_epochs=self.gen_epochs)

            classifier.n_species = self.n_species 
            classifier.n_params = self.n_params
            discr, gen = classifier.train(training_data=training_data, batch_size=batch_size, seed=idx)
            self.save(discr, gen, idx, classifier.fig)

    def save(self, discriminator, generator, idx, fig):
        print("\nSaving ", self.name, idx)
        path = RESULTS+self.name+"/trained_models/"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        discriminator.save(path+"discriminator_"+str(idx)+".h5")
        generator.save(path+"generator_"+str(idx)+".h5")
        fig.savefig(path+"training_"+str(idx)+".png")

    def load(self, rel_path):
        path = RESULTS+self.name+"/trained_models/"

        discriminators = []
        generators = []
        for idx in range(self.n_networks):
            classifier = GAN_abstraction(model=self.model, fixed_params=self.fixed_params,
                          timesteps=self.timesteps, noise_timesteps=self.noise_timesteps, 
                          discr_lr=self.discr_lr, gen_lr=self.gen_lr,
                          n_epochs=self.n_epochs, gen_epochs=self.gen_epochs)
            discr = keras.models.load_model(path+"discriminator_"+str(idx)+".h5")
            gen = keras.models.load_model(path+"generator_"+str(idx)+".h5") 
            discriminators.append(discr)
            generators.append(gen)

        return discriminators, generators

def main(args):

    gan = GAN_ensemble(model=args.model, fixed_params=args.fixed_params,
                          timesteps=args.timesteps, noise_timesteps=args.noise_timesteps, 
                          discr_lr=args.discr_lr, gen_lr=args.gen_lr,
                          n_epochs=args.epochs, gen_epochs=args.gen_epochs,
                          n_networks=args.networks)

    training_data = gan.load_data(n_traj=args.traj, model=args.model, timesteps=args.timesteps)

    gan.train(training_data=training_data,  batch_size=args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("--model", default="eSIR", type=str)
    parser.add_argument("--traj", default=1000, type=int)
    parser.add_argument("--networks", default=5, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--timesteps", default=32, type=int)
    parser.add_argument("--noise_timesteps", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--gen_epochs", default=2, type=int)
    parser.add_argument("--fixed_params", default=1, type=int)
    parser.add_argument("--gen_lr", default="0.0001", type=float)
    parser.add_argument("--discr_lr", default="0.0001", type=float)

    main(args=parser.parse_args())