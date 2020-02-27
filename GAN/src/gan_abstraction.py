import argparse
from tensorflow.keras.layers import Input, Dense, Conv1D, LeakyReLU, Dropout, Concatenate, \
                                    Embedding, Flatten, Reshape, RepeatVector
from tensorflow.keras.models import Sequential, Model
import numpy as np
import tensorflow as tf
from utils import load_from_pickle, execution_time
import time
from directories import *
import os
import itertools


class GAN_abstraction:

    def __init__(self, model, timesteps, noise_timesteps):
        self.model = model
        self.timesteps = timesteps
        self.noise_timesteps = noise_timesteps

    def load_data(self, n_traj, model, timesteps, path="../../SSA/data/"):
        if model == "SIR":
            filename = "SIR_training_set.pickle"
        elif model == "eSIR":
            filename = "eSIR_training_set.pickle"
        elif model == "Repress":
            fiename = "Repressilator_training_set_indip_vars.pickle"
        elif model == "Toggle":
            filename = "ToggleSwitch_training_set_indip_vars.pickle"

        traj_simulations = load_from_pickle(path=path+filename)
        print("traj_simulations: ", [print(key,val.shape) for key,val in traj_simulations.items()])

        trajectories = traj_simulations["X"][:n_traj,:timesteps,:]
        initial_states = traj_simulations["Y_s0"][:n_traj]
        params = traj_simulations["Y_par"][:n_traj]
        initial_states = np.expand_dims(initial_states, axis=1)

        self.n_species = initial_states.shape[-1]
        self.n_params = params.shape[1]

        print("\ntrajectories.shape = ", trajectories.shape)
        print("initial_states.shape = ", initial_states.shape)
        print("params.shape = ", params.shape)
        print("n_species = ", self.n_species)
        print("n_params = ", self.n_params)
        print("noise_timesteps = ", self.noise_timesteps)

        return trajectories, initial_states, params

    # def reshape_training_data(self, trajectories, initial_states, params):
    #     initial_states = np.expand_dims(initial_states, axis=1)
    #     full_trajectories = np.concatenate((initial_states, trajectories), axis=1)
    #     rep_params = np.repeat(params, full_trajectories.shape[1])
    #     rep_params = np.reshape(rep_params, (params.shape[0], full_trajectories.shape[1], params.shape[1]))
    #     x_train = np.concatenate((full_trajectories, rep_params), axis=2)   
    #     return x_train

    def generate_noise(self, batch_size, noise_timesteps):
        if noise_timesteps > self.timesteps:
            raise ValueError("noise_timesteps should be smaller than trajectories timesteps.")
        noise = np.random.rand(batch_size, noise_timesteps, self.n_species)
        return noise

    def generator(self, noise_timesteps, trajectories_timesteps):

        noise = Input(shape=(noise_timesteps, self.n_species)) 
        init_states = Input(shape=(1,self.n_species))
        par = Input(shape=(self.n_params,))
        full_traj = Concatenate(axis=1)([init_states,noise])
        embedded_par = Reshape((noise_timesteps+1,1))(Dense((noise_timesteps+1))(par))
        inputs = Concatenate(axis=-1)([full_traj,embedded_par])

        x = Conv1D(64, 3)(inputs)
        x = LeakyReLU()(x)
        x = Conv1D(128, 3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense((trajectories_timesteps)*(self.n_species), activation="relu")(x)
        outputs = Reshape((trajectories_timesteps, self.n_species))(x)

        model = Model(inputs=[noise,init_states,par], outputs=outputs)

        return model

    def discriminator(self, trajectories_timesteps):

        traj = Input(shape=(trajectories_timesteps, self.n_species)) 
        init_states = Input(shape=(1,self.n_species))
        par = Input(shape=(self.n_params,))
        full_traj = Concatenate(axis=1)([init_states,traj])
        embedded_par = Reshape((trajectories_timesteps+1,1))(Dense((trajectories_timesteps+1))(par))
        inputs = Concatenate(axis=-1)([full_traj,embedded_par])

        x = Conv1D(64, 2)(inputs)
        x = LeakyReLU()(x)
        x = Conv1D(128, 3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=[traj,init_states,par], outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
        return model

    def gan(self, discriminator, generator): 
        discriminator.trainable = False

        noise, init_states, par = generator.input
        gen_traj = generator.output
        gan_output = discriminator([gen_traj, init_states, par])
        model = Model(inputs=[noise, init_states, par], outputs=gan_output)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, training_data, n_epochs, gen_epochs, batch_size, noise_timesteps, trajectories_timesteps):
        trajectories, initial_states, params = training_data 
        n_batches = int(len(initial_states) / batch_size)+1

        generator = self.generator(noise_timesteps=noise_timesteps, 
                                   trajectories_timesteps=trajectories_timesteps)
        discriminator = self.discriminator(trajectories_timesteps=trajectories_timesteps)
        gan = self.gan(discriminator, generator)

        g_loss=0
        d_loss1=0
        d_loss2=0
        d_acc1=0
        d_acc2=0

        g_loss_list=[]
        d_loss1_list=[]
        d_loss2_list=[]
        d_acc1_list=[]
        d_acc2_list=[]

        start = time.time()
        for epoch in range(n_epochs):
            
            for batch_idx in range(n_batches-1):
                begin, end = batch_idx*batch_size, (batch_idx+1)*batch_size
                traj = trajectories[begin:end,:,:]
                init_states = initial_states[begin:end,:]
                par = params[begin:end,:]

                y_train_real = np.ones(len(init_states))
                d_loss1, d_acc1 = discriminator.train_on_batch([traj, init_states, par], y_train_real)

                noise = self.generate_noise(len(init_states), noise_timesteps)
                gen_traj = generator.predict([noise, init_states, par])
                y_train_fake = np.zeros(len(init_states))
                d_loss2, d_acc2 = discriminator.train_on_batch([gen_traj, init_states, par], y_train_fake)

                for _ in range(gen_epochs):
                    noise = self.generate_noise(len(init_states), noise_timesteps)
                    g_loss = gan.train_on_batch(x=[noise, init_states, par], y=y_train_real)

            d_loss1_list.append(d_loss1)
            d_acc1_list.append(d_acc1)
            d_loss2_list.append(d_loss2)
            d_acc2_list.append(d_acc2)   
            g_loss_list.append(g_loss)

            print(f"\n[Epoch {epoch + 1}]\t g_loss = {g_loss:.4f}", end="\t")
            print(f"d_loss1 = {d_loss1:.4f}\td_loss2 = {d_loss2:.4f}", end="\t")
            print(f"a1 = {int(100*d_acc1)}\ta2 = {int(100*d_acc2)}", end="\t")

        print("\n")
        execution_time(start=start, end=time.time())

        os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
        filename = self.model+"_t="+str(self.timesteps)+"_tNoise="+str(self.noise_timesteps)+\
                   "_epochs="+str(n_epochs)+"_epochsGen="+str(gen_epochs)
        discriminator.save(RESULTS+filename+"_discriminator.h5")
        generator.save(RESULTS+filename+"_generator.h5")

        plot_training(n_epochs, g_loss_list, d_loss1_list, d_loss2_list, d_acc1_list, d_acc2_list, filename)

        return discriminator, generator

    def load(rel_path):
        discriminator = keras.models.load_model(rel_path+self.model+"_discriminator.h5")
        generator = keras.models.load_model(rel_path+self.model+"_generator.h5")
        return discriminator, generator


def plot_training(n_epochs, g_loss, d_loss1, d_loss2, d_acc1, d_acc2, filename):

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.lineplot(range(1, n_epochs+1), g_loss, label='Generator loss')
    sns.lineplot(range(1, n_epochs+1), d_loss1, label='Discriminator loss real')
    sns.lineplot(range(1, n_epochs+1), d_loss2, label='Discriminator loss gen')
    sns.lineplot(range(1, n_epochs+1), d_acc1, label='Discriminator train acc real')
    sns.lineplot(range(1, n_epochs+1), d_acc2, label='Discriminator train acc gen')
    plt.tight_layout()

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS+filename+".png")

def _parallel_grid_search(model, n_traj, batch_size, epochs, gen_epochs, timesteps, noise_timesteps):

    gan = GAN_abstraction(model, timesteps, noise_timesteps)
    training_data = gan.load_data(n_traj=n_traj, model=model, timesteps=timesteps)
    gan.train(training_data=training_data, n_epochs=epochs, batch_size=batch_size,
              noise_timesteps=noise_timesteps, trajectories_timesteps=timesteps, gen_epochs=gen_epochs)


def grid_search(model, n_traj, batch_size):

    print("\n == Grid search training == ")
    from joblib import Parallel, delayed

    epochs_list = [150,200,300]
    gen_epochs_list = [5,10]
    timesteps_list = [64,96,118]
    noise_timesteps_list = [4,8,12]
    combinations = list(itertools.product(epochs_list, gen_epochs_list, timesteps_list, noise_timesteps_list))

    Parallel(n_jobs=10)(
        delayed(_parallel_grid_search)(model, n_traj, batch_size, epochs, gen_epochs, timesteps, 
                                       noise_timesteps)
        for (epochs, gen_epochs, timesteps, noise_timesteps) in combinations)


def main(args):

    # gan = GAN_abstraction(args.model, args.timesteps, args.noise_timesteps)
    # training_data = gan.load_data(n_traj=args.n_traj, model=args.model, timesteps=args.timesteps)

    # gan.train(training_data=training_data, n_epochs=args.epochs, batch_size=args.batch_size,
    #           noise_timesteps=args.noise_timesteps, trajectories_timesteps=args.timesteps,
    #           gen_epochs=args.gen_epochs)
    # discriminator, generator = gan.load(rel_path=RESULTS)

    grid_search(args.model, args.n_traj, args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("-n", "--n_traj", default=256, type=int)
    parser.add_argument("-t", "--timesteps", default=64, type=int)
    parser.add_argument("--noise_timesteps", default=5, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--model", default="SIR", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--gen_epochs", default=1, type=int)

    main(args=parser.parse_args())