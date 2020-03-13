import argparse
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv1D, LeakyReLU, Dropout, Concatenate, \
                                    Embedding, Flatten, Reshape, RepeatVector, Permute, \
                                    SeparableConv1D, Lambda
from tensorflow.keras.models import Sequential, Model
import numpy as np
import tensorflow as tf
from utils import load_from_pickle, execution_time, generate_noise
import time
from directories import *
import os
import itertools


class GAN_abstraction:

    def __init__(self, model, timesteps, noise_timesteps, fixed_params, embed, lr):
        self.model = model
        self.timesteps = timesteps
        self.noise_timesteps = noise_timesteps
        self.embed=embed
        self.fixed_params=fixed_params
        self.lr=lr
        self.round_traj=True

    def load_data(self, n_traj, model, timesteps, path="../../SSA/data/train/"):
        if model == "SIR":
            filename = "SIR_training_set"
        elif model == "eSIR":
            filename = "eSIR_training_set"
        elif model == "Repress":
            filename = "Repressilator_training_set"
        elif model == "Toggle":
            filename = "ToggleSwitch_training_set"

        if self.fixed_params:
            filename = filename+"_oneparam"

        traj_simulations = load_from_pickle(path=path+filename+".pickle")
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

        # [print(initial_states[i], params[i]) for i in range(50)]
        # exit()

        trajectories = tf.keras.utils.normalize(trajectories, axis=-1, order=2)
        initial_states = tf.keras.utils.normalize(initial_states, axis=-1, order=2)
        params = tf.keras.utils.normalize(params, axis=-1, order=2)

        return trajectories, initial_states, params


    def generator(self, timesteps, noise_timesteps):

        noise = Input(shape=(noise_timesteps, self.n_species)) 
        init_states = Input(shape=(1,self.n_species))
        par = Input(shape=(self.n_params,))
        full_traj = Concatenate(axis=1)([init_states,noise])

        if self.fixed_params:
            inputs = full_traj

        else:
            if self.embed:
                params = Reshape((noise_timesteps+1,1))(Dense((noise_timesteps+1))(par))
            else:
                params = Permute((1,2))(RepeatVector(noise_timesteps+1)(par))
            inputs = Concatenate(axis=-1)([full_traj,params])

        # x = Conv1D(64, 3, padding="same")(inputs)
        # x = LeakyReLU()(x)
        # x = Conv1D(128, 3)(x)
        # x = LeakyReLU()(x)
        # x = Flatten()(x)
        # x = Dense((timesteps)*(self.n_species), activation="relu")(x)
        # outputs = Reshape((timesteps, self.n_species))(x)        

        n_channels = inputs.shape[2]
        inputs_list = []
        for c in range(n_channels):
            F = Lambda(lambda w: w[:,:,c])
            x = F(inputs)
            x = Reshape((inputs.shape[1],1))(x)
            x = Conv1D(64, 3, padding="same")(x)
            x = LeakyReLU()(x)
            x = Conv1D(128, 3)(x)
            x = LeakyReLU()(x)
            x = Flatten()(x)
            x = Dense((timesteps), activation="relu")(x)
            inputs_list.append(x)
        outputs_list = Concatenate(axis=-1)(inputs_list)
        outputs = Reshape((timesteps, n_channels))(outputs_list)

        if self.fixed_params is False:
            model = Model(inputs=[noise,init_states,par], outputs=outputs)
        else:
            model = Model(inputs=[noise,init_states], outputs=outputs)
        
        return model

    def discriminator(self, timesteps):

        traj = Input(shape=(timesteps, self.n_species)) 
        init_states = Input(shape=(1,self.n_species))
        par = Input(shape=(self.n_params,))
        full_traj = Concatenate(axis=1)([init_states,traj])

        if self.fixed_params is False:

            if self.embed:
                params = Reshape((timesteps+1,1))(Dense((timesteps+1))(par))
            else:
                params = Permute((1,2))(RepeatVector(timesteps+1)(par))

            inputs = Concatenate(axis=-1)([full_traj,params])

        else:
            inputs = full_traj
        
        # x = Conv1D(64, 2)(inputs)
        # x = LeakyReLU()(x)
        # x = Conv1D(128, 3)(x)
        # x = LeakyReLU()(x)
        # x = Flatten()(x)
        # x = Dropout(0.3)(x)
        # outputs = Dense(1, activation="sigmoid")(x)

        n_channels = inputs.shape[2]
        inputs_list = []
        for c in range(n_channels):
            F = Lambda(lambda w: w[:,:,c])
            x = F(inputs)
            x = Conv1D(64, 2)(inputs)
            x = LeakyReLU()(x)
            # x = Conv1D(128, 3)(x)
            # x = LeakyReLU()(x)
            x = Flatten()(x)
            x = Dropout(0.3)(x)
            inputs_list.append(x)
        outputs_list = Concatenate(axis=-1)(inputs_list)
        outputs = Dense(1, activation="sigmoid")(x)

        if self.fixed_params is False:
            model = Model(inputs=[traj,init_states,par], outputs=outputs)
        else:
            model = Model(inputs=[traj,init_states], outputs=outputs)

        if model.trainable == False:
            if self.round_traj:
                traj = K.round(traj)
            
            if self.fixed_params is False:
                model = Model(inputs=[traj,init_states,par], outputs=outputs)
            else:
                model = Model(inputs=[traj,init_states], outputs=outputs)

        opt = keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def gan(self, discriminator, generator): 
        discriminator.trainable = False

        if self.fixed_params is False:
            noise, init_states, par = generator.input
        else:
            noise, init_states = generator.input
        gen_traj = generator.output

        if self.fixed_params is False:
            gan_output = discriminator([gen_traj, init_states, par])
            model = Model(inputs=[noise, init_states, par], outputs=gan_output)
        else:
            gan_output = discriminator([gen_traj, init_states])
            model = Model(inputs=[noise, init_states], outputs=gan_output)

        opt = keras.optimizers.Adam(lr=self.lr)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def train(self, training_data, n_epochs, gen_epochs, batch_size, timesteps, noise_timesteps):
        trajectories, initial_states, params = training_data 
        n_batches = int(len(initial_states) / batch_size)+1

        generator = self.generator(timesteps=timesteps, noise_timesteps=noise_timesteps)
        discriminator = self.discriminator(timesteps=timesteps)
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
            epoch_start = time.time()

            perm_idxs = np.random.permutation(len(trajectories))
            trajectories = trajectories[perm_idxs]
            initial_states = initial_states[perm_idxs]
            params = params[perm_idxs]
            
            for batch_idx in range(n_batches-1):
                begin, end = batch_idx*batch_size, (batch_idx+1)*batch_size
                traj = trajectories[begin:end,:,:]
                init_states = initial_states[begin:end,:]
                par = params[begin:end,:]

                y_train_real = np.ones(len(init_states))
                d_loss1, d_acc1 = discriminator.train_on_batch([traj, init_states, par], 
                                  y_train_real)

                noise = generate_noise(len(init_states), noise_timesteps, self.n_species)
                gen_traj = generator.predict([noise, init_states, par])

                ## debug
                # print(initial_states[:3])
                # print(gen_traj[0][:2][:2])

                if self.round_traj:
                    gen_traj = np.round(gen_traj)

                    # print(gen_traj[0][:2][:2])

                y_train_fake = np.zeros(len(init_states))
                d_loss2, d_acc2 = discriminator.train_on_batch([gen_traj, init_states, par], 
                                  y_train_fake)

                for _ in range(gen_epochs):
                    noise = generate_noise(len(init_states), noise_timesteps, self.n_species)
                    g_loss = gan.train_on_batch(x=[noise, init_states, par], y=y_train_real)

            d_loss1_list.append(d_loss1)
            d_loss2_list.append(d_loss2)
            g_loss_list.append(g_loss)
            d_acc1_list.append(d_acc1*100)
            d_acc2_list.append(d_acc2*100)   

            print(f"\n[Epoch {epoch + 1}]\t g_loss = {g_loss:.4f}", end="\t")
            print(f"d_loss1 = {d_loss1:.4f}\td_loss2 = {d_loss2:.4f}", end="\t")
            print(f"a1 = {int(100*d_acc1)}\ta2 = {int(100*d_acc2)}", end="\t")
            print(f"time = {time.time()-epoch_start:.2f}", end="\t")

        print("\n")
        execution_time(start=start, end=time.time())

        os.makedirs(os.path.dirname(RESULTS+"gan_models/"), exist_ok=True)
        filename = self.model+"_t="+str(self.timesteps)+"_tNoise="+str(self.noise_timesteps)+\
                   "_ep="+str(n_epochs)+"_epG="+str(gen_epochs)+"_lr="+str(self.lr)

        print("\nSaving:")
        print(RESULTS+"gan_models/"+filename+"_discriminator.h5")
        print(RESULTS+"gan_models/"+filename+"_generator.h5")
        discriminator.save(RESULTS+"gan_models/"+filename+"_discriminator.h5")
        generator.save(RESULTS+"gan_models/"+filename+"_generator.h5")

        plot_training(n_epochs, g_loss_list, d_loss1_list, d_loss2_list, d_acc1_list, 
                      d_acc2_list, filename)

        return discriminator, generator

    def load(self, rel_path, n_epochs, gen_epochs):
        path = rel_path+"gan_models/"
        filename = self.model+"_t="+str(self.timesteps)+"_tNoise="+str(self.noise_timesteps)+\
                   "_ep="+str(n_epochs)+"_epG="+str(gen_epochs)+"_lr="+str(self.lr)
        discriminator = keras.models.load_model(path+filename+"_discriminator.h5")
        generator = keras.models.load_model(path+filename+"_generator.h5") 
        return discriminator, generator


def plot_training(n_epochs, g_loss, d_loss1, d_loss2, d_acc1, d_acc2, filename):

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,2,figsize=(12,6))

    sns.lineplot(range(1, n_epochs+1), g_loss, label='Generator loss', ax=ax[0])
    sns.lineplot(range(1, n_epochs+1), d_loss1, label='Discriminator loss real', ax=ax[0])
    sns.lineplot(range(1, n_epochs+1), d_loss2, label='Discriminator loss gen', ax=ax[0])
    sns.lineplot(range(1, n_epochs+1), d_acc1, label='Discriminator train acc real', ax=ax[1])
    sns.lineplot(range(1, n_epochs+1), d_acc2, label='Discriminator train acc gen', ax=ax[1])

    # ax[0].set_yscale('log')
    ax[0].set_ylabel("loss")
    ax[1].set_ylabel("accuracy")
    plt.tight_layout()

    os.makedirs(os.path.dirname(RESULTS+"gan_models/"), exist_ok=True)
    plt.savefig(RESULTS+"gan_models/"+filename+".png")


# === MAIN EXECUTIONS ===


def _parallel_grid_search(model, n_traj, batch_size, epochs, gen_epochs, timesteps, 
                          noise_timesteps, embed, fixed_params):

    gan = GAN_abstraction(model, timesteps, noise_timesteps, embed, fixed_params)
    training_data = gan.load_data(n_traj=n_traj, model=model, timesteps=timesteps)
    gan.train(training_data=training_data, n_epochs=epochs, batch_size=batch_size,
              noise_timesteps=noise_timesteps, timesteps=timesteps, gen_epochs=gen_epochs)


def grid_search(args):

    print("\n == Grid search training == ")
    from joblib import Parallel, delayed

    epochs_list = [80]
    gen_epochs_list = [5,10]
    noise_timesteps_list = [128]
    print("epochs_list =", epochs_list)
    print("gen_epochs_list =", gen_epochs_list)
    print("noise_timesteps_list", noise_timesteps_list)
    combinations = list(itertools.product(epochs_list, gen_epochs_list, noise_timesteps_list))

    Parallel(n_jobs=4)(
        delayed(_parallel_grid_search)(args.model, args.n_traj, args.batch_size, epochs, 
                                       gen_epochs, args.timesteps, noise_timesteps,
                                       args.embed, args.fixed_params)
        for (epochs, gen_epochs, noise_timesteps, embed, fixed_params) in combinations)


def full_gan_training(args):

    gan = GAN_abstraction(args.model, args.timesteps, args.noise_timesteps, lr=args.lr,
                          embed=args.embed, fixed_params=args.fixed_params)

    training_data = gan.load_data(n_traj=args.n_traj, model=args.model, timesteps=args.timesteps)

    gan.train(training_data=training_data, n_epochs=args.epochs, batch_size=args.batch_size,
              noise_timesteps=args.noise_timesteps, timesteps=args.timesteps,
              gen_epochs=args.gen_epochs)


def main(args):

    full_gan_training(args)
    # grid_search(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("-n", "--n_traj", default=1000, type=int)
    parser.add_argument("-t", "--timesteps", default=128, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--model", default="eSIR", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--gen_epochs", default=5, type=int)
    parser.add_argument("--noise_timesteps", default=5, type=int)
    parser.add_argument("--embed", default=False, type=bool)
    parser.add_argument("--fixed_params", default=True, type=bool)
    parser.add_argument("--lr", default="0.001", type=float)

    main(args=parser.parse_args())