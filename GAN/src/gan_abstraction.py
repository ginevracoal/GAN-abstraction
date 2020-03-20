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


class GAN_abstraction:

    def __init__(self, model, timesteps, noise_timesteps, fixed_params, gen_lr, discr_lr, n_epochs, 
                 gen_epochs):
        self.model = model
        self.timesteps = timesteps
        self.noise_timesteps = noise_timesteps
        self.fixed_params=fixed_params
        self.n_epochs=n_epochs
        self.gen_epochs=gen_epochs

        self.architecture="conv2D"
        self.discr_noise=0
        self.batch_normalization=0

        ## set lr
        if discr_lr < gen_lr:
            raise ValueError("Discriminator lr should be > generator lr")
        self.gen_lr=gen_lr
        self.discr_lr=discr_lr

        ## set filename
        self.path = model+"_t="+str(timesteps)+"_tNoise="+str(noise_timesteps)+\
                    "_ep="+str(n_epochs)+"_epG="+str(gen_epochs)+\
                    "_lrD="+str(discr_lr)+"_lrG="+str(gen_lr)+"_"+str(self.architecture)#+"_sample"
        self.path = self.path+"_fixPar/" if self.fixed_params==1 else self.path+"/"

    def load_data(self, n_traj, model, timesteps, path="../../SSA/data/train/"):
        if model == "SIR":
            filename = "SIR_training_set"
        elif model == "eSIR":
            filename = "eSIR_training_set"
        elif model == "Repress":
            filename = "Repressilator_training_set"
        elif model == "Toggle":
            filename = "ToggleSwitch_training_set"

        if self.fixed_params==1:
            filename = filename+"_oneparam"

        traj_simulations = load_from_pickle(path=path+filename+".pickle")
        print("traj_simulations: ", [print(key,val.shape) for key,val in traj_simulations.items()])

        trajectories = traj_simulations["X"][:n_traj,:timesteps,:]
        initial_states = traj_simulations["Y_s0"][:n_traj]
        params = traj_simulations["Y_par"][:n_traj]
        initial_states = np.expand_dims(initial_states, axis=1)

        self.n_species = initial_states.shape[-1]
        self.n_params = params.shape[1]

        params = np.expand_dims(params, axis=-1)
        params = np.tile(params,(1,self.n_species))

        print("\ntrajectories.shape = ", trajectories.shape)
        print("initial_states.shape = ", initial_states.shape)
        print("params.shape = ", params.shape)
        print("n_species = ", self.n_species)
        print("n_params = ", self.n_params)
        print("noise_timesteps = ", self.noise_timesteps)

        return trajectories, initial_states, params

    def generator(self):

        noise = Input(shape=(self.noise_timesteps, self.n_species)) 
        init_states = Input(shape=(1,self.n_species))
        params = Input(shape=(self.n_params,self.n_species))

        if self.fixed_params==1:
            inputs = Concatenate(axis=1)([init_states,noise]) 
        else:
            inputs = Concatenate(axis=1)([init_states,noise,params])

        if self.batch_normalization==1:
            inputs = BatchNormalization()(inputs)   

        if self.architecture == "conv1D":
            x = Conv1D(64, 3)(inputs)
            x = LeakyReLU()(x)
            x = Conv1D(128, 3)(x)
            x = LeakyReLU()(x)
            x = Flatten()(x)
            x = Dense((self.timesteps)*(self.n_species), activation="relu")(x)
            outputs = Reshape((self.timesteps, self.n_species))(x)    

        elif self.architecture == "ch_conv1D":
            channels_outputs = []
            for c in range(self.n_species):
                select_channel = Lambda(lambda w: w[:,:,c])
                x = select_channel(inputs)
                x = Reshape((x.shape[1],1))(x)
                x = Conv1D(64, 3)(x)
                x = LeakyReLU()(x)
                x = Conv1D(128, 3)(x)
                x = LeakyReLU()(x)
                x = Flatten()(x)
                x = Dense((self.timesteps), activation="relu")(x)
                channels_outputs.append(x)
            channels_outputs = Concatenate(axis=-1)(channels_outputs)
            outputs = Reshape((self.timesteps, self.n_species))(channels_outputs)
        elif self.architecture == "conv2D":
            x = Reshape((inputs.shape[1],inputs.shape[2],1))(inputs)  
            x = Conv2D(64,(2,2),padding="same")(x)
            x = LeakyReLU()(x)
            x = Conv2D(128,(2,2))(x)
            x = LeakyReLU()(x)
            x = Flatten()(x)
            x = Dense((self.timesteps)*(self.n_species), activation="relu")(x)
            outputs = Reshape((self.timesteps, self.n_species))(x)   

        if self.fixed_params==1:
            model = Model(inputs=[noise,init_states], outputs=outputs)
        else:
            model = Model(inputs=[noise,init_states,params], outputs=outputs)        
        # print(model.summary())    
        return model

    def discriminator(self):

        trajectories = Input(shape=(self.timesteps, self.n_species)) 
        init_states = Input(shape=(1,self.n_species))
        params = Input(shape=(self.n_params,self.n_species))
        
        if self.fixed_params==1:
            inputs = Concatenate(axis=1)([init_states,trajectories]) 
        else:
            inputs = Concatenate(axis=1)([init_states,trajectories,params])

        if self.batch_normalization==1:
            inputs = BatchNormalization()(inputs)   

        if self.architecture == "conv1D":
            x = Conv1D(32, 3, data_format="channels_last")(inputs)
            x = LeakyReLU()(x)
            x = Flatten()(x)
            x = Dropout(0.3)(x)
            outputs = Dense(1, activation="sigmoid")(x)
        elif self.architecture == "ch_conv1D":
            channels_outputs = []
            for c in range(self.n_species):
                select_channel = Lambda(lambda w: w[:,:,c])
                x = select_channel(inputs)
                print(x.shape)
                x = Reshape((x.shape[1],1))(x)
                x = Conv1D(32, 3, data_format="channels_last")(x)
                x = LeakyReLU()(x)
                x = Flatten()(x)
                x = Dropout(0.3)(x)
                channels_outputs.append(x)
            outputs = Concatenate(axis=-1)(channels_outputs)
            outputs = Dense(1, activation="sigmoid")(outputs)   
        elif self.architecture == "conv2D":
            x = Reshape((inputs.shape[1],inputs.shape[2],1))(inputs)  
            x = Conv2D(32,(2,2),padding="same")(x)
            x = LeakyReLU()(x)
            x = Dropout(0.3)(x)
            x = Flatten()(x)
            outputs = Dense(1, activation="sigmoid")(x)

        if self.fixed_params==1:
            model = Model(inputs=[trajectories,init_states], outputs=outputs)
        else:
            model = Model(inputs=[trajectories,init_states,params], outputs=outputs)

        opt = keras.optimizers.Adam(lr=self.discr_lr)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        # print(model.summary())
        return model

    def gan(self, discriminator, generator): 

        discriminator.trainable = False

        if self.fixed_params==1:
            noise, init_states = generator.input
            gen_traj = generator.output
            gan_output = discriminator([gen_traj, init_states])
            model = Model(inputs=[noise, init_states], outputs=gan_output)

        else:
            noise, init_states, params = generator.input
            gen_traj = generator.output
            gan_output = discriminator([gen_traj, init_states, params])
            model = Model(inputs=[noise, init_states, params], outputs=gan_output)            

        opt = keras.optimizers.Adam(lr=self.gen_lr)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        # print(model.summary())
        return model

    def generate_real_samples(self, training_data, batch_size):
        trajectories, initial_states, params = training_data 
        idxs = np.random.randint(0, len(trajectories), batch_size)
        t, s, p = (trajectories[idxs], initial_states[idxs], params[idxs])
        return [t,s] if self.fixed_params==1 else [t,s,p]

    def generate_latent_samples(self, training_data, batch_size):
        _, initial_states, params = training_data 

        # noise
        noise_shape = (batch_size, self.noise_timesteps, self.n_species)
        noise = np.random.normal(loc=0., scale=1., size=noise_shape)

        # labels
        s_idxs = np.random.randint(0, len(initial_states), batch_size)
        p_idxs = np.random.randint(0, len(params), batch_size)
        s, p = (initial_states[s_idxs], params[p_idxs])

        # # campionamento da tutto il dataset
        # s = np.random.randint(0, 50, (batch_size,1,2))
        # p_idxs = np.random.randint(0, len(params), batch_size)
        # p = params[p_idxs]

        return [noise,s] if self.fixed_params==1 else [noise,s,p]

    def generate_fake_samples(self, training_data, batch_size, generator):
        latent_data = self.generate_latent_samples(training_data, batch_size)
        gen_traj = generator.predict(latent_data)
        gen_traj = np.round(gen_traj)  

        if self.fixed_params==1:
            _, s = latent_data
            return [gen_traj,s]
        else:
            _, s, p = latent_data
            return [gen_traj,s,p]

    def train(self, training_data, batch_size):

        generator = self.generator()
        discriminator = self.discriminator()
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

        n_batches = int(len(training_data[0])/batch_size)
        half_batch = int(batch_size*0.5)

        start = time.time()
        for epoch in range(self.n_epochs):
            for _ in range(n_batches):

                # == d1 training ==

                # if self.discr_noise==1:
                #     discr_noise = generate_noise(batch_size, self.noise_timesteps, self.n_species, 
                #                                  scale=0.01)
                #     t = t+discr_noise

                x_train_real = self.generate_real_samples(training_data, half_batch)
                y_train_real = np.ones(half_batch)
                d_loss1, d_acc1 = discriminator.train_on_batch(x_train_real, y_train_real)

                # == d2 training ==
                # noise = generate_noise(batch_size, self.noise_timesteps, self.n_species)
                # if self.discr_noise==1:
                #     noise = noise+discr_noise

                x_train_fake = self.generate_fake_samples(training_data, half_batch, generator)
                y_train_fake = np.zeros(half_batch)
                d_loss2, d_acc2 = discriminator.train_on_batch(x_train_fake, y_train_fake)

                # == g training ==
                for _ in range(self.gen_epochs):
                    x_latent = self.generate_latent_samples(training_data, batch_size)            
                    y_train_real = np.ones(batch_size)
                    g_loss = gan.train_on_batch(x=x_latent, y=y_train_real)

            d_loss1_list.append(d_loss1)
            d_loss2_list.append(d_loss2)
            g_loss_list.append(g_loss)
            d_acc1_list.append(d_acc1*100)
            d_acc2_list.append(d_acc2*100)   

            print(f"\n[Epoch {epoch + 1}]\t g_loss = {g_loss:.4f}", end="\t")
            print(f"d_loss1 = {d_loss1:.4f}\td_loss2 = {d_loss2:.4f}", end="\t")
            print(f"a1 = {int(100*d_acc1)}\ta2 = {int(100*d_acc2)}", end="\t")

        print("\n")
        execution_time(start=start, end=time.time())

        path = RESULTS+self.path+"trained_model/"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("\nSaving:"+"\n"+path+"discriminator.h5"+"\n"+path+"generator.h5")
        discriminator.save(path+"discriminator.h5")
        generator.save(path+"generator.h5")

        self.plot_training(g_loss_list, d_loss1_list, d_loss2_list, d_acc1_list, d_acc2_list)

        return discriminator, generator

    def load(self, rel_path):
        path = rel_path+self.path+"trained_model/"
        discriminator = keras.models.load_model(path+"discriminator.h5")
        generator = keras.models.load_model(path+"generator.h5") 
        return discriminator, generator


    def plot_training(self, g_loss, d_loss1, d_loss2, d_acc1, d_acc2):

        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1,2,figsize=(12,6))

        n_epochs=self.n_epochs
        sns.lineplot(range(1, n_epochs+1), g_loss, label='Generator loss', ax=ax[0])
        sns.lineplot(range(1, n_epochs+1), d_loss1, label='Discriminator loss real', ax=ax[0])
        sns.lineplot(range(1, n_epochs+1), d_loss2, label='Discriminator loss gen', ax=ax[0])
        sns.lineplot(range(1, n_epochs+1), d_acc1, label='Discriminator train acc real', ax=ax[1])
        sns.lineplot(range(1, n_epochs+1), d_acc2, label='Discriminator train acc gen', ax=ax[1])

        ax[0].set_ylabel("loss")
        ax[1].set_ylabel("accuracy")
        plt.tight_layout()

        path=RESULTS+self.path+"trained_model/"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path+"training.png")


# === MAIN EXECUTIONS ===


def _parallel_grid_search(model, n_traj, batch_size, epochs, gen_epochs, timesteps, 
                          noise_timesteps, fixed_params, lr):

    gan = GAN_abstraction(model=model,timesteps=timesteps, noise_timesteps=noise_timesteps, lr=lr, 
                          n_epochs=epochs, gen_epochs=args.gen_epochs, fixed_params=fixed_params)
    training_data = gan.load_data(n_traj=n_traj, model=model, timesteps=timesteps)
    gan.train(training_data=training_data,batch_size=batch_size)


# def grid_search(args):

#     print("\n == Grid search training == ")
#     from joblib import Parallel, delayed

#     epochs_list = [80]
#     gen_epochs_list = [5,10]
#     noise_timesteps_list = [128]
#     print("epochs_list =", epochs_list)
#     print("gen_epochs_list =", gen_epochs_list)
#     print("noise_timesteps_list", noise_timesteps_list)
#     combinations = list(itertools.product(epochs_list, gen_epochs_list, noise_timesteps_list))

#     Parallel(n_jobs=4)(
#         delayed(_parallel_grid_search)(args.model, args.traj, args.batch_size, epochs, 
#                                        gen_epochs, args.timesteps, noise_timesteps, args.fixed_params)
#         for (epochs, gen_epochs, noise_timesteps, fixed_params) in combinations)


def full_gan_training(args):

    gan = GAN_abstraction(model=args.model, fixed_params=args.fixed_params,
                          timesteps=args.timesteps, noise_timesteps=args.noise_timesteps, 
                          discr_lr=args.discr_lr, gen_lr=args.gen_lr,
                          n_epochs=args.epochs, gen_epochs=args.gen_epochs)

    training_data = gan.load_data(n_traj=args.traj, model=args.model, timesteps=args.timesteps)

    gan.train(training_data=training_data,  batch_size=args.batch_size)


def main(args):

    full_gan_training(args)
    # grid_search(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("--model", default="eSIR", type=str)
    parser.add_argument("--traj", default=1000, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--timesteps", default=128, type=int)
    parser.add_argument("--noise_timesteps", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--gen_epochs", default=10, type=int)
    parser.add_argument("--fixed_params", default=1, type=int)
    parser.add_argument("--gen_lr", default="0.0001", type=float)
    parser.add_argument("--discr_lr", default="0.0001", type=float)

    main(args=parser.parse_args())