import argparse
from tensorflow.keras.layers import Input, Dense, Conv1D, LeakyReLU, Dropout, Concatenate, Flatten, Reshape
from tensorflow.keras.models import Sequential, Model
import numpy as np
import tensorflow as tf
from utils import load_from_pickle, execution_time
import time


class GAN_abstraction:

    def __init__(self, model, timesteps, noise_timesteps):
        self.timesteps = timesteps
        self.noise_timesteps = noise_timesteps

    def load_data(self, n_traj, model, timesteps, path="../../SSA/data/"):
        if model == "SIR":
            filename = "SIR_training_set.pickle"
        if model == "eSIR":
            filename = "eSIR_training_set.pickle"

        traj_simulations = load_from_pickle(path=path+filename)
        # print("traj_simulations: ", [print(key,val.shape) for key,val in traj_simulations.items()])

        trajectories = traj_simulations["X"][:n_traj,:timesteps,:]
        initial_states = traj_simulations["Y_s0"][:n_traj]
        params = traj_simulations["Y_par"][:n_traj]

        self.n_species = initial_states.shape[1]
        self.n_params = params.shape[1]

        print("\ntrajectories.shape = ", trajectories.shape)
        print("initial_states.shape = ", initial_states.shape)
        print("params.shape = ", params.shape)
        print("n_species = ", self.n_species)
        print("n_params = ", self.n_params)

        return trajectories, initial_states, params

    def reshape_training_data(self, trajectories, initial_states, params):
        initial_states = np.expand_dims(initial_states, axis=1)
        #print(initial_states.shape, trajectories.shape)
        full_trajectories = np.concatenate((initial_states, trajectories), axis=1)
        rep_params = np.repeat(params, self.timesteps+1)
        rep_params = np.reshape(rep_params, (params.shape[0], self.timesteps+1, params.shape[1]))
        #print(full_trajectories.shape, rep_params.shape)
        x_train = np.concatenate((full_trajectories, rep_params), axis=2)   
        #print("\nx_train.shape =", x_train.shape)
        return x_train

    def generate_noise(self, batch_size, noise_timesteps):
        if noise_timesteps > self.timesteps:
            raise ValueError("noise_timesteps should be smaller than trajectories timesteps.")
        noise = np.random.rand(batch_size, noise_timesteps, self.n_species)
        return noise

    def generator(self, noise_timesteps, trajectories_timesteps):
        inputs = Input(shape=(noise_timesteps, self.n_species))
        x = Conv1D(64, 3)(inputs)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense((self.n_species+self.n_params)*(trajectories_timesteps+1), activation="relu")(x)
        outputs = Reshape((trajectories_timesteps+1,self.n_species+self.n_params))(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def discriminator(self, trajectories_timesteps):
        inputs = Input(shape=(trajectories_timesteps+1, self.n_species+self.n_params))
        x = Conv1D(64, 2)(inputs)
        x = LeakyReLU()(x)
        x = Conv1D(128, 3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer="adam")
        return model

    def gan(self, discriminator, generator):
        discriminator.trainable = False
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, training_data, n_epochs, batch_size, noise_timesteps, trajectories_timesteps):
        trajectories, initial_states, params = training_data 
        n_batches = int(len(initial_states) / batch_size)+1

        generator = self.generator(noise_timesteps=noise_timesteps, 
                                   trajectories_timesteps=trajectories_timesteps)
        discriminator = self.discriminator(trajectories_timesteps=trajectories_timesteps)
        gan = self.gan(discriminator, generator)
 
        start = time.time()
        for epoch in range(n_epochs):
            for batch_idx in range(1,n_batches):
                begin, end = batch_idx*batch_size, (batch_idx+1)*batch_size
                traj = trajectories[begin:end,:,:]
                init_states = initial_states[begin:end,:]
                par = params[begin:end,:]
                #print(traj.shape, initial_states.shape, params.shape, begin, end)
                x_train_real = self.reshape_training_data(traj, init_states, par)
                y_train_real = np.ones(len(init_states))
                d_loss1 = discriminator.train_on_batch(x_train_real, y_train_real)

                noise = self.generate_noise(len(init_states), noise_timesteps)
                x_train_fake = generator.predict(noise)
                y_train_fake = np.zeros(len(init_states))
                d_loss2 = discriminator.train_on_batch(x_train_fake, y_train_fake)

                noise = self.generate_noise(len(init_states), noise_timesteps)
                #fake_input = self.reshape_training_data(noise, initial_states, params)
                g_loss = gan.train_on_batch(noise, y_train_real)

            print(f"\n[Epoch {epoch + 1}]\t g_loss: {g_loss:.8f}\t d_loss: {d_loss1+d_loss2:.8f}", end="\t")

        print("\n")
        execution_time(start=start, end=time.time())

        discriminator.save("discriminator.h5")
        generator.save("generator.h5")
        return discriminator, generator



def main(args):

    gan = GAN_abstraction(args.model, args.timesteps, args.noise_timesteps)
    training_data = gan.load_data(n_traj=args.n_traj, model=args.model, timesteps=args.timesteps)
    gan.train(training_data=training_data, n_epochs=args.epochs, batch_size=args.batch_size,
              noise_timesteps=args.noise_timesteps, trajectories_timesteps=args.timesteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("-n", "--n_traj", default=500, type=int)
    parser.add_argument("-t", "--timesteps", default=100, type=int)
    parser.add_argument("--noise_timesteps", default=3, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--model", default="SIR", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.002, type=float)

    main(args=parser.parse_args())