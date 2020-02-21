import argparse
from keras.layers import Model, Input, Dense, Conv1D, LeakyReLU, Dropout, Concatenate, Flatten, Reshape
from keras.models import Sequential
import numpy as np


class CGAN_abstraction:

    def __init__(self, n_species, n_params, timesteps, noise_timesteps):
        self.n_species = n_species
        self.n_params = n_params
        self.timesteps = timesteps
        self.noise_timesteps = noise_timesteps

    def load_data(self, timesteps, dir, path="../SSA/data/"):
        # dir = name of the model
        # load from pickle:
        # - X = trajectories of length timesteps-1
        # - y_S0 = initial state
        # - y_par = reactions params
        # print shapes
        if timesteps > trajectories.shape[1]:
            raise ValueError()
        # cut trajectories at timesteps length
        return trajectories, initial_state, params

    def reshape_training_data(trajectories, initial_state, params):
        # concatenate initial_state + trajectories as a 2d array of shape=(timesteps,n_species)
        # build repeated params as a 2d array of shape=(timesteps,n_params)
        # merge everything into a single 2d array of shape=(timesteps,n_species+n_params)   
        return x_train

    def generate_noise(self, batch_size, noise_timesteps):
        if noise_timesteps > self.timesteps:
            raise ValueError("noise_timesteps should be smaller than trajectories timesteps.")
        noise = np.random.rand(shape=(self.n_species,noise_timesteps-1))
        return noise

    def generator(self, noise_timesteps, trajectories_timesteps):
        x = Input(shape=(noise_timesteps, self.n_species+self.n_params))
        x = Conv1D(64, (3,3))(x)
        x = LeakyReLU()(x)
        x = Conv1D(128, (3,3))(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(self.n_species*trajectories_timesteps, activation="relu")(x)
        x = Reshape((self.n_species,trajectories_timesteps))(x)
        model = Model(x)
        model.compile(loss="binary_crossentropy", optimizer="adam")
        return model

    def discriminator(self, n_species, trajectories_timesteps):
        x = Input(shape=(n_species, trajectories_timesteps))
        x = Conv1D(64, (3,3))(x)
        x = LeakyReLU()(x)
        x = Conv1D(128, (3,3))(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dropout(0.3)(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(x)
        model.compile(loss='binary_crossentropy', optimizer="adam")
        return model

    def gan(self, discriminator, generator):
        discriminator.trainable = False
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, trajectories, initial_state, params, n_epochs, batch_size):
        n_batches = int(x_train.shape[0] / batch_size)

        generator = self.generator(noise_timesteps, trajectories_timesteps)
        discriminator = self.discriminator(n_species, trajectories_timesteps)
        gan = self.gan(discriminator, generator)
 
        for i in range(n_epochs):
            for j in range(n_batches):
                trajectories = trajectories[j,j+batch_size]
                initial_state = initial_state[j,j+batch_size]

                x_train_real = reshape_training_data(trajectories, initial_state, params)
                d_loss1, _ = discriminator.train_on_batch(x_train_real, 1)

                noise = generate_noise(batch_size, noise_timesteps)
                x_train_fake = generator.predict(noise)
                d_loss2, _ = discriminator.train_on_batch(x_train_fake, 0)

                noise = generate_noise(batch_size, noise_timesteps)
                fake_input = reshape_training_data(noise, initial_state, params)
                g_loss = gan.train_on_batch(fake_input, 1)

        discriminator.save("discriminator.h5")
        generator.save("generator.h5")
        return discriminator, generator

    def evaluate(self):
        pass


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN.")
    parser.add_argument("-t", "--timesteps", nargs="?", default=100, type=int)
    parser.add_argument("--model", nargs='?', default="SIR", type=str)
    parser.add_argument("--epochs", nargs='?', default=5, type=int)
    parser.add_argument("--lr", nargs='?', default=0.002, type=float)

    main(args=parser.parse_args())