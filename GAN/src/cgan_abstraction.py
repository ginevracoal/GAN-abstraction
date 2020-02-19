from keras.layers import Model, Input, Dense, Conv1D, LeakyReLU, Dropout, Concatenate, Flatten, Reshape
from keras.optimizers import Adam


class CGAN_abstraction():

    def __init__(self, n_species, n_params):
        self.n_species = n_species
        self.n_params = n_params

    def noise_matrix(self, noise_timesteps):
        pass

    def preprocess_data(self, timesteps, n_species, n_params):

        # todo: cut trajectories
        # todo: concatenate initial_state + trajectories as a 2d array of shape=(timesteps,n_species)
        # todo: build repeated params as a 2d array of shape=(timesteps,n_params)
        # todo: merge everything into a single 2d array of shape=(timesteps,n_species+n_params)

        pass

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

    def train(self):
        pass

    def evaluate(self):
        pass

# todo: define parser