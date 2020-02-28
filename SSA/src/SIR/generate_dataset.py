import numpy as np
from numpy.random import randint, random
import stochpy
import pandas as pd
import os
import shutil
from tqdm import tqdm
import pickle


class AbstractionDataset(object):

    def __init__(self, n_init_states, n_params, n_trajs, state_space_bounds, param_space_bounds, model_name, time_step, T):
        # state_space_bounds : shape = (state_space_dim,2)
        # param_space_bounds : shape = (param_space_dim,2)
        self.n_init_states = n_init_states
        self.n_params = n_params
        self.n_trajs = n_trajs
        self.n_training_points = n_init_states*n_params*n_trajs
        self.state_space_bounds = state_space_bounds
        self.param_space_bounds = param_space_bounds
        self.state_space_dim = state_space_bounds.shape[0]
        self.param_space_dim = param_space_bounds.shape[0]
        self.stoch_mod = stochpy.SSA(IsInteractive=False)
        self.stoch_mod.Model(model_name+'.psc')
        self.directory_name = model_name
        self.time_step = time_step
        self.T = T # end time
        self.N = None # population size

    def set_popul_size(self, N):
        self.N = N


    def time_resampling(self, data):
        time_index = 0
        # Il nuovo array dei tempi
        time_array = np.linspace(0, self.T, num=int(self.T / self.time_step+1))
        # new_data conterrà i dati con la nuova scansione temporale
        # la prima colonna contiene gli istanti di tempo, e quindi corrisponde a time_array
        new_data = np.zeros((time_array.shape[0], data.shape[1]))
        new_data[:, 0] = time_array
        for j in range(len(time_array)):
            while time_index < data.shape[0] - 1 and data[time_index + 1][0] < time_array[j]:
                time_index = time_index + 1
            if time_index == data.shape[0] - 1:
                new_data[j, 1:] = data[time_index, 1:]
            else:
                new_data[j, 1:] = data[time_index, 1:]
        return new_data


    def set_initial_states(self, init_state):
        S = int(init_state[0])
        I = int(init_state[1])
        R = int(init_state[2])
        self.stoch_mod.ChangeInitialSpeciesCopyNumber("S", S)
        self.stoch_mod.ChangeInitialSpeciesCopyNumber("I", I)
        self.stoch_mod.ChangeInitialSpeciesCopyNumber("R", R)


    def set_parameters(self, params):
        self.stoch_mod.ChangeParameter("beta", params[0])
        self.stoch_mod.ChangeParameter("gamma", params[1])
        
    def sample_initial_states(self, n_points=None):
        if n_points == None:
            n_points = self.n_init_states

        susc = randint(low=0, high=self.N, size=(n_points,1))
        inf =  np.zeros((n_points,1))
        for i in range(n_points):
            inf[i] = randint(low=0, high=self.N-susc[i], size=1)
        rec = self.N-susc-inf
        set_of_init_states = np.hstack((susc,inf,rec))
            
        return set_of_init_states


    def sample_parameters_settings(self, n_points=None):
        if n_points == None:
            n_points = self.n_params

        set_of_params = np.zeros((n_points, self.param_space_dim))
        for i in range(self.param_space_dim):
            set_of_params[:,i] = (self.param_space_bounds[i,1] - self.param_space_bounds[i,0])*random(size=(n_points,))+self.param_space_bounds[i,0]

        return set_of_params


    def generate_training_set(self):

        Yp = np.zeros((self.n_training_points,self.param_space_dim))
        Ys = np.zeros((self.n_training_points,self.state_space_dim))

        X = np.zeros((self.n_training_points, int(self.T/self.time_step), self.state_space_dim))


        set_of_params = self.sample_parameters_settings()
        initial_states = self.sample_initial_states()
        
        count = 0
        for p in range(self.n_params):
            self.set_parameters(set_of_params[p,:])
            for i in tqdm(range(self.n_init_states)): 
                self.set_initial_states(initial_states[i,:])

                for k in range(self.n_trajs):
                    self.stoch_mod.DoStochSim(method="Direct", trajectories=self.n_trajs, mode="time", end=self.T)
                
                    self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

                    datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).drop("N",axis = 1).as_matrix()
                    
                    new_datapoint = self.time_resampling(datapoint)
                    X[count,:,:] = new_datapoint[1:,1:self.state_space_dim+1]
                    Yp[count,:] = set_of_params[p,:]
                    Ys[count,:] = initial_states[i,:self.state_space_dim]
                    
                    count += 1

        self.X = X
        self.Y_par = Yp
        self.Y_s0 = Ys
        self.Y = np.hstack((Yp,Ys))

    def generate_validation_set(self, n_val_points, n_val_trajs_per_point):

        Yp = np.zeros((n_val_points,self.param_space_dim))
        Ys = np.zeros((n_val_points,self.state_space_dim))

        X = np.zeros((n_val_points,  n_val_trajs_per_point,int(self.T/self.time_step), self.state_space_dim))


        set_of_params = self.sample_parameters_settings(n_val_points)
        initial_states = self.sample_initial_states(n_val_points)
            
        for ind in range(n_val_points):
            self.set_parameters(set_of_params[ind,:])
            self.set_initial_states(initial_states[ind,:])

            Yp[ind,:] = set_of_params[ind,:]
            Ys[ind,:] = initial_states[ind,:self.state_space_dim]
                
            for k in range(n_val_trajs_per_point):
                if k%100 == 0:
                    print(ind, "/", n_val_points, "------------------K iter: ", k, "/", n_val_trajs_per_point)
                
                self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
                self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

                datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).drop("N",axis = 1).as_matrix()
                
                new_datapoint = self.time_resampling(datapoint)
                X[ind,k,:,:] = new_datapoint[1:,1:self.state_space_dim+1]
            
        self.X = X
        self.Y_par = Yp
        self.Y_s0 = Ys
        self.Y = np.hstack((Yp,Ys))


    def save_dataset(self, filename):
        dataset_dict = {"X": self.X, "Y": self.Y, "Y_par": self.Y_par, "Y_s0": self.Y_s0}
        with open(filename, 'wb') as handle:
            pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    




def run_training():
    n_init_states = 200
    n_params = 50
    n_trajs = 3

    state_space_dim = 2
    param_space_dim = 2

    time_step = 0.1
    n_steps = 128
    T = n_steps*time_step

    N = 100

    param_space_bounds = np.array([[0.5,5],[0.5,3]])
    state_space_bounds = np.array([[0,N],[0,N]])

    sir_dataset = AbstractionDataset(n_init_states, n_params, n_trajs, state_space_bounds, param_space_bounds, 'SIR', time_step, T)
    sir_dataset.set_popul_size(N)

    sir_dataset.generate_training_set()
    sir_dataset.save_dataset("../../data/SIR/SIR_training_set.pickle")


def run_validation():
    n_init_states = 0
    n_params = 0
    n_trajs = 0

    state_space_dim = 2
    param_space_dim = 2

    time_step = 0.1
    n_steps = 128
    T = n_steps*time_step

    N = 100

    n_val_points = 20
    n_trajs_per_point = 2000

    param_space_bounds = np.array([[0.5,5],[0.5,3]])
    state_space_bounds = np.array([[0,N],[0,N]])

    sir_dataset = AbstractionDataset(n_init_states, n_params, n_trajs, state_space_bounds, param_space_bounds, 'SIR', time_step, T)
    sir_dataset.set_popul_size(N)

    sir_dataset.generate_validation_set(n_val_points, n_trajs_per_point)
    sir_dataset.save_dataset("../../data/SIR/SIR_validation_set.pickle")

run_training()

run_validation()