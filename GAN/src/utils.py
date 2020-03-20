import pickle as pkl
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def save_to_pickle(data, relative_path, filename):
    filepath = relative_path + filename
    print("\nSaving pickle: ", filepath)
    os.makedirs(os.path.dirname(relative_path), exist_ok=True)
    with open(filepath, 'wb') as f:
        pkl.dump(data, f)

def load_from_pickle(path):
    print("\nLoading from pickle: ",path)
    with open(path, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    return data

def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))

def generate_noise(batch_size, noise_timesteps, n_species, scale=1.):
    noise = np.random.normal(loc=0., scale=scale, size=(batch_size, noise_timesteps, n_species))
    return noise

def rescale(data, path, filename):
    # scaler = MinMaxScaler()
    # orig_shape = data.shape
    # data = data.reshape(data.shape[0],-1)
    # scaler.fit(data)
    # rescaled_data = scaler.transform(data)
    # rescaled_data = rescaled_data.reshape(orig_shape)

    mean = np.mean(data)
    std = np.std(data)
    rescaled_data = (data-mean)/std
    scaler = {"mean":mean,"std":std}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_to_pickle(scaler, path, filename+"_scaler.pkl")
    return rescaled_data

def load_and_rescale(data, path):
    scaler = load_from_pickle(path+"_scaler.pkl")
    
    # orig_shape = data.shape
    # data = data.reshape(data.shape[0],-1)
    # data = scaler.transform(data)   
    # rescaled_data = rescaled_data.reshape(orig_shape)

    rescaled_data = (data-scaler["mean"])/scaler["std"]

    return rescaled_data