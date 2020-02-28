import pickle as pkl
import numpy as np


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

def generate_noise(batch_size, noise_timesteps, n_species):
    noise = np.random.rand(batch_size, noise_timesteps, n_species)
    return noise