import pickle as pkl


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

