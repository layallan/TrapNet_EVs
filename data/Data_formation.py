import numpy as np
import h5py
from os import listdir
from os.path import isfile, join, isdir
import scipy.io as sio

def data_from_dir(path):
    print(f'Loading--{path}...')
    files = [f for f in listdir(path) if isfile(join(path, f))]
    data = []
    temp_data = []
    length_per_channel = []
    
    for file in files:
        if file.startswith('.'):
            continue
        # Check file extension and use appropriate loader
        file_path = join(path, file)
        if file.endswith('.mat'):
            try:
                # Try loading with scipy.io.loadmat
                temp = sio.loadmat(file_path)
            except NotImplementedError:
                # If scipy fails, try using h5py
                with h5py.File(file_path, 'r') as f:
                    # Assuming the key for data in the .mat file is known and consistent
                    keys = list(f.keys())
                    # Typical key in HDF5 MAT-files is '#refs#'
                    if '#refs#' in keys:
                        first_key = list(f['#refs#'].keys())[0]
                        temp = np.array(f['#refs#'][first_key])
                    else:
                        key = keys[-1]
                        temp = np.array(f[key])

            temp_data.append(temp.flatten())
            length_per_channel.append(len(temp))
    
    length_stand = min(length_per_channel)
    for temp in temp_data:
        data.append(temp[:length_stand])
    data = np.column_stack(data)
    print(f'Loaded--{path}')
    return data, length_stand
