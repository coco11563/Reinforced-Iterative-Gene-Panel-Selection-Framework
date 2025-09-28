import os
import numpy as np
import pandas as pd
import h5py

def load(task_name):
    data_dir = r"data/"
    hdf_path=f"{data_dir}/{task_name}.hdf"
    h5_path=f"{data_dir}/{task_name}.h5"
    if os.path.exists(hdf_path):
        return load_hdf(hdf_path)
    elif os.path.exists(h5_path):
        return load_h5(h5_path)
    else:
        print("can not find dataset!")
        raise TabError

def load_hdf(path):
    df=pd.read_hdf(path)
    n_cell=df.shape[0]
    X=df.iloc[:,:-1].astype('float')
    X=X.reset_index(drop=True)
    X.columns=range(X.shape[1])
    y=df.iloc[:,-1].to_numpy(dtype=np.float64).reshape(-1,1)
    return X,y

def load_h5(path):
    data=h5py.File(path)
    X=np.array(data['X'])
    y=np.array(data['Y'])
    return X,y