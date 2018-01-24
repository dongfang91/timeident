import json
import h5py
import numpy as np
import csv
import sys
if sys.version_info[0]==2:
    import cPickle as pickle
else:
    import pickle


def savein_json(filename, array):
    with open(filename+'.txt', 'w') as outfile:
        json.dump(array, outfile)
    print("Save into files: ",filename)
    outfile.close()

def readfrom_json(filename):
    with open(filename+'.txt', 'r') as outfile:
        data = json.load(outfile)
    outfile.close()
    return data

def savein_pickle(file,array):
    with open(file, 'wb') as handle:
        pickle.dump(array, handle)

def readfrom_pickle(file):
    with open(file, 'rb') as handle:
        if sys.version_info[0] == 2:
            b = pickle.load(handle)
        else:
            b = pickle.load(handle,encoding='latin1')
    return b

def readfrom_txt(path):
    data =open(path).read()
    return data

def textfile2list(path):
    data = readfrom_txt(path)
    list_new =list()
    for line in data.splitlines():
        list_new.append(line)
    return list_new



def load_hdf5(filename,labels):
    data = list()
    with h5py.File(filename + '.hdf5', 'r') as hf:
        print("List of datum in this file: ", hf.keys())
        for label in labels:
            x = hf.get(label)
            x_data = np.array(x)
            del x
            print("The shape of datum "+ label +": ",x_data.shape)
            data.append(x_data)
    return data

def save_hdf5(filename,labels,data,dtypes):
    f = h5py.File(filename+ ".hdf5", "w")
    data_size = len(labels)
    for index in range(data_size):
        f.create_dataset(labels[index], data=data[index], dtype=dtypes[index])




