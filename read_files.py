# encoding: utf-8
import json
import h5py
import numpy as np
import sys
import shutil
if sys.version_info[0]==2:
    import cPickle as pickle
else:
    import pickle
import os

def create_folder(filename):
    a = '/'.join(filename.split('/')[:-1])
    if not os.path.exists(a):
        os.makedirs(a)



def savein_json(filename, array):
    create_folder(filename)
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
    create_folder(file)
    with open(file, 'wb') as handle:
        pickle.dump(array, handle)

def readfrom_pickle(file):
    with open(file, 'rb') as handle:
        if sys.version_info[0] == 2:
            data = pickle.load(handle)
        else:
            data = pickle.load(handle,encoding='latin1')
    return data

def readfrom_txt(path):
    data =open(path).read()
    return data

def textfile2list(path):
    data = readfrom_txt(path)
    txt_list =list()
    for line in data.splitlines():
        txt_list.append(line)
    return txt_list

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
    create_folder(file)
    f = h5py.File(filename+ ".hdf5", "w")
    data_size = len(labels)
    for index in range(data_size):
        f.create_dataset(labels[index], data=data[index], dtype=dtypes[index])

def movefiles(old_address,new_address,dir_simples,abbr):
    for dir_simple in dir_simples:
        desti = new_address+dir_simple +abbr
        shutil.copy(old_address+dir_simple+abbr,desti)

