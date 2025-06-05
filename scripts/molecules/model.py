# Modified from https://github.com/aksub99/molecular-vae/tree/master

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import gzip
import h5py
import argparse
import os
import h5py
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import model_selection

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, representation, values):
        'Initialization'
        self.lines = representation
        self.values = values

  def __len__(self):
        'Denotes the total number of samples'
        return self.values.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        y = y_train[index,:]
        x = x_train[index,:]
        return x, y
      
    
def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss

class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


def chunk_iterator(dataset, chunk_size=10):
    # Split the indices into chunks
    chunk_indices = np.array_split(np.arange(len(dataset)), len(dataset) // chunk_size)
    for chunk_ixs in chunk_indices:
        chunk = dataset[chunk_ixs]
        yield (chunk_ixs, chunk)

def one_hot_index(vec, charset):
    return map(charset.index, vec)

def decode_smiles_from_indexes(vec, charset):
    # Ensure that each element in 'vec' is a string (not numpy.bytes_)
    return "".join(map(lambda x: str(charset[x], 'utf-8') if isinstance(charset[x], bytes) else charset[x], vec)).strip()

def load_dataset(filename, split = True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)


def load_dataset_chunked(filename, split=True, batch_size=10000):
    # Open the HDF5 file explicitly
    import h5py
    h5f = h5py.File(filename, 'r')

    # Memory-mapping the data (this avoids loading the entire dataset into memory at once)
    data_test = np.array(h5f['data_test'], dtype='float32', copy=False)
    
    # Handle charset as strings directly
    charset = h5f['charset']
    if charset.dtype.kind in {'S', 'O'}:  # If it's a string or object type
        charset = [x.decode('utf-8') if isinstance(x, bytes) else x for x in charset]  # Decode bytes if needed
    else:
        charset = np.array(charset, dtype='float32', copy=False)
    
    if split:
        # Instead of loading the entire data_train, we'll iterate in chunks
        data_train = h5f['data_train']
        total_samples = data_train.shape[0]
        
        # Define the generator that reads data in chunks
        def data_batch_generator():
            """Generator to load data in batches."""
            for i in range(0, total_samples, batch_size):
                batch = data_train[i:i+batch_size]  # Read a batch from disk
                yield batch

        # Return the generator, data_test, and charset
        return (data_batch_generator(), data_test, charset)
    else:
        # If not splitting, return data_test and charset only
        return (data_test, charset)
    
    # Don't forget to close the file manually when done
    h5f.close()
