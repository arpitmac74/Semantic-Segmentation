import os, glob, pickle
import numpy as np
from utils import *

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torchvision.models.segmentation
from unet.unet_model import UNet
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from evaluate import evaluate
import random

# =============================
# =========================================
# METHODS FOR MANAGING and LOADING DATA 
# ======================================================================
# 
# This script contains methods for:
# 
#   (1) Saving data summary into a data.pickle file which contains a 
#       dictionary containing study IDs, pp statistics and whether or not
#       that study is in the training or validation cohort. This dict
#       should be updated every time new data is added, prior to using 
#       the data.load() method.
# 
#   (2) Load random slice(s) of data from random studies in either training
#       or validation cohort.
# 
# ======================================================================


def create_summary(root='../data', valid_ratio=0.2):
    """
    Method to create a summary *.pickle file in the current directory
    containing summary dictionary for the dataset of form:

    summary = {
        'train': [
            {'studyid': 'study_00', 'mean': 0, 'sd': 0},
            {'studyid': 'study_01', 'mean': 0, 'sd': 0}, ...
        ],
        'valid': [
            {'studyid': 'study_02', 'mean': 0, 'sd': 0},
            {'studyid': 'study_03', 'mean': 0, 'sd': 0}, ...
        ]
    }

    """
    dfiles = glob.glob('%s/*/dat.npy' % root)
    summary = {'train': [], 'valid': []} 

    for n, dfile in enumerate(dfiles): 
        print('Saving summary %03i: %s' % (n + 1, dfile))
        dat = np.memmap(dfile, dtype='int16', mode='r')

        sid = os.path.basename(os.path.dirname(dfile))
        group = 'train' if np.random.rand() > valid_ratio else 'valid'
        summary[group].append({
            'studyid': sid,
            'mean': np.mean(dat[dat > 0]),
            'sd': np.std(dat[dat > 0])})

    pickle.dump(summary, open('%s/data.pickle' % root, 'wb'))

def load(mode='train', n=1, sid=None, z=None, return_mask=False,get = 0.0):
    """
    Method to open n random slices of data and corresponding labels. Note that this
    method will load data in a stratified manner such that approximately 50% of all 
    returned data will contain tumor.

    :params

      (str) mode : 'train' or 'valid'
      (int) n : number of examples to open
      (str) sid : if provided, will load specific study ID
      (int) z : if provided, will load specifc slice
      (bool) return_mask : if True, will also return mask containing brain parenchyma

    :return

      (np.array) dat : N x I x J x 1 input (dtype = 'float32')
      (np.array) lbl : N x I x J x 1 label (dtype = 'uint8')
      (np.array) msk : N x I x J x 1 lmask (dtype = 'float32'), (optional)

    """
    global root, summary
    random = True

    # --- Load specific slice
    if sid is not None and z is not None:
        for mode in ['train', 'valid']:
            indices = [n for n, stats in enumerate(summary[mode]) if stats['studyid'] == sid]
            if len(indices) > 0:
                indices = [indices[0]]
                random = False
                break

    # --- Load random n-slices
    if random:
        indices = np.random.randint(0, len(summary[mode]), n)

    dats = []
    lbls = []
    msks = []
    
    for ind in indices:
        stats = summary[mode][ind]
        
        # --- Load data and labels 
        fname = '%s/%s/dat.npy' % (root, stats['studyid'])
        # print(fname)
        dat = np.memmap(fname, dtype='int16', mode='r')
        
        xSize = 256
        ySize = 256
        
        n_ch = 1
        dat = dat.reshape(-1, xSize, ySize, n_ch) 

        fname = '%s/%s/lbl.npy' % (root, stats['studyid'])
        if os.path.exists(fname):
            lbl = np.memmap(fname, dtype='uint8', mode='r')
            lbl = lbl.reshape(-1, xSize, ySize, 1)
        else:
            lbl = np.ndarray((dat.shape))
        # --- Determine slice
        if random:
            reduce_sum = np.sum(lbl, axis=(1,2,3))
            z = np.nonzero(reduce_sum > 0)[0] if np.random.rand() > get else np.nonzero(reduce_sum == 0)[0]
            np.random.shuffle(z)
            z = z[0]
            
        msks.append((dat[z, ..., :1] > 0))
        dats.append((dat[z] - stats['mean']) / stats['sd'])
        lbls.append(lbl[z])

    dats = np.stack(dats, axis=0)
    lbls = np.stack(lbls, axis=0)
    msks = np.stack(msks, axis=0).astype('float32')

    if return_mask:
        return dats, lbls, msks

    else:
        return dats, lbls

def set_root(loc=None):
    """
    Method to set root directory location of data

    """
    global root, summary

    if loc is None:
        paths = [
            '/data/npy',      # AWS
            './data'             # Google CoLaboratory
        ]
        for path in paths:
            if os.path.exists(path):
                loc = path
                break

    root = loc
    summary_file = '%s/data.pickle' % root
    # print(root)
    if not os.path.exists(summary_file):
        create_summary(root=root)

    summary = pickle.load(open(summary_file, 'rb'))
    # print(summary)

# --- Set root and summary
global root, summary
set_root()
