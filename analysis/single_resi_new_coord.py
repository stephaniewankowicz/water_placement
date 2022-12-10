#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from analysis_functions import *	
from qfit.structure import Structure
import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from scipy.signal import argrelextrema
from scipy import stats
import sklearn
from sklearn.cluster import MeanShift
from sklearn.cluster import  estimate_bandwidth
from sklearn.model_selection  import LeaveOneOut
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
import itertools
from DICT4A import DICT4A
from DICT4A_ALLAT import DICT4A_ALLAT


def remove_overlapping_wat(out_coords_all_KDE, density_all):
    #remove overlapping waters
    #find the waters that are closest to each other
    dist = cdist(out_coords_all_KDE.reshape(-1,3), out_coords_all_KDE.reshape(-1,3))
    np.fill_diagonal(dist, np.inf)


    #find the waters that are closest to each other
    min_dist_idx = np.argmin(dist, axis=1)



os.chdir('/Users/stephaniewanko/Downloads/water_tracking/')
s = Structure.fromfile('/Users/stephaniewanko/Downloads/water_tracking/135l.pdb').reorder()
res = list(s.residues)[6]

print(res.resn)
print(res.resi)

os.chdir('/Users/stephaniewanko/Downloads/water_tracking/normalized_water/')
cont_dict = np.load('cont_dict.npy',allow_pickle='TRUE').item()
min_ang = np.load('min_ang.npy',allow_pickle='TRUE').item()
max_ang = np.load('max_ang.npy',allow_pickle='TRUE').item()
all_coord_info = np.load('dih_info.npy',allow_pickle='TRUE').item()

os.chdir('/Users/stephaniewanko/Downloads/water_tracking/water_norm/')
pt=75
length = 30000
band = '_2'
center_coords = np.load(f'center_coor_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
norm_val = np.load(f'norm_val_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
cutoff_idx = np.load(f'cutoff_idx_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
cutoff_idx_all = np.load(f'cutoff_idx_{length}_all.npy',allow_pickle='TRUE').item()
all_density_vals = np.load(f'density_vals_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
labels = np.load(f'labels_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
spread = np.load(f'spread_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
all_xyz_coords = np.load(f'all_xyz_coords_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
rel_b_list = np.load(f'rel_b_list_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
q_list = np.load(f'q_list_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
labels = np.load(f'labels_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()


new_center_coords, new_all_xyz_coords, dens_v_all, b_all, q_all, resi_norm, new_spread = get_new_coords_og(all_coord_info,
                                                                                       res, 
                                                                                       center_coords, 
                                                                                       min_ang, 
                                                                                       spread, 
                                                                                       all_density_vals, 
                                                                                       cont_dict,
                                                                                       cutoff_idx,
                                                                                       all_xyz_coords,
                                                                                       rel_b_list,
                                                                                       q_list, norm_val, s,
                                                                                       use_cutoff=False)
