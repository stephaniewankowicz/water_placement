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

os.chdir('/Users/stephaniewanko/Downloads/water_tracking/normalized_water/')
cont_dict = np.load('cont_dict.npy',allow_pickle='TRUE').item()
min_ang = np.load('min_ang.npy',allow_pickle='TRUE').item()
max_ang = np.load('max_ang.npy',allow_pickle='TRUE').item()
all_coord_info = np.load('dih_info.npy',allow_pickle='TRUE').item()

os.chdir('/Users/stephaniewanko/Downloads/water_tracking/norm/')
pt=75
length = 30000
band = '_2'
center_coords = np.load(f'center_coor_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
cutoff_idx = np.load(f'cutoff_idx_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
cutoff_idx_all = np.load(f'cutoff_idx_{length}_all.npy',allow_pickle='TRUE').item()
all_density_vals = np.load(f'density_vals_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
labels = np.load(f'labels_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
spread = np.load(f'spread_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
all_xyz_coords = np.load(f'all_xyz_coords_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
rel_b_list = np.load(f'rel_b_list_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
q_list = np.load(f'q_list_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()


def reassign_bfactors(s, out_coords_all_KDE, density_all, pdb_out):
    s = Structure.fromfile(s).reorder()
    bfactor_out = pd.DataFrame(columns=['resid','resname','bfactor'])
    s_wat = s.extract('resn', 'HOH', '==')
    for c in set(s_wat.chain):
        for r in set(s_wat.extract("chain", c, "==").resi):
            wat = s_wat.extract(f'chain {c} and resi {r}').coor
            dist = np.linalg.norm(out_coords_all_KDE.reshape(-1,3) - wat, axis=1)
            print(np.exp(density_all[dist == min(dist)]))
            bfactor_out = bfactor_out.concat({'resid':r,'resname':'HOH','bfactor':np.exp(density_all[dist == min(dist)])[0]},ignore_index=True)
            s.extract(f'chain {c} and resi {r}').b = np.exp(density_all[dist == min(dist)])
    s.tofile(f'{pdb_out}.pdb')
    bfactor_out.to_csv(f'{pdb_out}.csv',index=False)

#GET KDE
#subset waters
os.chdir('/Users/stephaniewanko/Downloads/water_tracking/')
s = Structure.fromfile('/Users/stephaniewanko/Downloads/water_tracking/1cc7.pdb').reorder()

out_coords, out_coords_all_KDE, out_coords_all_dens, sz_all, density_all = place_all_wat(all_coord_info,
                                                                       s, 
                                                                       center_coords, 
                                                                       min_ang, 
                                                                       spread, 
                                                                       all_density_vals, 
                                                                       cont_dict,
                                                                       cutoff_idx,
                                                                       all_xyz_coords,
                                                                       rel_b_list,
                                                                       q_list,
                                                                       use_cutoff=False
                                                                       )

reassign_bfactors('1cc7.pdb', out_coords_all_KDE, density_all, '1cc7_KDE_norm')  
