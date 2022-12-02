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


os.chdir('/Users/stephaniewanko/Downloads/water_tracking/normalized_water/')
pt=1
length = 1000
band = '_25'
center_coords = np.load(f'center_coor_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
cutoff_idx = np.load(f'cutoff_idx_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
cutoff_idx_all = np.load(f'cutoff_idx_{length}_all.npy',allow_pickle='TRUE').item()
all_density_vals = np.load(f'density_vals_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
labels = np.load(f'labels_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
spread = np.load(f'spread_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
all_xyz_coords = np.load(f'all_xyz_coords_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
rel_b_list = np.load(f'rel_b_list_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
q_list = np.load(f'q_list_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()

os.chdir('/Users/stephaniewanko/Downloads/water_tracking/')
s = Structure.fromfile('7KQO.pdb').reorder()
s = s.extract("e", "H", "!=")


#GET KDE
#subset waters
os.chdir('/Users/stephaniewanko/Downloads/water_tracking/')
s_wat = Structure.fromfile('1cc7.pdb').reorder()
waters = s_wat.extract("resn", "HOH", "==")
for c in set(waters.chain):
        for r in set(waters.extract("chain", c, "==").resi):
            wat = waters.extract(f'chain {c} and resi {r}').coor
            dist = np.linalg.norm(out_coords_all_KDE.reshape(-1,3) - wat, axis=1)
            print(min(dist))
            print(np.exp(density_all[dist == min(dist)]))


# build_density_pdb(new_all_xyz_coords.reshape(-1,3), 
#                   '4IRX_121_density_75_1.0_2_lys.pdb', 
#                   np.exp(dens_v_all))


# build_center_placement_pdb(new_coords.reshape(-1,3),
#                            '7kq0_all.pdb', 'test.pml')

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



##DISTANCE FROM CENTER

out_coords, out_coords_all, out_coords_all_dens, sz_all = place_all_centers(s, all_coord_info,
                                                                       center_coords, 
                                                                       min_ang, 
                                                                       spread, 
                                                                       all_density_vals, 
                                                                       cont_dict,
                                                                       cutoff_idx,
                                                                       all_xyz_coords,
                                                                       rel_b_list,
                                                                       q_list,                                                                        use_cutoff=False
                                                                        )
os.chdir('/Users/stephaniewanko/Downloads/water_tracking/')
build_center_placement_pdb(out_coords.reshape(-1,3), 'all_water_1cc7.pdb', 
                           'test.pml')

#read file back in
os.chdir('/Users/stephaniewanko/Downloads/water_tracking/')
water_og = Structure.fromfile('1cc7.pdb').reorder()
water_og = water_og.extract('resn', 'HOH', '==')
water_new = Structure.fromfile('all_water_1cc7.pdb').reorder()
ratio, deltas = water_RMSD(water_og, water_new)
file_n = 'ratio_75_3.txt'
print(ratio)
with open(file_n, 'w') as file:
      file.write(str(ratio))
name = f'{args.directory1}/{args.pdb}_{args.pt}_{args.band}.jpg'
#plot_waters_detected(deltas, 'ratio_75_3.jpg')
