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
from DICT4A import DICT4A
from DICT4A_ALLAT import DICT4A_ALLAT
import time

def parrallel_score_samples(kde, samples, thread_count=int(16)):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))


def structure_based_KDE(coords_all, norm_factor, s):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5, atol=0.0005,rtol=0.001).fit(coords_all,sample_weight = norm_factor)
    s_wat = s.extract('resn', 'HOH', '==')
    new_water = s_wat.coor
    #density = parrallel_score_samples(kde, coords_all, 16)
    #normalize by the number of protein 4 coor (sep by atom types & dihedrals) we have in our set/total number of proteins 
    #print('parrallel')
    density = kde.score_samples(new_water) #* norm_factor
    #print(density)
    return density

def reassign_bfactors(s, out_coords_all_KDE, density_all, pdb_out):
    #s = Structure.fromfile(s).reorder()
    bfactor_out = pd.DataFrame(columns=['resid','resname','bfactor'])
    s_wat = s.extract('resn', 'HOH', '==')
    n = 0
    for c in set(s_wat.chain):
        for r in set(s_wat.extract("chain", c, "==").resi):
            wat = s_wat.extract(f'chain {c} and resi {r}').coor
            dist = np.linalg.norm(out_coords_all_KDE.reshape(-1,3) - wat, axis=1) #all_density[n]}, ignore_index=True)
            bfactor_out = bfactor_out.append({'resid':r,'resname':'HOH','bfactor':(np.exp(density_all[n])*1000)}, ignore_index=True)  #(np.exp(density_all[dist == min(dist)])[0])*1000},ignore_index=True)
            s.extract(f'chain {c} and resi {r}').b = np.exp(density_all[n])*1000 #(np.exp(density_all[dist == min(dist)]))*1000
            n += 1
    s.tofile(f'{pdb_out}.pdb')
    bfactor_out.to_csv(f'{pdb_out}.csv',index=False)

def remove_overlapping_wat(out_coords_all_KDE, density_all):
    #remove overlapping waters
    #find the waters that are closest to each other
    dist = cdist(out_coords_all_KDE.reshape(-1,3), out_coords_all_KDE.reshape(-1,3))
    np.fill_diagonal(dist, np.inf)


    #find the waters that are closest to each other
    min_dist_idx = np.argmin(dist, axis=1)



#os.chdir('/Users/stephaniewanko/Downloads/water_tracking/')
s = Structure.fromfile('135l.pdb').reorder()

#os.chdir('/Users/stephaniewanko/Downloads/water_tracking/protein_norm/')
cont_dict = np.load('cont_dict.npy',allow_pickle='TRUE').item()
min_ang = np.load('min_ang.npy',allow_pickle='TRUE').item()
max_ang = np.load('max_ang.npy',allow_pickle='TRUE').item()
all_coord_info = np.load('dih_info.npy',allow_pickle='TRUE').item()

#os.chdir('/Users/stephaniewanko/Downloads/water_tracking/protein_norm/')
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
print('30000')
start = time.perf_counter()
#out_coords_all_KDE, norm_all = place_all_wat(all_coord_info,
#                                                                       s, 
#                                                                       center_coords, 
#                                                                       min_ang, 
#                                                                       spread, 
#                                                                       all_density_vals, 
#                                                                       cont_dict,
#                                                                       cutoff_idx,
#                                                                       all_xyz_coords,
#                                                                       rel_b_list,
#                                                                       q_list, norm_val,
#                                                                       use_cutoff=False
#                                                                       )
end = time.perf_counter()
ms = (end-start) * 10**6
print(f'Place all wat 16 processes: {ms:.03f} micro secs.')
#np.save('all_out_coord_test_30000.npy', out_coords_all_KDE)
#np.save('normalization_test_30000.npy', norm_all)
out_coords_all_KDE = np.load('all_out_coord_test_30000.npy',allow_pickle='TRUE')
norm_all = np.load('normalization_test_30000.npy',allow_pickle='TRUE')


start = time.perf_counter()
density_all = structure_based_KDE(out_coords_all_KDE.reshape(-1,3), norm_all,s)
end = time.perf_counter()
ms = (end-start) * 10**6
print(f'Structure based KDE tophat with 0.1% error: {ms:.03f} micro secs.')

start = time.perf_counter()
reassign_bfactors(s, out_coords_all_KDE, density_all, '135l_all_protein_norm_05band_water')
end = time.perf_counter()
ms = (end-start) * 10**6
print(f'Reassign B: {ms:.03f} micro secs.')

