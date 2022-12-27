#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from analysis_functions import *	
from qfit.structure import Structure
import glob
from multiprocess import Pool #multiprocessing
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
from argparse import ArgumentParser
from DICT4A import DICT4A
from DICT4A_ALLAT import DICT4A_ALLAT
import time


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--pdb", help="Name of PDB")
    p.add_argument("--band", help="bandwidth")
    args = p.parse_args()
    return args

def parallel_score_samples(kde, samples, thread_count=int(16)):
    with Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))


def structure_based_KDE(coords_all, norm_factor, s):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.3, atol=0.0005,rtol=0.001).fit(coords_all,sample_weight = norm_factor)
    s_wat = s.extract('resn', 'HOH', '==')
    new_water = s_wat.coor
    density = parallel_score_samples(kde, new_water, 16)
    #density = kde.score_samples(new_water) #* norm_factor
    return density

def reassign_bfactors(s, out_coords_all_KDE, density_all, pdb_out,PDB,band):
    bfactor_out = pd.DataFrame(columns=['chain','resid','resname','alt','bfactor', 'PDB', 'band'])
    s_wat = s.extract('resn', 'HOH', '==')
    n = 0
    for c in set(s_wat.chain):
        for r in set(s_wat.extract("chain", c, "==").resi):
            print(np.unique(s_wat.extract(f'chain {c} and resi {r}').altloc))
            if len(np.unique(s_wat.extract(f'chain {c} and resi {r}').altloc)) > 1:
               for a in set(s_wat.extract(f'chain {c} and resi {r}').altloc):
                 wat = s_wat.extract(f'chain {c} and resi {r} and altloc {a}').coor
                 dist = np.linalg.norm(out_coords_all_KDE.reshape(-1,3) - wat, axis=1)
                 bfactor_out = bfactor_out.append({'chain':c, 'resid':r,'resname':'HOH','alt':a,'bfactor':(np.exp(density_all[n])*1000),'PDB':PDB,'band':band},ignore_index=True)  #(np.exp(density_all[dist == min(dist)])[0])*1000},ignore_index=True)
                 s.extract(f'chain {c} and resi {r} and altloc {a}').b = np.exp(density_all[n])*1000
                 n += 1
            else:
              wat = s_wat.extract(f'chain {c} and resi {r}').coor
              dist = np.linalg.norm(out_coords_all_KDE.reshape(-1,3) - wat, axis=1) #all_density[n]}, ignore_index=True)
              bfactor_out = bfactor_out.append({'chain':c, 'resid':r,'resname':'HOH','alt':'','bfactor':(np.exp(density_all[n])*1000),'PDB':PDB,'band':band},ignore_index=True)  #(np.exp(density_all[dist == min(dist)])[0])*1000},ignore_index=True)
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



def load_data(pdb):
    s = Structure.fromfile(pdb + '.pdb').reorder()

    #os.chdir('/wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis/')
    cont_dict = np.load('cont_dict.npy',allow_pickle='TRUE').item()
    min_ang = np.load('min_ang.npy',allow_pickle='TRUE').item()
    max_ang = np.load('max_ang.npy',allow_pickle='TRUE').item()
    all_coord_info = np.load('dih_info.npy',allow_pickle='TRUE').item()

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
    return all_coord_info, s, center_coords, min_ang, spread, all_density_vals, cont_dict, cutoff_idx, all_xyz_coords, rel_b_list, q_list, norm_val

def place_all_water(all_coord_info, s, center_coords, min_ang, spread, all_density_vals, cont_dict, cutoff_idx, all_xyz_coords, rel_b_list, q_list, norm_val, pdb):
    start = time.perf_counter()
    s_protein = s.extract("record", "HETATM", "!=")
    out_coords_all_KDE, norm_all = place_all_wat(all_coord_info,
                                                                       s_protein, 
                                                                       center_coords, 
                                                                       min_ang, 
                                                                       spread, 
                                                                       all_density_vals, 
                                                                       cont_dict,
                                                                       cutoff_idx,
                                                                       all_xyz_coords,
                                                                       rel_b_list,
                                                                       q_list, norm_val,
                                                                       use_cutoff=False
                                                                       )
    end = time.perf_counter()
    ms = (end-start) * 10**6
    print(f'Place all wat 16 processes: {ms:.03f} micro secs.')
    np.save(pdb + '_all_out_coord_test_30000.npy', out_coords_all_KDE)
    np.save(pdb + '_normalization_test_30000.npy', norm_all)
    #out_coords_all_KDE = np.load('all_out_coord_test_30000.npy',allow_pickle='TRUE')
    #norm_all = np.load('normalization_test_30000.npy',allow_pickle='TRUE')
    return out_coords_all_KDE, norm_all    

def main():
    args = parse_args()
    #all_coords_info, s, center_coords, min_ang, spread, all_density_vals, cont_dict, cutoff_idx, all_xyz_coords, rel_b_list, q_list, norm_val = load_data(args.pdb)
    s = Structure.fromfile(args.pdb + '.pdb').reorder()
    #out_coords_all_KDE, norm_all = place_all_water(all_coords_info, s, center_coords, min_ang, spread, all_density_vals, cont_dict, cutoff_idx, all_xyz_coords, rel_b_list, q_list, norm_val, args.pdb)
    start = time.perf_counter()
    out_coords_all_KDE = np.load(args.pdb + '_all_out_coord_test_30000.npy',allow_pickle='TRUE')
    norm_all = np.load(args.pdb +  '_normalization_test_30000.npy',allow_pickle='TRUE')
    density_all = structure_based_KDE(out_coords_all_KDE.reshape(-1,3), norm_all,s)
    end = time.perf_counter()
    ms = (end-start) * 10**6
    print(f'Structure based KDE tophat with 0.1% error: {ms:.03f} micro secs.')

    start = time.perf_counter()
    reassign_bfactors(s, out_coords_all_KDE, density_all, args.pdb + '_' + args.band ,args.pdb, args.band)
    end = time.perf_counter()
    ms = (end-start) * 10**6
    print(f'Reassign B: {ms:.03f} micro secs.')

if __name__ == '__main__':
    main()
