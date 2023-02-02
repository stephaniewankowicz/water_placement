#!/usr/bin/env python3
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
from argparse import ArgumentParser
from DICT4A_test2 import DICT4A
#from DICT4A_ALLAT import DICT4A_ALLAT
import time


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--pdb", help="Name of PDB")
    args = p.parse_args()
    return args

def load_data(pdb):
    os.chdir('/wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis/test_proteins/' + pdb)
    s = Structure.fromfile(pdb + '.pdb').reorder()

    os.chdir('/wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/new_new_DICT/')
    cont_dict = np.load('cont_dict.npy',allow_pickle='TRUE').item()
    min_ang = np.load('min_ang.npy',allow_pickle='TRUE').item()
    max_ang = np.load('max_ang.npy',allow_pickle='TRUE').item()
    all_coord_info = np.load('dih_info.npy',allow_pickle='TRUE').item()

    pt=50
    length = 50000
    band = '_1'
    center_coords = np.load(f'center_coor_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
    norm_val = np.load(f'norm_wat_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
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
    np.save(pdb + '_all_out_coord_test_50000.npy', out_coords_all_KDE)
    np.save(pdb + '_normalization_wat_test_50000.npy', norm_all)
    return out_coords_all_KDE, norm_all

def main():
    args = parse_args()
    os.chdir('/wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis')
    all_coords_info, s, center_coords, min_ang, spread, all_density_vals, cont_dict, cutoff_idx, all_xyz_coords, rel_b_list, q_list, norm_val = load_data(args.pdb)
    os.chdir('/wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis')
    out_coords_all_KDE, norm_all = place_all_water(all_coords_info, s, center_coords, min_ang, spread, all_density_vals, cont_dict, cutoff_idx, all_xyz_coords, rel_b_list, q_list, norm_val, args.pdb)


if __name__ == '__main__':
    main()
