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
    p.add_argument("--band", help="bandwidth")
    p.add_argument("-d", help="directory with npy")
    args = p.parse_args()
    return args

def parallel_score_samples(kde, samples, thread_count=int(16)):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))


def structure_based_KDE(coords_all, norm_factor, s, bandwidth):
    kde = KernelDensity(kernel='gaussian', bandwidth=float(bandwidth), atol=0.0005,rtol=0.001).fit(coords_all,sample_weight = norm_factor)
    s_wat = s.extract('resn', 'HOH', '==')
    new_water = s_wat.coor
    density = parallel_score_samples(kde, new_water, 16)
    return density

def reassign_bfactors(s, out_coords_all_KDE, density_all, pdb_out,PDB,band):
    bfactor_out = pd.DataFrame(columns=['chain','resid','resname','alt','bfactor', 'PDB', 'band'])
    s_wat = s.extract('resn', 'HOH', '==')
    s_wat = s_wat.extract('name', 'O', '==') #we only want oxygen info
    n = 0
    for c in set(s_wat.chain):
        for r in set(s_wat.extract("chain", c, "==").resi):
            if len(np.unique(s_wat.extract(f'chain {c} and resi {r}').altloc)) > 1:
               for a in set(s_wat.extract(f'chain {c} and resi {r}').altloc):
                 print(a)
                 wat = s_wat.extract(f'chain {c} and resi {r} and altloc {a}').coor
                 dist = np.linalg.norm(out_coords_all_KDE.reshape(-1,3) - wat, axis=1)
                 bfactor_out = bfactor_out.append({'chain':c, 'resid':r,'resname':'HOH','alt':a,'bfactor':(np.exp(density_all[n])*1000),'PDB':PDB,'band':band},ignore_index=True)  #(np.exp(density_all[dist == min(dist)])[0])*1000},ignore_index=True)
                 s.extract(f'chain {c} and resi {r} and altloc {a}').b = np.exp(density_all[n])*1000
                 n += 1
            else:
              wat = s_wat.extract(f'chain {c} and resi {r}').coor
              dist = np.linalg.norm(out_coords_all_KDE.reshape(-1,3) - wat, axis=1) #all_density[n]}, ignore_index=True)
              bfactor_out = bfactor_out.append({'chain':c,'resid':r,'resname':'HOH','alt':'','bfactor':(np.exp(density_all[n])*1000),'PDB':PDB,'band':band},ignore_index=True)  #(np.exp(density_all[dist == min(dist)])[0])*1000},ignore_index=True)
              s.extract(f'chain {c} and resi {r}').b = np.exp(density_all[n])*1000 #(np.exp(density_all[dist == min(dist)]))*1000
              n += 1
    s.tofile(f'{pdb_out}_wat_norm.pdb')
    bfactor_out.to_csv(f'{pdb_out}_wat_norm.csv',index=False)

def main():
    args = parse_args()
    os.chdir('/wynton/group/fraser/swankowicz/water_loc/water-scripts/4_atom_scripts/analysis/test_proteins/' + args.pdb)
    s = Structure.fromfile(args.pdb + '.pdb').reorder() 
    os.chdir(args.d)
    out_coords_all_KDE = np.load(args.pdb + '_all_out_coord_test_50000.npy',allow_pickle='TRUE')
    norm_all = np.load(args.pdb + '_normalization_wat_test_50000.npy',allow_pickle='TRUE')
    density_all = structure_based_KDE(out_coords_all_KDE.reshape(-1,3), norm_all, s, args.band)
    reassign_bfactors(s, out_coords_all_KDE, density_all, args.pdb + '_' + args.band ,args.pdb, args.band)

if __name__ == '__main__':
    main()
