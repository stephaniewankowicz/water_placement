#!/usr/bin/env python
import os.path
import os
import numpy as np
import pandas as pd
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

def new_dihedral(p):
    """
    Function for calculated the dihedral of a given set of 4 points.
    (not my function)
    Parameters 
    ----------
    p : nd.array, shape=(4, 3)
        4 points you want to calculate a dihedral from
    Returns
    -------
    dih_ang : float
        calculated dihedral angle
    """
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    dih_ang = np.degrees(np.arctan2(y, x))
    return dih_ang



def rigid_transform_3D(A, B):
    '''
    * Not my function *
    from : https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    function for calculating the optimal rotation and transformation matrix for a set of 4 points
    onto another set of 4 points
    Input: expects 3xN matrix of points
    Returns R,t
        R = 3x3 rotation matrix
        t = 3x1 column vector
    '''
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B
    
    return R, t



def place_all_centers(s, 
                  all_coord_info, 
                  center_coords, 
                  min_ang, 
                  spread, 
                  all_density_vals, 
                  cont_dict,
                  cutoff_idx,
                  all_xyz_coords,
                  rel_b_list,
                  q_list,
                  use_cutoff=False):
    '''
    lazy function for placing center waters on a whole pdb structure
    '''
    prot = s.extract('resn', 'HOH', '!=').coor
    out_coords = np.array([])
    sz_all = np.array([])
    out_coords_all={}
    out_coords_all_dens={}
    for r in list(s.residues):
        if r.resn[0] in list(DICT4A.keys()):
            print(r.resn[0])
            new_center_coords, new_all_xyz_coords, dens_v_all, b_all, q_all, new_spread = get_new_coords_og(all_coord_info,
                                                                                       r, 
                                                                                       center_coords, 
                                                                                       min_ang, 
                                                                                       spread, 
                                                                                       all_density_vals, 
                                                                                       cont_dict,
                                                                                       cutoff_idx,
                                                                                       all_xyz_coords,
                                                                                       rel_b_list,
                                                                                       q_list,
                                                                                       use_cutoff
                                                                                     )
            min_d = np.min(cdist(new_center_coords.reshape(-1,3), prot), axis=1)
            # veryyy loose cuttoff here to not include waters
            out_coords = np.append(out_coords, new_center_coords.reshape(-1,3)[np.where(min_d>2.1)])
    return out_coords, out_coords_all, out_coords_all_dens, sz_all



def place_all_wat(all_coord_info, s,
                  center_coords, 
                  min_ang, 
                  spread, 
                  all_density_vals, 
                  cont_dict,
                  cutoff_idx,
                  all_xyz_coords,
                  rel_b_list,
                  q_list,
                  use_cutoff=False):
    '''
    lazy function for placing waters on a whole pdb structure
    '''
    prot = s.extract('resn', 'HOH', '!=').coor
    out_coords = np.array([])
    sz_all = np.array([])
    out_coords_all = []
    density_all = []
    out_coords_all_dict = {}
    out_coords_all_dens={}
    for r in list(s.residues):
        if r.resn[0] in list(DICT4A.keys()):
            print(r.resn[0])
            print(r.resi[0])
            new_center_coords, new_all_xyz_coords, dens_v_all, b_all, q_all, new_spread = get_new_coords_og(all_coord_info,
                                                                                       r, 
                                                                                       center_coords, 
                                                                                       min_ang, 
                                                                                       spread, 
                                                                                       all_density_vals, 
                                                                                       cont_dict,
                                                                                       cutoff_idx,
                                                                                       all_xyz_coords,
                                                                                       rel_b_list,
                                                                                       q_list, s,
                                                                                       use_cutoff=False)
            min_d = np.min(cdist(new_center_coords.reshape(-1,3), prot), axis=1)
            density_all = np.append(density_all, dens_v_all)
            # veryyy loose cuttoff here to not include waters
            out_coords = np.append(out_coords, new_center_coords.reshape(-1,3)[np.where(min_d>2.1)])
            out_coords_all = np.append(out_coords_all, new_all_xyz_coords.reshape(-1,3))
            #sz_all = np.append(sz_all, new_spread[np.where(min_d>2.1)])  
            out_coords_all_dict[(r.chain[0], r.resi[0])] = new_all_xyz_coords.reshape(-1,3)
            out_coords_all_dens[(r.chain[0], r.resi[0])] = dens_v_all
    return out_coords, out_coords_all, out_coords_all_dens, sz_all, density_all




def build_center_placement_pdb(xyz_coor, pdb_fn, sphere_size_fn):
    '''
    builds a pdb file with centers of clusters, as well as a pml file to size the
    centers by their spread.
    '''
    listy_wat = {}
    count=1
    for i, (xi, yi, zi) in enumerate(zip(np.array(xyz_coor)[:,0],
                                         np.array(xyz_coor)[:,1],
                                         np.array(xyz_coor)[:,2])):
        listy_wat[i] = ['HETATM', str(i+1), 'O', 'HOH', str(i+1), str(format(xi, '.3f')), str(format(yi, '.3f')), str(format(zi, '.3f')), '1.00', '1.00', 'O']
    file = pdb_fn
    file = open(file, "w") 
    for row in list(listy_wat.values()):
        file.write("{: >1} {: >4} {: >2} {: >5} {: >5} {: >11} {: >7} {: >7} {: >5} {: >5} {: >11}\n".format(*row))
    file.close()
    #with open(sphere_size_fn, 'w') as f:
    #    for i, rad in enumerate(spread): 
    #        print(rad)
    #        f.write('show spheres, resi %s and %s\n' % (i, pdb_fn[:-4]))
    #       f.write('set sphere_scale, %s, resi %s and %s\n' % (rad/3, i, pdb_fn[:-4]))
    return



def build_density_pdb(xyz_coor, fn, density):
    '''
    builds a pdb file with waters around the residue (and density in b-factor column).
    '''
    listy_wat = {}
    count=1
    # max of 10000 waters bc otherwise the pdb breaks since this is a very hacky way of building a pdb lol
    sample = len(density)//10000 + 1 
    for i, (xi, yi, zi, di) in enumerate(zip(np.array(xyz_coor)[:,0][::sample],
                                             np.array(xyz_coor)[:,1][::sample],
                                             np.array(xyz_coor)[:,2][::sample], 
                                        np.array(density)[::sample])):
        count=count+1
        listy_wat[i] = ['HETATM', str(count), 'O', 'HOH', str(count), str(format(xi, '.3f')), str(format(yi, '.3f')), str(format(zi, '.3f')), '1.00', str(format(di, '.3f')), 'O']

    file = fn
    file = open(file, "w") 
    for row in list(listy_wat.values()):
        file.write("{: >1} {: >4} {: >2} {: >5} {: >5} {: >11} {: >7} {: >7} {: >5} {: >5} {: >11}\n".format(*row))
    file.close()
    return