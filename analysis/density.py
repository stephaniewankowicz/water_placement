#!/usr/bin/env python

import os
import numpy as np
#import pandas as pd
from argparse import ArgumentParser
from sklearn.neighbors import KernelDensity
from sklearn.cluster import MeanShift
from sklearn.cluster import  estimate_bandwidth
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd



def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", help="directory to write files out to")
    p.add_argument("--in_dir", help="input directory (where to read files from)")
    p.add_argument("--pt", help="percentile cuttoff for meanshift")
    p.add_argument("--length", help="max number of waters to use")
    p.add_argument("--band", help="bandwidth for MSE")
    args = p.parse_args()
    return args


def find_density(coords_all, pt, length, band):
    '''
    Function for finding density of water distribution around a set of 4 atoms, and
    also clustering density using Mean Shift clustering from sklearn
    Parameters
    ----------
    coords_all : dictionary
        a kind of confusing dictionary (sorry) formatted
        {4 atom set : tuple(center 4 at coord,
                            4 atom coords, 
                            water coords, 
                            normalized b vals for water, 
                            occupancy for water, water normalized value)}
    pt : int
        percentile cutoff for what to do mean shift clustering on
    length : int
        maximum size of data to do analysis on
    band: int
        bandwidth in meanshift clustering algorithm
    Returns
    -------
    writes out to a bunch of npy files including:
        center_coor : center coordinates of clusters from mean shift
        spread : spread around each center coordinate
        dens_all : density value of every point
        coord_set_all : coordinate value of every point
        rel_b_list_all : normalized b factor of every point
        q_list_all : occupancy of every point
        labs : labels of points included in meanshift clustering
        cutoff_idx : indices of points included in meanshift clustering
    
    '''
    # initialize dictionaries
    center_coor={} 
    spread={}
    dens_all={}
    coord_set_all={}
    rel_b_list_all = {}
    q_list_all={}
    labs={}
    cutoff_idx={}
    cutoff_idx_current={}
    density_all = []
    labels_out = []
    #get normalization factor:
    total_protein = 0
    band = int(band)
    pt = int(pt)
    length = int(length)
    for atom_set, atom_set_v in coords_all.items(): #for every 4 sets of atoms
        for dih, coords in atom_set_v.items(): #for every dihedral group of that 4 set of atoms
            total_protein += len(coords[1])
    print(f'total protein: {total_protein}')
    
    for atom_set, atom_set_v in coords_all.items():
        #print(f'atom set: {atom_set}')
        center_coor[atom_set] = {}
        spread[atom_set] = {}
        dens_all[atom_set] = {}
        coord_set_all[atom_set] = {}
        rel_b_list_all[atom_set] = {}
        q_list_all[atom_set] = {}
        labs[atom_set]={}
        cutoff_idx[atom_set]={}
        cutoff_idx_current[atom_set]={}
        for dih_assig, coords in atom_set_v.items():
            #print(f'dih_assig: {dih_assig}')
            center_coor_tmp=[]
            spread_tmp=[]
            #normalization factor
            norm_factor = len(coords[1])/total_protein
            # water coordinates
            wat_coords = np.array(list(coords_all[atom_set][dih_assig][2])).flatten().reshape(-1,3)
            # normalized b for waters
            rel_b_val = np.array(coords_all[atom_set][dih_assig][3])
            # occupancy
            q_val = np.array(coords_all[atom_set][dih_assig][4])
            # normalized water
            norm_val = np.array(coords_all[atom_set][dih_assig][5])
            #norm_val = np.ones(len(wat_coords))
            #norm_val = norm_val * norm_factor
            # this is so we can subsample a b
            sampling = max(1, int(len(wat_coords)/int(length)))
            wat_coords = wat_coords[::sampling]
            rel_b_val = rel_b_val[::sampling]
            q_val = q_val[::sampling]
            norm_val = norm_val[::sampling]
            # first, we fo KDE, to find a density value for each point
            # here I have bandwidth = 1 since that generally works and is the default param
            print(wat_coords.shape)
            kde = KernelDensity(kernel='gaussian', bandwidth=1, rtol=1E-4, atol=1E-4).fit(wat_coords) #, sample_weight=norm_val
            #normalize by the number of protein 4 coor (sep by atom types & dihedrals) we have in our set/total number of proteins 
            density = kde.score_samples(wat_coords) #* norm_factor
            idx = np.where(density>np.percentile(density, pt))[0]
            cutoff_idx[atom_set][dih_assig]={}
            # just so that we can see the indices of a bunch of percentile cutoffs
            for pt_i in np.arange(0, 100, 10):
                cutoff_idx[atom_set][dih_assig][pt_i] = np.where(density>np.percentile(density, pt_i))[0]
            if len(idx)>0:
                #print(len(idx))
                #print(atom_set, dih_assig)
                # now doing meanshift on these points
                # the bin_seeding means that points are binned into grids. Increasing min_bin_freq
                # increases the # of points that need to be in a bin
                # cluster_all=False means we don't cluster everything
                # these params have not been optimized at all
                msc = MeanShift(bandwidth=band, bin_seeding=False)
                msc.fit(wat_coords[idx])
                cluster_centers = msc.cluster_centers_
                labels = msc.labels_
                cluster_label = np.unique(labels)
                n_clusters = len(cluster_label)
                # this is to find "spread" of points (or distance of furthest point within a
                # cluster to it's cluster center)
                #print(n_clusters)
                density_all.append(density)
                label = str(atom_set) + '_' + str(dih_assig) + '_' + str(len(coords[1]))#str(n_clusters)
                labels_out = np.append(labels_out, label)
                #print(cluster_label)
                for i, cc in enumerate(cluster_centers):
                    pos = np.argsort(cdist([cc], wat_coords[idx]))[0][0]
                    center_coor_tmp.append(wat_coords[idx][pos])
                    radius = max(cdist(wat_coords[idx][labels==i], [cluster_centers[i]]))
                    spread_tmp.append(radius)
                    center_coor[atom_set][dih_assig] = center_coor_tmp # xyz of center of cluster
                    spread[atom_set][dih_assig] = spread_tmp # "radius" of cluster
                    labs[atom_set][dih_assig] = labels # cluster each point belongs in
            else:
                print('no idx')
                print(atom_set, dih_assig)
            dens_all[atom_set][dih_assig] = density # density val of all points
            coord_set_all[atom_set][dih_assig] = wat_coords # xyz of all points
            rel_b_list_all[atom_set][dih_assig] = rel_b_val # b factors of points
            q_list_all[atom_set][dih_assig] = q_val # occupancy of waters
            cutoff_idx_current[atom_set][dih_assig] = idx # indices above percentile density
                
    print('saving')
    np.save(f'center_coor_{length}_{pt}_{band}.npy', center_coor) 
    np.save(f'spread_{length}_{pt}_{band}.npy', spread) 
    np.save(f'density_vals_{length}_{pt}_{band}.npy', dens_all) 
    np.save(f'all_xyz_coords_{length}_{pt}_{band}.npy', coord_set_all) 
    np.save(f'rel_b_list_{length}_{pt}_{band}.npy', rel_b_list_all) 
    np.save(f'q_list_{length}_{pt}_{band}.npy', q_list_all) 
    np.save(f'labels_{length}_{pt}_{band}.npy', labs)
    np.save(f'cutoff_idx_{length}_{pt}_{band}.npy', cutoff_idx_current)
    np.save(f'cutoff_idx_{length}_all.npy', cutoff_idx)
    return coord_set_all, center_coor

os.chdir('/Users/stephaniewanko/Downloads/water_tracking/normalized_water')
coords_all = np.load('dih_info.npy',allow_pickle='TRUE').item()

os.chdir('/Users/stephaniewanko/Downloads/water_tracking/non_norm')
args = parse_args()
coord_set_all, center_coord = find_density(coords_all, args.pt, args.length, args.band)

#fig = plt.figure()
#ax = plt.subplot(111)
#for i in range(96):
#    ax.hist(all_density_norm[i], density=False, label=labels_norm[i])

#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# fig = plt.figure()
# ax = plt.subplot(111)
# for i in range(96):
#     ax.hist(all_density_nonorm[i], density=False, label=labels_bonorm[i])

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# def main():
#     args = parse_args()
#     out_dir = args.out
#     in_dir = args.in_dir
#     pt = args.pt
#     length = args.length
#     os.chdir(in_dir)
#     coords_all = np.load('dih_info.npy',allow_pickle='TRUE').item()
#     os.chdir(out_dir)
#     find_density(coords_all, pt, length, args.band)

#if __name__ == '__main__':
    #main()
#
