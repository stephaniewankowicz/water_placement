#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from qfit.structure import Structure
from DICT4A import DICT4A
from DICT4A_ALLAT import DICT4A_ALLAT
from analysis_functions import *	

def coord_likelihood(all_coords, coord):
  # Calculate the Euclidean distance between the input coordinate and each point in the array of coordinates
  distances = [np.linalg.norm(coord - c) for c in all_coords]
  #close_distances = distance[] #subset
  likelihood = 1 / np.mean(distances)
  return likelihood

def get_likelihood_all_waters(out_coords, density_all, s, pdb_out=''):
    s_wat = s.extract('resn', 'HOH', '==')
    ML_out = pd.DataFrame(columns=['resid','resname','likelihood'])
    for c in set(s_wat.chain):
        for r in set(s_wat.extract("chain", c, "==").resi):
            wat = s_wat.extract(f'chain {c} and resi {r}').coor
            wat_like = coord_likelihood(out_coords.reshape(-1, 3), wat)
            ML_out = ML_out.append({'resid':r,'resname':'HOH','likelihood':wat_like},ignore_index=True)
    ML_out.to_csv(f'likelihood_{pdb_out}.csv',index=False)
    return ML_out

os.chdir('/Users/stephaniewanko/Downloads/water_tracking/')
s = Structure.fromfile('/Users/stephaniewanko/Downloads/water_tracking/135l.pdb').reorder()

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
cutoff_idx = np.load(f'cutoff_idx_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
cutoff_idx_all = np.load(f'cutoff_idx_{length}_all.npy',allow_pickle='TRUE').item()
norm_val = np.load(f'norm_val_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
all_density_vals = np.load(f'density_vals_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
labels = np.load(f'labels_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
spread = np.load(f'spread_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
all_xyz_coords = np.load(f'all_xyz_coords_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
rel_b_list = np.load(f'rel_b_list_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()
q_list = np.load(f'q_list_{length}_{pt}{band}.npy',allow_pickle='TRUE').item()



out_coords, out_coords_all_KDE, out_coords_all_dens, sz_all, density_all, norm_all  = place_all_wat(all_coord_info,
                                                                        s, 
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
#np.save('density.npy', density_all)
#np.save('out_coords.npy', out_coords_all_KDE)
#out_coords_all_KDE = np.load('out_coords.npy',allow_pickle='TRUE')
#density_all = np.load('density.npy',allow_pickle='TRUE')

            
get_likelihood_all_waters(out_coords_all_KDE, density_all, s, pdb_out='135l_nonnorm')
