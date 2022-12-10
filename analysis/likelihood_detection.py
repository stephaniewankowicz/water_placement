#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from qfit.structure import Structure
from DICT4A import DICT4A
from DICT4A_ALLAT import DICT4A_ALLAT
from analysis_functions import *	

#define the function
def coord_likelihood_function(base_coord, all_coords, sigma):
    #print(-np.sum((base_coord-all_coords)**2)
    return np.exp(-np.sum((base_coord-all_coords)**2, axis=1)/(2*sigma**2))

def coord_log_likelihood_function(base_coord, all_coords, sigma):
    #print(-np.sum((base_coord-all_coords)**2)
    return np.log(-np.sum((base_coord-all_coords)**2, axis=1)/(2*sigma**2))

def threeD_likelihood_function(x, y, z, x_array, y_array, z_array, sigma):
    return np.exp(-((x-x_array)**2 + (y-y_array)**2 + (z-z_array)**2)/2*sigma**2)


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
#np.save('lower_band_density.npy', density_all)
#np.save('lower_band_out_coords.npy', out_coords_all_KDE)
#out_coords_all_KDE = np.load('out_coords.npy',allow_pickle='TRUE')
#density_all = np.load('density.npy',allow_pickle='TRUE')



def get_likelihood_all_waters(out_coords, density_all, s, pdb_out=''):
    s_wat = s.extract('resn', 'HOH', '==')
    ML_out = pd.DataFrame(columns=['resid','resname','likelihood'])
    for c in set(s_wat.chain):
        for r in set(s_wat.extract("chain", c, "==").resi):
            wat = s_wat.extract(f'chain {c} and resi {r}').coor
            print(wat)
            wat_like = coord_likelihood_function(wat,  out_coords.reshape(-1, 3), 1)
            log_likelihood = coord_log_likelihood_function(wat,  out_coords.reshape(-1, 3), 1)
            print(np.exp(density_all[wat_like== min(wat_like)])[0])
            print(density_all[log_likelihood== min(log_likelihood)])
            ML_out = ML_out.append({'resid':r,'resname':'HOH','likelihood':(np.exp(density_all[wat_like== min(wat_like)])[0])*1000},ignore_index=True)
    #s.tofile(f'{pdb_out}.pdb')
    ML_out.to_csv(f'likelihood_{pdb_out}.csv',index=False)
    return ML_out
            
get_likelihood_all_waters(out_coords_all_KDE, density_all, s, pdb_out='135l_nonnorm')
