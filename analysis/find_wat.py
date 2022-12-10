# this is so that we will itterate through each chain and atom
import os.path
import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from qfit.structure import Structure
import glob
from multiprocessing import Pool
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from scipy.spatial.distance import cdist
from DICT4A import DICT4A
from DICT4A_ALLAT import DICT4A_ALLAT
from multiprocessing import Pool
from scipy.signal import argrelextrema

pd.set_option('display.max_columns', None)

def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--pdb", help="path to input pdb directories (with pattern)")
    p.add_argument("--out", help="directory to write files out to")
    args = p.parse_args()
    return args

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

def project_data(
        data, populations=None, x_range=None, n_points=1000, sigma=None):
    '''
    ***not my own function*** borrowed from someone in my previous lab but super useful
    To gaussianize a histogram
    Parameters
    ----------
    data : nd.array, shape=(n_states, )
        The value of the order parameters.
    populations : nd.array, shape=(n_states, )
        The population of each state.
    x_range : array, shape=(2, ), default=None,
        The x-axis plot range. i.e. [1, 5].
    n_points : int, default=1000,
        Number of points to use for plotting data.
    sigma : float, default=None,
        The width to use for each gaussian. If none is supplied, defaults to
        1/20th of the `x_range`.
    
    Returns
    -------
    xs : nd.array, shape=(n_points, ),
        The x-axis values of the resultant projection.
    ys : nd.array, shape=(n_points, )
        The y-axis values of the resultant projection.
    '''
    data_spread = data.max() - data.min()
    if populations is None:
        populations = np.ones(data.shape[0])/data.shape[0]
    if x_range is None:
        delta_0p1 = data_spread*0.1
        x_range = [data.min()-delta_0p1, data.max()+delta_0p1]
    if sigma is None:
        sigma = data_spread/20.
    range_spread = x_range[1] - x_range[0]
    xs = range_spread*(np.arange(n_points)/n_points) + x_range[0]
    ys = np.zeros(xs.shape[0])
    for n in np.arange(data.shape[0]):
        ys += sample_gaussian(xs, populations[n], data[n], sigma=sigma)
    return xs, ys

def sample_gaussian(xs, A, B, sigma):
    ys = A/(sigma*np.sqrt(2*np.pi)) * np.exp((-0.5*((xs-B)/sigma)**2))
    return ys

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
        print("matrix A is not 3xN")#, it is {num_rows} x {num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        print("matrix B is not 3xN") #, it is {num_rows} x {num_cols}")

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

def get_coord_list(fn):      
    '''
    A function for obtaining the coordinates of waters close (<3.5 A) to sets of 4 atoms 
    defined by DICT4A
    
    Parameters 
    ----------
    fn : string
       a filename of the structure you want to obtain the information from
    Returns 
    -------
    df : DataFrame
        a dataframe containing information for later analysis. that information inclues infor of the
        sets of 4 atoms in their structure and their surrounding water
    '''
    # this is so we don't keep waters that "clash" ie are within this distance to the atoms
    dict_dist = {
        'Cm' : 3.0,
        'Nm' : 2.4,
        'Om' : 2.4,
        'S' : 2.4,
        'C' : 3.0,
        'N' : 2.4,
        'O' : 2.4
    }
    coord_list = []
    #print(fn)
    pdb_id = fn[8:12] #fn[8:14]
    #print(f'starting {pdb_id}')
    wat_arr = [] #array of water ids for normalization
    try:
        pdb_struct = Structure.fromfile(fn).reorder()
    except ValueError:
        return
    except RuntimeError:
        return
    water = pdb_struct.extract('resn HOH') # extracting just the waters
    # so we can get each residue+chain combo and itterate through that
    #print(water) 
    res_list_idx = np.unique([(r.chain[0]+str(r.resi[0])) for r in list(pdb_struct.residues)])
    #print(len(res_list_idx))
    num1 = 0
    for idx, res_s in enumerate(res_list_idx):
        #num1 += 1
        if res_s[0].isalpha():
            res = pdb_struct.extract(f'resi {res_s[1::]} and chain {res_s[0]}')
            if res.resn[0] in list(DICT4A.keys()): # make sure it's actually a residue!
                res_info = np.array(list(zip(res.coor, res.b, res.q)), dtype=object)[:,0]
                # check for altlocs (if no altlocs, this will just be '')
                all_altlocs = np.unique(res.altloc) 
                for a in all_altlocs:
                    # atom indices of residue only with the altloc we want (this is for when residues 
                    # define altloc sidechain and nonaltloc backbone together - we want to keep the 
                    # whole residue!
                    poss_idx = np.where(np.array((res.altloc=='')*1 + (res.altloc==a)*1)>=1)[0] 
                    name2idx = dict(zip(res.name[poss_idx], np.arange(len(res.name[poss_idx]))))
                    for at_s_k, at_s_v in DICT4A[res.resn[0]].items(): 
                        all_atoms = [y for x in at_s_k 
                                     for y in (x if isinstance(x, tuple) else (x,))]
                        check_all_atoms = [name2idx[e] for e in all_atoms if e in name2idx]
                        # basically checks to make sure we have all the atoms
                        if len(all_atoms) == len(check_all_atoms): 
                             # want to deal with aromatics and prolines a bittt differently 
                             # (or at least the atom sets that contain their rings)
                            if 'PR1' in at_s_v or 'YR1' in at_s_v or 'FR1' in at_s_v or 'WR1' in at_s_v or 'HR1' in at_s_v:
                                # this is all the atoms in the atom set
                                all_atoms = [y for x in at_s_k 
                                              for y in (x if isinstance(x, tuple) else (x,))]
                                at_coords = []
                                all_coords = []
                                for at in at_s_k:
                                    if len(at) > 2: # basically, if it is a ring
                                        ring_idx = [name2idx[e] for e in at] # get indices of rings
                                        if len(ring_idx) == len(at): # so if we have all residues
                                            ring_coords = res.coor[poss_idx][ring_idx]
                                            # get the center of mass!
                                            ring_com = sum(ring_coords)/len(ring_coords) 
                                            for rc in ring_coords:
                                                all_coords.append(rc)
                                            num1 += 1
                                            at_coords.append(ring_com) 
                                    else: # otherwise we deal with it normally
                                        at_coords.append(res.coor[poss_idx][name2idx[at]])
                                        num1 += 1
                                        all_coords.append(res.coor[poss_idx][name2idx[at]])
                                b_fact_prot = sum(res.b[poss_idx])/len(res.b[poss_idx])
                                # looking at distance to water with all atoms included
                                dist2wat = cdist(water.coor, all_coords)
                                prod = np.full((dist2wat[:,0].shape), True)
                                for i in range(dist2wat.shape[1]):
                                    non_clash_tmp = (dist2wat[:,i]>dict_dist[DICT4A_ALLAT[res.resn[0]][tuple(at_s_v)][i]])
                                    prod = prod*non_clash_tmp
                                lower_cutoff = np.where(prod)[0]
                                upper_cuttoff = list(np.where(dist2wat<3.5)[0])
                                water_coords_idx = list(set(upper_cuttoff).intersection(lower_cutoff))
                            # if not dealing with a ring:
                            else:
                                # indices we currently care about
                                curr_idx = [name2idx[e] for e in at_s_k]
                                # coordinates of 4 atoms
                                at_coords = res.coor[poss_idx][curr_idx] 
                                # to save avg b factor of whole residue
                                b_fact_prot = sum(res.b[poss_idx])/len(res.b[poss_idx]) 
                                # distance between 4 atoms and ALL waters
                                dist2wat = cdist(water.coor, at_coords) 
                                # we only want waters within 3.5 A
                                upper_cuttoff = list(np.where(dist2wat<3.5)[0]) 
                                # now we want to only keep waters that DON'T CLASH with the atoms
                                prod = np.full((dist2wat[:,0].shape), True)
                                for i in range(dist2wat.shape[1]):
                                    non_clash_tmp = (dist2wat[:,i]>dict_dist[at_s_v[i]])
                                    prod = prod*non_clash_tmp
                                lower_cutoff = np.where(prod)[0]
                                # these are the "good" waters we want to keep
                                water_coords_idx = list(set(upper_cuttoff).intersection(lower_cutoff))
                            # if we have at least one water, we want to record it!
                            if len(water_coords_idx)>0: 
                                print(f'putting into df: {len(water_coords_idx)}')
                                water_coords = water.coor[np.array(water_coords_idx)]
                                water_b = water.b[np.array(water_coords_idx)]
                                water_q = water.q[np.array(water_coords_idx)]
                                water_b_reshape = water_b.reshape(-1,water_coords.shape[0])
                                water_q_reshape = water_q.reshape(-1,water_coords.shape[0])
                                water_coor_reshape = np.array(water_coords_idx).reshape(-1,water_coords.shape[0])
                                water_resi = water.resi[np.array(water_coords_idx)]
                                water_resi_reshape = water_resi.reshape(-1,water_coords.shape[0])
                                water_info = np.hstack((water_coords,
                                                        water_b_reshape.T, 
                                                        water_q_reshape.T, water_coor_reshape.T, water_resi_reshape.T))    
                                water_info = [tuple(w) for w in water_info]
                                unique_wat = list(set(water_info))
                                wat_coord=[]
                                b=[]
                                q=[]
                                #get all water ids 
                                for e in np.array(unique_wat):
                                    wat_coord = (np.array(e)[0:3])
                                    b = float(np.array(e)[3])
                                    q = float(np.array(e)[4])
                                    wat_id = float(np.array(e)[6])
                                    if not wat_id in wat_arr:
                                       wat_arr.append(wat_id)
                                    coord_list.append(tuple((pdb_id, # the PDB code
                                                     res.chain[0], # chain #
                                                     res.resi[0], # residue #
                                                     res.resn[0], # residue name
                                                     a, # alt loc
                                                     b_fact_prot, # avg b factor of residue
                                                     tuple(at_s_v), # gerneralized name of 4 atoms
                                                     tuple(at_s_k), # name of 4 atoms
                                                     at_coords, # coords of atom set
                                                     wat_coord, # coords if water
                                                     b, q, wat_id, 1.0, 1.0))) # water b and q and coor id and water/protein norm
    #normalize based on wat_id
    coord_df = pd.DataFrame(coord_list, columns =['id', 'chain', 'resi', 'resn', 'alt', 'resb', 'names', 'un_names', 'prot_coord', 'wat_coord', 'b', 'q', 'water_id', 'wat_norm', 'prot_norm']) 

    for wat_id in wat_arr:
        norm_value = len(coord_df[coord_df['water_id'] == wat_id].index) 
        coord_df['wat_norm'] = np.where(coord_df['water_id'] == wat_id, coord_df['wat_norm'].div(norm_value), coord_df['wat_norm'])
    
    #protein normalization    
    for name4atom in np.unique(coord_df['names']):
        num_prot = len(coord_df[coord_df['names'] == name4atom].index)
        coord_df['prot_norm'] = np.where(coord_df['names'] == name4atom, coord_df['prot_norm'].div(num_prot), coord_df['prot_norm'])
 
    dih = [new_dihedral(np.array(e)) for e in coord_df['prot_coord']]
    dih_new=[]
    # not really necessary 
    for di in dih:
        if di<0:
            di+=360
            dih_new.append(di)
        else:
            dih_new.append(di)

    coord_df['dih_new'] = dih_new
    coord_df['dih'] = dih

       
    print(f'finishing {pdb_id}')
    if len(coord_list)>0:
        #df = build_df(pdb_id, coord_list) # turn this into a df 
        return coord_df
    else:
        print('error')
        return 
       

def find_max_ang_idx(df, bucket, max_ang, dih_vals, atoms):
    max_point={}
    for label in np.unique(bucket):
        indx = np.where(np.array(bucket)==label)[0]
        all_dih_vals=np.array(dih_vals)[indx]
        # find idx of value at max
        max_idx = np.argmin(abs(all_dih_vals - max_ang[int(label)]))
        # make this our example prot coord for this dih
        max_point[label] = np.array(df['prot_coord'][(df['names'] == atoms)])[max_idx]  
    return max_point

def cluster_dih_angs(atoms, df):
    '''
    this function goes through the coordinate data and assigns the dihedrals to bins 
    based off the distribution of the angles. It does this by turning the dihedral distribution 
    into a gaussian, and then finds local max/min to define the clusters
    Parameters
    ---------
    atoms : tuple
        name of four atoms we are clustering
    df : DataFrame
        df containing the dihedrals we are clustering with
    Returns
    ------
    bucket : array
        contains the clustering label for each dihedral angle from df
    min_ang : list
        local minima for dihdral distribution
    max_ang : list
        local maxima for dihdral distribution
    cont : boolean
        whether a peak in the distribution wraps around (from 180 <--> -180)
    '''
    # the first step of this is to determine our dihedral binning cutoffs (so we can cluster
    # properly`
    dih_vals = df['dih'][(df['names'] == atoms)]#&(df_cur_al['dih_new'] !='no_dih')]
    cont = False
    if len(dih_vals)>0:
        # i've found that this val for sigma works best
        xs, ys = project_data(np.array(dih_vals), sigma=10) 
        min_ind = argrelextrema(ys, np.less) # find dih idx for minima
        max_ind = argrelextrema(ys, np.greater) # find dih idx for maxima
        max_ang = [e for e in xs[max_ind]] # find dih val for minima
        min_ang = [e for e in xs[min_ind]] # find dih val for maxima
        bucket=[]
        if len(min_ang)==0: # case where there is just one peak
            bucket = 0
            #bucket=np.zeros(len(dih_vals))
            cont=True
        else: # otherwise, we have to assign dih vals to buckets
            for dv in dih_vals:
                # we want to find which min val closest to
                diff_from_mins = dv - np.array(min_ang)
                closest_min_idx = np.argsort(abs(diff_from_mins))[0] 
                #print(closest_min_idx)
                if diff_from_mins[closest_min_idx]>0:
                    # if above closest min, assign to bucket above
                    bucket.append(closest_min_idx+1)
                else:
                    bucket.append(closest_min_idx)
            if ys[np.argmin(abs(xs))] >.0003: # means that the peak wraps around
                cont = True
                bucket = np.array(bucket) + (np.array(bucket)==0)*max(bucket)
                max_ang[max(bucket)] = max(max_ang[0], max_ang[max(bucket)])
    return bucket, min_ang, max_ang, cont, dih_vals

        
def obtain_new_xyz(df, max_point, indx, atoms):
    '''
    Function for obtaining the rotated and transformed coordinates for 4 protein atoms and their
    associated water atoms
    Parameters
    ----------
    df : DataFrame
        df with protein and water coordinates        
    max_point : nd.array
        protein coordinate of peak of dihedral max we will be superimposing onto
    indx : array
        indices that are in dihedral cluster we are looking at
    atoms : tuple
        set of 4 atoms we are looking at
    Returns
    -------
    new_p_coord : list
        new xyz coords for atoms sets
    new_wt_coord : list
        new xyz coords for waters
    norm_b_val : list
        normalized b valuse for waters
    wat_q_val : list
        occupancies for waters
    '''
    new_p_coord = []
    new_wt_coord = []
    norm_b_val = []
    wat_q_val = np.array(df['q'][(df['names'] == atoms)])[indx]
    wat_norm_val = np.array(df['wat_norm'][(df['names'] == atoms)])[indx]
    prot_norm_val = np.array(df['prot_norm'][(df['names'] == atoms)])[indx]
    # zip together the protein and water coordinates with this specific label
    zipped_vals = zip(np.array(df['prot_coord'][(df['names'] == atoms)])[indx],
                np.array(df['wat_coord'][(df['names'] == atoms)])[indx],
                np.array(df['b'][(df['names'] == atoms)])[indx],
                np.array(df['resb'][(df['names'] == atoms)])[indx])
    for i,(p_coord, w_coord, wat_b_val, prot_bval) in enumerate(zipped_vals):
        R, t = rigid_transform_3D(np.array(p_coord).T, np.array(max_point).T)
        new_p_coord.append(np.dot(R, np.array(p_coord).T)+t)
        new_wt_coord.append(np.dot(R, np.array(w_coord).T)+t.T)
        norm_b_val.append(wat_b_val/prot_bval)
    return new_p_coord, new_wt_coord, norm_b_val, wat_q_val, wat_norm_val, prot_norm_val

def build_dictionaries(df, out_dir):
    '''
    function for building dictionaries that are the final output (using obtain_new_xyz from above)
    
    Parameters 
    ----------
    df : DataFrame
        df of all the water/atoms sets info
    out_dir : string
        where you want the output directories to be written to
    
    Returns
    -------
    nothing, but writes out to these files (which are dictionaries), 
    and for every atom set tells you:
    max_ang.npy : the local maxima of the dihdreal distribution
    min_ang.npy : the local minima of the dihedral distribution
    dih_info.npy : all the info of the xyz coords of the template sets + protein + waters, 
        and b factors and occupancy
    cont_dict.npy : if we want to continue a dihedral bin through 180->-180
    '''
    # create our empty dicts
    max_ang_dict={}
    min_ang_dict={}
    cont_dict={}
    dih_info={}
    # first have to define all posible general 4 atom combos
    poss_atom_combo = set()
    for resn, combos in DICT4A.items():
        for gen_combo in combos.values():
            poss_atom_combo.add(tuple(gen_combo))
    # now to build dictionaries...
    for at in poss_atom_combo:
        print(f'building dicts for {at}')
        # first we must define cutoffs for dihedral clustering
        bucket, min_ang, max_ang, cont, dih_vals = cluster_dih_angs(at, df) 
        # find our model protein coordinates (max of each dihedral cluster)
        max_point = find_max_ang_idx(df, bucket, max_ang, dih_vals, at)
        dih_info_tmp={}
        for label in np.unique(bucket): # for each dihedral cluser
            indx = np.where(np.array(bucket)==label)[0] # find indices
            # now we have to do translation/rotation on all prot coords+waters
            new_p_coord, new_wt_coord, norm_b_val, wat_q_val, wat_norm_val, prot_norm_val = obtain_new_xyz(df,
                                                                              max_point[label],
                                                                              indx, 
                                                                              at)
            dih_info_tmp[label] = (max_point[label], new_p_coord, new_wt_coord, norm_b_val, wat_q_val, wat_norm_val, prot_norm_val)
        dih_info[at] = dih_info_tmp
        max_ang_dict[at] = max_ang
        min_ang_dict[at] = min_ang
        cont_dict[at] = cont
    np.save(f"max_ang.npy", max_ang_dict, allow_pickle='TRUE')
    np.save(f"min_ang.npy", min_ang_dict, allow_pickle='TRUE')
    np.save(f"dih_info.npy", dih_info, allow_pickle='TRUE')
    np.save(f"cont_dict.npy", cont_dict, allow_pickle='TRUE')

def main():
    args = parse_args()
    fn = args.pdb
    out_dir = args.out
    os.chdir(fn)
    fns=[]
    for fi in glob.glob("*/*/*pdb"):
        fns.append(fi)
    print('starting pool')
    # build up a massive dataframe of this stuff 
    with Pool(processes=32) as p:
        df = pd.concat(p.starmap(get_coord_list, zip(fns)))
    os.chdir(out_dir)
    #df.to_csv(f'all.csv')
    # now, use this df to build some dictionaries
    build_dictionaries(df, out_dir)

if __name__ == '__main__':
    main()
    
