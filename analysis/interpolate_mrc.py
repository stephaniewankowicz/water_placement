#!/usr/bin/env python
import numpy as np
#import pymol
from cctbx import maptbx

# # Load the density map from a file

density = np.load('/Users/stephaniewanko/Downloads/water_tracking/water_norm/pymol_density.npy')
print(density)
density_int = density.astype(np.int8)
import mrcfile

with mrcfile.new('filename.mrc', overwrite=True) as mrc:
    mrc.set_data(density_int)
    mrc.update_header_from_data()

# # Create a pymol map object
# pmap = pymol.vfont.PymolMap()

# # Set the map data and dimensions
# pmap.set_map_data(density)
# pmap.set_dim(density.shape[0], density.shape[1], density.shape[2])

# # Set the map origin and cell dimensions
# pmap.set_origin(0, 0, 0)
# pmap.set_cell_size(1, 1, 1)

# # Add the map to pymol
# pymol.cmd.load_map(pmap, 'density')


# import gemmi
# mtz = gemmi.Mtz(with_base=True)
# print(density.shape)
# mtz.set_data(density)
# mtz.write_to_file('output.mtz')
#2. Create a gemmi Crystal object and populate its data fields with the numpy array data:

#crystal = gemmi.Crystal()
#crystal.set_data_from_array(data)

#3. Write the Crystal object out as an mtz file:

#gemmi.write_mtz_file(crystal, 'test_density.mtz')


# # Load the density map from a file
# density = np.load('/Users/stephaniewanko/Downloads/water_tracking/water_norm/density.npy')

# # Create a cctbx map object from density
# cctbx_map = maptbx.real_space_grid_3d(data=density)

# # Save the map object to a CCP4 file
# cctbx_map.as_mtz_dataset(column_root_label='DENSITY').mtz_object().write('density.mtz')


# # Load the density map from a file
# density = np.load('density.npy')

# # Create a CCP4 MTZ object
# mtz_obj = mtz.object()

# # Set the cell dimensions and space group
# mtz_obj.set_cell_parameters(a=density.shape[0], b=density.shape[1], c=density.shape[2], alpha=90, beta=90, gamma=90, sg='P1')

# # Add a column to the MTZ object to store the density values
# mtz_obj.add_column('DENSITY', type='F')

# # Set the density values in the MTZ object
# mtz_obj.set_column_data('DENSITY', density.flatten())

# # Save the MTZ object to a file
# mtz_obj.write_to_file('density.mtz')
