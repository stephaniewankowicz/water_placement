import numpy as np
import gemmi

#go through each each vocxel grid, get coordinates, return KDE value at that point
#most of my grid points should be 0


grid3 = gemmi.FloatGrid()
grid3.set_unit_cell(gemmi.UnitCell(37.119, 35.163, 37.525, 90, 92.48, 90))
grid3.set_size_from_spacing(0.5, gemmi.GridSizeRounding.Nearest)


# Define an array of weighted coordinates
_7b7b = np.load('7b7b_all_out_coord_test_30000.npy')
_7b7b_weights = np.load('7b7b_normalization_test_30000.npy')
coordinates = np.array(_7b7b.reshape(-1, 3))

# Go through each voxel grid point and calculate the KDE value

for i in range(grid3.nu):
    for j in range(grid3.nv):
        for k in range(grid3.nw):
            # Calculate the KDE value at the grid point
            print(np.array([i, j, k]))
            for coord in coordinates:
                distance = np.linalg.norm(np.array([i, j, k]) - coord)
            #find the position in the array of the coordinate that has the minimum distance
            min_index = np.argmin(distance)
            weight = _7b7b_weights[min_index]
            print(weight)

            # Set the value of the grid point
            grid3.set_value(i, j, k, weight)

ccp4 = gemmi.Ccp4Map()
ccp4.grid = grid3
ccp4.grid.unit_cell.set(37.119, 35.163, 37.525, 90, 92.48, 90)
ccp4.grid.spacegroup = gemmi.SpaceGroup('P1')
ccp4.update_ccp4_header()
ccp4.write_ccp4_map('7b7b_density.ccp4')

