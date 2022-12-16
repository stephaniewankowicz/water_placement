#!/usr/bin/env python

import numpy as np
import os
from scipy.ndimage import measurements

from cctbx import uctbx
from cctbx.array_family import flex

#load in data
os.chdir('/Users/stephaniewanko/Downloads/water_tracking/water_norm')
water_density = np.load('water_density_GLU_7.npy')
water_coords = np.load('water_coords_GLU_7.npy')

#interpolate the data

coords = water_coords.reshape(-1,3)
weights = water_density.reshape(-1)


def interpolate_density_map(coords, weights, resolution):
  # Determine the bounding box for the coordinates
  xmin, ymin, zmin = coords.min(axis=0)
  xmax, ymax, zmax = coords.max(axis=0)

  # Create a grid of points within the bounding box
  x, y, z = np.mgrid[xmin:xmax:resolution, ymin:ymax:resolution, zmin:zmax:resolution]

  # Interpolate the density at each grid point
  density = np.zeros_like(x)
  for i in range(len(coords)):
      density += weights[i] * np.exp(-((x - coords[i,0])**2 + (y - coords[i,1])**2 + (z - coords[i,2])**2))

  return density

density = interpolate_density_map(coords, weights, 0.1)

# Save the density map to a file
np.save('density.npy', density)


    





