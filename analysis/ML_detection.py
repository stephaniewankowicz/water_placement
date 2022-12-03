import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D array of weighted values

x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
z = np.linspace(0, 10, 100)

X, Y, Z = np.meshgrid(x, y, z)

W = np.exp(-((X-5)**2 + (Y-5)**2 + (Z-5)**2)/2)

# Find the maximum likelihood in 3D

max_likelihood = np.where(W == np.amax(W))

print(max_likelihood)

# Plot the 3D array of weighted values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
