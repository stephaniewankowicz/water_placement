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


import numpy as np
+import matplotlib.pyplot as plt
+
+#define the function
+def likelihood(x,y,x_array,y_array,sigma):
+    #define the likelihood function
+    #x and y are the coordinates of the point we are testing
+    #x_array and y_array are the coordinates of the points we are using to test
+    #sigma is the standard deviation of the gaussian
+    #returns the likelihood of the point existing
+    #the likelihood is the sum of the gaussian of each point in the array
+    #the gaussian is defined as 1/(sigma*sqrt(2*pi))*exp(-(x-x_array[i])**2/(2*sigma**2))
+    #the sum is over all the points in the array
+    #the likelihood is the product of the gaussians
+    #the product is over all the points in the array
