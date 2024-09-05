import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

mu = 0.4 # 0.021375048021624925
mgdI = 3 #2.074419135114389


def get_theta_double_dot(theta, theta_dot):
    return -mu * theta_dot - mgdI * np.sin(theta)


#plt.figure(figsize=(17, 13), dpi=200) #, dpi=400  #¤#

# Define the dimensions of the vector field
x = np.linspace(-10, 10, 91)
y = np.linspace(-8, 8, 41)

# Create a grid of points
X, Y = np.meshgrid(x, y)

# Define the components of the vector field
scale_of_vectors = 110
X_dot = Y
Y_dot = get_theta_double_dot(X, Y)

# akser:
plt.axhline(0, color='black')
plt.axvline(0, color='black')

'''# Create a fading colormap
#cmap = plt.cm.cividis_r  # Choose any colormap you like
cmap = plt.cm.Greys  # Choose any colormap you like
colors = cmap(np.linspace(0, 1, len(x)))'''

# Create a fading colormap
n = 256
colors = plt.cm.Greys(np.linspace(0, 1, n))
colors[:, 3] = np.linspace(0, 1, n)  # Set alpha values to fade from 0 to 1
custom_cmap = LinearSegmentedColormap.from_list("custom_Greys", colors)

# Normalize Y values to the range [0, 1] for color mapping
Y_normalized = (Y - np.min(y)) / (np.max(y) - np.min(y))


# Create a fading colormap
cmap = plt.cm.viridis  # Choose any colormap you like
colors = cmap(np.linspace(0, 1, len(x)))

# Plot quiver plot with fading colors
plt.quiver(X, Y, X_dot, Y_dot, scale=scale_of_vectors, color=colors, alpha=0.7)  # Adjust alpha as needed




# Plot the vectors
#plt.quiver(X, Y, X_dot, Y_dot, cmap=custom_cmap, alpha=0.3)
#plt.quiver(X, Y, X_dot, Y_dot, cmap='viridis', alpha=0.3)

#plt.xlim(-3, 3)
#plt.axis('off')

# Set labels and title
plt.xlabel('Vinkel')
plt.ylabel('Vinkelhastighed')
plt.title('Vektorfelt for penduls mulige tilstande')

plt.show()




