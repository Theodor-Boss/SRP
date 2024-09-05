import numpy as np
import matplotlib.pyplot as plt





def F(x, y):
    x_out = np.exp(np.sin((2 * x + 2.5) ** 2)) / (2 * x + 2.5) * 0.7 - 0.9
    y_out = 1 / (0.7 * y + 1) + (0.7 * y + 1) ** 2 - 2.8
    return x_out, y_out

'''x = np.linspace(-1, 1, 1000)
y, z = F(x, x)

plt.plot(x, z)
plt.show()'''


# Define the dimensions of the vector field
x = np.linspace(-8, 8, 20)
y = np.linspace(-8, 8, 20)

# Create a grid of points
X, Y = np.meshgrid(x, y)

# Define the components of the vector field
scale_of_vectors = 100
X_dot, Y_dot = F(X, Y)

# akser:
plt.axhline(0, color='black')
plt.axvline(0, color='black')

# Plot the vectors
plt.quiver(X, Y, X_dot, Y_dot, scale = scale_of_vectors)


# Set labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vektorfelt')

plt.show()




