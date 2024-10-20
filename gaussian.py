import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import sys


def load_data():
    input_files = [
        "calibrated_omegas1.npz",
        "calibrated_omegas2.npz",
        "calibrated_omegas3.npz",
        "calibrated_omegas4.npz",
        "calibrated_omegas5.npz"
    ]

    data = []
    for file in input_files:
        with np.load(file) as npz_data:
            data.append((npz_data["ts"], npz_data["calibrated_omegas"]))
    return data


data = load_data()


start, stop = 4, 169

mask = (data[0][0] > start) & (data[0][0] < stop)

x_data, y_data = data[0][0][mask], data[0][1][mask]


X = x_data.reshape(-1, 1)
y = y_data

# Define a flexible kernel
k1 = C(1.0, (1e-3, 1e3))
k2 = Matern(length_scale=1.0, nu=1.5)  # Matern kernel is flexible for various smoothness levels
k3 = WhiteKernel(noise_level=1.0)
kernel = k1 * k2 + k3

# Create and fit the Gaussian Process Regression model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, normalize_y=True, alpha=1e-2)
gp.fit(X, y)

# Make predictions on a fine grid
X_pred = np.linspace(np.min(x_data), np.max(x_data), 10000).reshape(-1, 1)
y_pred, sigma = gp.predict(X_pred, return_std=True)

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='r', label='Observations', s=3)
plt.plot(X_pred, y_pred, color='b', label='Prediction')
"""plt.fill_between(X_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma,
                 alpha=0.2, color='b', label='95% confidence interval')"""
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gaussian Process Regression')
plt.legend()
plt.show()

"""# Compute and plot the first derivative
X_der = np.linspace(np.min(x_data), np.max(x_data), 100).reshape(-1, 1)
y_der, _ = gp.predict(X_der, return_std=False)
dy_dx = np.gradient(y_der.ravel(), X_der.ravel())

plt.figure(figsize=(10, 6))
plt.plot(X_der, dy_dx, label='First Derivative')
plt.xlabel('X')
plt.ylabel('dy/dx')
plt.title('First Derivative of GP Regression')
plt.legend()
plt.show()"""
