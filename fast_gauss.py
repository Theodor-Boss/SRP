import numpy as np
import GPy
import matplotlib.pyplot as plt


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


# Load data (replace with your actual data loading method)
data = load_data()

start, stop = 4, 169
mask = (data[0][0] > start) & (data[0][0] < stop)
x_data, y_data = data[0][0][mask], data[0][1][mask]

X = x_data.reshape(-1, 1)
y = y_data.reshape(-1, 1)

# Define the kernel (RBF kernel for simplicity)
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

# Define a sparse GP model (with inducing points, Z)
Z = np.random.rand(10, 1) * (X.max() - X.min()) + X.min()  # 10 inducing points chosen randomly
model = GPy.models.SparseGPRegression(X, y, kernel, Z=Z)

# Optimize the model
model.optimize(messages=True)

# Make predictions on a fine grid
X_pred = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)
y_pred, y_var = model.predict(X_pred)

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='r', label='Observations', s=3)
plt.plot(X_pred, y_pred, color='b', label='Prediction')
plt.fill_between(X_pred.ravel(), y_pred.ravel() - 1.96 * np.sqrt(y_var).ravel(),
                 y_pred.ravel() + 1.96 * np.sqrt(y_var).ravel(),
                 alpha=0.2, color='b', label='95% confidence interval')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Sparse Gaussian Process Regression (GPy)')
plt.legend()
plt.show()
