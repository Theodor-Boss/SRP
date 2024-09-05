"""
### 1 ###
Denne fil starter en animation af et fysisk pendul og viser samtidig faserummet. Den samme animation kan findes som en GIF på min hjemmeside:
https://theodormyhre.dk/projekt3/#evolution
"""
import numpy as np
import matplotlib.pyplot as plt


def get_theta_double_dot(theta, theta_dot):
    return -mu * theta_dot - mgdI * np.sin(theta)


def theta(t, delta_t):
    theta = THETA_0
    theta_dot = THETA_DOT_0
    for time in np.arange(0, t, delta_t):
        theta_double_dot = get_theta_double_dot(theta, theta_dot)
        theta += theta_dot * delta_t
        theta_dot += theta_double_dot * delta_t
    return theta, theta_dot


def theta_past(t, delta_t):  # Bruges til at regne baglæns i tiden
    theta = THETA_0
    theta_dot = THETA_DOT_0
    for time in np.arange(0, t, delta_t):
        theta_double_dot = get_theta_double_dot(theta, theta_dot)
        theta -= theta_dot * delta_t
        theta_dot -= theta_double_dot * delta_t
    return theta, theta_dot


mu = 0.3
mgdI = 3
delta_t = 0.00001

THETA_0 = 3.14159 - 2 * np.pi
THETA_DOT_0 = 0

THETA_0, THETA_DOT_0 = theta_past(10, delta_t)

ts = np.linspace(0, 25, 400)
thetas = np.empty_like(ts)
omegas = np.empty_like(ts)

delta_t = 0.01

for i, t in enumerate(ts):
    thetas[i], omegas[i] = theta(t, delta_t)


min_x, max_x = np.min(thetas) - 1, np.max(thetas) + 1
min_y, max_y = np.min(omegas) - 1, np.max(omegas) + 1

x = np.linspace(min_x, max_x, 49)
y = np.linspace(min_y, max_y, 49)
X, Y = np.meshgrid(x, y)
scale_of_vectors = 100
X_dot = Y
Y_dot = get_theta_double_dot(X, Y)

cmap = plt.cm.viridis  # Choose any colormap you like
colors = cmap(np.linspace(0, 1, len(x)))

fig = plt.figure(figsize=(10, 5))

# fig, ax = plt.subplots()

gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.0)

# Create the subplots using the gridspec
ax0 = fig.add_subplot(gs[0, 0])  # ax[0] takes up 3 parts
ax1 = fig.add_subplot(gs[0, 1])  # ax[1] takes up 1 part

fig.suptitle(
    'Pendulets Faserum/Vektorfelt for Pendulet', fontsize=20, fontweight='bold'
)

for i, t in enumerate(ts):
    ax0.cla()
    ax0.quiver(
        X, Y, X_dot, Y_dot, scale=scale_of_vectors, color=colors, alpha=0.7
    )
    ax0.plot(thetas[:i+1], omegas[:i+1], color="red")
    ax0.scatter(thetas[i], omegas[i], c="red")

    ax0.axhline(0, color='black')
    ax0.axvline(0, color='black')
    ax0.set_xlim(min_x, max_x)
    ax0.set_ylim(min_y, max_y)

    ax0.set_xlabel('Vinkel')
    ax0.set_ylabel('Vinkelhastighed')

    ax1.cla()
    ax1.plot([0, np.sin(thetas[i])], [0, -np.cos(thetas[i])], color='blue', linewidth=3)
    lod = plt.Circle(
        (np.sin(thetas[i]), -np.cos(thetas[i])), 0.15, color='blue', fill=True
    )
    ax1.add_patch(lod)
    ax1.set_aspect('equal')
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.axis("off")

    plt.pause(0.1)
    # plt.savefig(f"GIF2/image{i}.png")

plt.show()
