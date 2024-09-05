"""
### 2 ###
Hint: Alt+Z i VS Code (Windows)
Denne animation demonstrerer at mit måle-udstyr ikke var kalibret. Der forventes en vinkelhastighed på nul, når pendulet står stille. Det står stille i slutningen (~170-190s). Det ses, at vinkelhastigheden er tæt på nul men ikke tæt nok til min smag (det kan have betydning for den videre databehandling)
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


extracted_data1 = "extracted_data1.npz"
extracted_data2 = "extracted_data2.npz"
extracted_data3 = "extracted_data3.npz"
extracted_data4 = "extracted_data4.npz"
extracted_data5 = "extracted_data5.npz"

with np.load(extracted_data1) as data:
    ts1 = data["ts"]
    omegas1 = data["omegas"]

with np.load(extracted_data2) as data:
    ts2 = data["ts"]
    omegas2 = data["omegas"]

with np.load(extracted_data3) as data:
    ts3 = data["ts"]
    omegas3 = data["omegas"]

with np.load(extracted_data4) as data:
    ts4 = data["ts"]
    omegas4 = data["omegas"]

with np.load(extracted_data5) as data:
    ts5 = data["ts"]
    omegas5 = data["omegas"]


def antiderivative(xs, ys):
    h = np.diff(xs)
    a1 = ys[:-1]
    a2 = ys[1:]
    trapez_sum = 0.5 * (a1 + a2) * h
    integrals = np.concatenate(([0], np.cumsum(trapez_sum)))
    return integrals  # dvs. det ubestemte integral for y(x) gennem (xs[0],0)


mask_rest = ts1 < 170
mask_line = (ts1 >= 170) & (ts1 < 190)
mask_noise = ts1 >= 190

def logarithmic(xs, y0, y1, a):
    return (np.exp(a * xs) - 1) * (y0 - y1) / (1 - np.exp(a)) + y0


frames = 256

a = -7

x_mins = logarithmic(np.linspace(0, 1, frames), np.min(ts1) - 10, 158, a)
x_maxs = logarithmic(np.linspace(0, 1, frames), np.max(ts1), 198, a)
y_mins = logarithmic(np.linspace(0, 1, frames), np.min(omegas1) - 5, -0.011, a)
y_maxs = logarithmic(np.linspace(0, 1, frames), np.max(omegas1) + 5, 0.018, a)


fig, ax = plt.subplots(figsize=(10, 6))

plt.plot(ts1[mask_rest], omegas1[mask_rest], color='C0')
plt.plot(ts1[mask_line], omegas1[mask_line], color="red", label="Sted der burde være nul")
plt.plot(ts1[mask_noise], omegas1[mask_noise], color="magenta", label="Rystelser efter")

ax.axhline(0, color='black')
ax.set_xlabel("Tid")
ax.set_ylabel("Vinkelhastighed")
plt.title("Ikkekalibrerede vinkelhastighed")
plt.legend(loc='lower center')

# Animation
for i in range(frames):
    ax.set_xlim(x_mins[i], x_maxs[i])
    ax.set_ylim(y_mins[i], y_maxs[i])
    plt.pause(0.05)
    # plt.savefig(f"GIF3/image{i}.png")  # For at gemme billederne og lave en GIF

plt.show()
