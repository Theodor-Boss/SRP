"""
### 2 ###
Hint: Alt+Z i VS Code (Windows)
Denne animation demonstrerer at mit måle-udstyr ikke var kalibret. Der forventes en vinkelhastighed på nul, når pendulet står stille. Det står stille i slutningen (~170-190s) og antiderivativen svarende til vinklen burde altså være en vandret linje, men der ses en tydelig hældning på linjen, når der zoomes ind. Hældningen på denne linje svarer til det vinklehastigheden er offsat med.
"""
import numpy as np
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


antiderivative1 = antiderivative(ts1, omegas1)

mask_line = (ts1 > 170) & (ts1 < 190)


fig, ax = plt.subplots()
plt.plot(ts1, antiderivative1)
plt.plot(ts1[mask_line], antiderivative1[mask_line], color="red", label="Sted der burde være vandret")
plt.title("Antiderivativen svarende til vinklen")
plt.legend()

def logarithmic(xs, y0, y1, a):
    return (np.exp(a * xs) - 1) * (y0 - y1) / (1 - np.exp(a)) + y0


frames = 256

a = -7

x_mins = logarithmic(np.linspace(0, 1, frames), -100, 143, a)
x_maxs = logarithmic(np.linspace(0, 1, frames), np.max(ts1) + 100, 198, a)
y_mins = logarithmic(np.linspace(0, 1, frames), np.min(antiderivative1) - 100, -329.45, a)
y_maxs = logarithmic(np.linspace(0, 1, frames), np.max(antiderivative1) + 100, -328.92, a)


# Animation
for i in range(frames):
    ax.set_xlim(x_mins[i], x_maxs[i])
    ax.set_ylim(y_mins[i], y_maxs[i])
    plt.pause(0.05)
    # plt.savefig(f"GIF1/image{i}.png")  # For at gemme billedet og lave en GIF
plt.show()
