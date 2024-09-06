"""
### 3 ###
Hint: Alt+Z i VS Code (Windows)
Endnu en animation der demonstrerer at mit måle-udstyr ikke var kalibret. Der forventes en vinkelhastighed på nul, når pendulet står stille. Det står stille i slutningen (~170-190s) og antiderivativen svarende til vinklen burde altså være en vandret linje, men der ses en tydelig hældning på linjen, når der zoomes ind. Hældningen på denne linje svarer til det, vinklehastigheden er offsat med. Der foretages en lineær regression på dette stykke for at få hældningen.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys


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

mask_rest = ts1 < 170
mask_line = (ts1 >= 170) & (ts1 < 190)
mask_noise = ts1 >= 190
mask_start = ts1 < 130

X_train, X_test, y_train, y_test = train_test_split(ts1[mask_line].reshape(-1, 1), antiderivative1[mask_line], test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.6f}")
print(f"R-squared: {r2:.6f}")


y_extrapolate = model.predict(ts1.reshape(-1, 1))

# Bruges til at indstille hvor der zoomes til
"""
def on_key_press(event):
    if event.key == 'a':
        x_lims = plt.gca().get_xlim()
        y_lims = plt.gca().get_ylim()
        print(f"x-axis limits: {x_lims}")
        print(f"y-axis limits: {y_lims}")
"""

fig, ax = plt.subplots(figsize=(10, 6))

# PLOTTING
ax.plot(ts1[mask_rest], antiderivative1[mask_rest], color='C0')
ax.plot(ts1[mask_noise], antiderivative1[mask_noise], color="C0")
ax.plot(ts1[mask_line], antiderivative1[mask_line], color="red")
ax.plot(ts1[mask_rest], y_extrapolate[mask_rest], "--", color="gray", label="Ekstrapolering af lineær regression på røde stykke")
ax.plot(ts1[mask_noise], y_extrapolate[mask_noise], "--", color="gray")
ax.set_xlabel("Tid")
ax.set_title("Stamfunktion til ikke-kalibreret vinkelhastighed")
ax.legend(loc='upper right')


def log_scaling(ts, a):
    # return (np.exp(a * ts) - 1) / (np.exp(a) - 1)
    return (np.sin(ts * np.pi / 2)) ** 2 * (1+(a*(ts-1))**2)


def linear_transform(ll0, ur0, ll1, ur1):
    ll_vec = np.array([ll1[0] - ll0[0], ll1[1] - ll0[1]])
    ur_vec = np.array([ur1[0] - ur0[0], ur1[1] - ur0[1]])
    return ll_vec, ur_vec


ll0 = (np.float64(-9.881582498550415), np.float64(-348.9079746508174))
ur0 = (np.float64(207.51323246955872), np.float64(16.77612238700534))
ll1 = (np.float64(20.78964566988444), np.float64(-332.40978058810117))
ur1 = (np.float64(202.7565775170868), np.float64(-326.8162434812083))

dll, dur = linear_transform(ll0, ur0, ll1, ur1)

ts = log_scaling(np.linspace(0, 1, 150), 1.65)


x_mins = ll0[0] + dll[0] * ts
x_maxs = ur0[0] + dur[0] * ts
y_mins = ll0[1] + dll[1] * ts
y_maxs = ur0[1] + dur[1] * ts


frames = len(ts)

# Animation
for i in range(frames):
    ax.set_xlim(x_mins[i], x_maxs[i])
    ax.set_ylim(y_mins[i], y_maxs[i])
    plt.pause(0.01)
    # plt.savefig(f"GIF4/image{i}.png")  # For at gemme billedet og lave en GIF

plt.show()
