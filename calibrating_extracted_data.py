"""
### 4 ###
Dette script tager de udvundne data for pendulet og kalibrerer
vinkelhastigheden. Dette gøres ved at tilpasse en lineær model til
en stamfunktion for vinkelhastigheden for den del af dataene, hvor pendulet er i
hvile. Hældningen af ​​denne model trækkes derefter fra vinkelhastigheden
for hele datasættet, for at få de "ægte" vinkelhastigheder.

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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


def calibrate_omega(ts, omegas, hvile):
    """
    "hvile" er en tuple som angiver
    tidsintervallet, pendulet er i hvile.
    """
    stamfunktion = antiderivative(ts, omegas)
    mask_hvile = (ts >= hvile[0]) & (ts < hvile[1])
    X_train, X_test, y_train, y_test = train_test_split(
        ts[mask_hvile].reshape(-1, 1), stamfunktion[mask_hvile],
        test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    slope = model.coef_[0]
    kalibreret_omegas = omegas - slope
    return kalibreret_omegas, r2, mse


# Brugt til at finde frem til hvile-intervallerne som er hardcodet nedenfor
"""
plt.plot(ts1, omegas1)
plt.show()

plt.plot(ts2, omegas2)
plt.show()

plt.plot(ts3, omegas3)
plt.show()

plt.plot(ts4, omegas4)
plt.show()

plt.plot(ts5, omegas5)
plt.show()
sys.exit()
"""

kalibreret_omegas1, r2_1, mse1 = calibrate_omega(ts1, omegas1, (170, 192))
kalibreret_omegas2, r2_2, mse2 = calibrate_omega(ts2, omegas2, (179, 211))
kalibreret_omegas3, r2_3, mse3 = calibrate_omega(ts3, omegas3, (167, 199))
kalibreret_omegas4, r2_4, mse4 = calibrate_omega(ts4, omegas4, (162, 291))
kalibreret_omegas5, r2_5, mse5 = calibrate_omega(ts5, omegas5, (170, 251))


# Gemmer de kalibrerede vinkelhastigheder til den videre databehandling
calibrated_omegas1 = "calibrated_omegas1.npz"
calibrated_omegas2 = "calibrated_omegas2.npz"
calibrated_omegas3 = "calibrated_omegas3.npz"
calibrated_omegas4 = "calibrated_omegas4.npz"
calibrated_omegas5 = "calibrated_omegas5.npz"

np.savez(calibrated_omegas1, ts=ts1, calibrated_omegas=kalibreret_omegas1)
np.savez(calibrated_omegas2, ts=ts2, calibrated_omegas=kalibreret_omegas2)
np.savez(calibrated_omegas3, ts=ts3, calibrated_omegas=kalibreret_omegas3)
np.savez(calibrated_omegas4, ts=ts4, calibrated_omegas=kalibreret_omegas4)
np.savez(calibrated_omegas5, ts=ts5, calibrated_omegas=kalibreret_omegas5)


# Statistiske deskriptorer:
"""
print(r2_1)
print(mse1)
print()

print(r2_2)
print(mse2)
print()

print(r2_3)
print(mse3)
print()

print(r2_4)
print(mse4)
print()

print(r2_5)
print(mse5)
print()
"""


# Koden nedenfor plotter stamfunktioner til hver af de fem seriers kalibrerede vinkelhastigheder.

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ts1, antiderivative(ts1, kalibreret_omegas1))
ax.axhline(0, color="black")
ax.set_xlabel("Tid")
ax.set_title("En stamfunktion til kalibreret vinkelhastighed 1")

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ts2, antiderivative(ts2, kalibreret_omegas2))
ax.axhline(0, color="black")
ax.set_xlabel("Tid")
ax.set_title("En stamfunktion til kalibreret vinkelhastighed 2")

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ts3, antiderivative(ts3, kalibreret_omegas3))
ax.axhline(0, color="black")
ax.set_xlabel("Tid")
ax.set_title("En stamfunktion til kalibreret vinkelhastighed 3")

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ts4, antiderivative(ts4, kalibreret_omegas4))
ax.axhline(0, color="black")
ax.set_xlabel("Tid")
ax.set_title("En stamfunktion til kalibreret vinkelhastighed 4")

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ts5, antiderivative(ts5, kalibreret_omegas5))
ax.axhline(0, color="black")
ax.set_xlabel("Tid")
ax.set_title("En stamfunktion til kalibreret vinkelhastighed 5")

plt.show()
