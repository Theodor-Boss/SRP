"""
### 5 ###
Denne fil beregner vinklerne ved at integrere
de kalibrerede vinkelhastigheder.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


input_file1 = "calibrated_omegas1.npz"
input_file2 = "calibrated_omegas2.npz"
input_file3 = "calibrated_omegas3.npz"
input_file4 = "calibrated_omegas4.npz"
input_file5 = "calibrated_omegas5.npz"

with np.load(input_file1) as data:
    ts1 = data["ts"]
    kalibreret_omegas1 = data["calibrated_omegas"]

with np.load(input_file2) as data:
    ts2 = data["ts"]
    kalibreret_omegas2 = data["calibrated_omegas"]

with np.load(input_file3) as data:
    ts3 = data["ts"]
    kalibreret_omegas3 = data["calibrated_omegas"]

with np.load(input_file4) as data:
    ts4 = data["ts"]
    kalibreret_omegas4 = data["calibrated_omegas"]

with np.load(input_file5) as data:
    ts5 = data["ts"]
    kalibreret_omegas5 = data["calibrated_omegas"]


def antiderivative(xs, ys):
    h = np.diff(xs)
    a1 = ys[:-1]
    a2 = ys[1:]
    trapez_sum = 0.5 * (a1 + a2) * h
    integrals = np.concatenate(([0], np.cumsum(trapez_sum)))
    return integrals


def advance_average(xs, ys, interval):
    """
    "hvile" er en tuple som angiver
    tidsintervallet, pendulet er i hvile.
    """
    stamfunktion = antiderivative(xs, ys)
    mask_interval = (interval[0] <= xs) & (xs <= interval[1])
    X_train, X_test, y_train, y_test = train_test_split(
        xs[mask_interval].reshape(-1, 1), stamfunktion[mask_interval],
        test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # slope = model.coef_[0]
    # kalibreret_omegas = ys - slope
    # return kalibreret_omegas, r2, mse
    return model


# 0) Hvile-intervallerne:
hvile1 = (170, 192)
hvile2 = (179, 211)
hvile3 = (167, 199)
hvile4 = (162, 291)
hvile5 = (170, 251)

# 1) Vinkel...ish:
stamfunktion1 = antiderivative(ts1, kalibreret_omegas1)
stamfunktion2 = antiderivative(ts2, kalibreret_omegas2)
stamfunktion3 = antiderivative(ts3, kalibreret_omegas3)
stamfunktion4 = antiderivative(ts4, kalibreret_omegas4)
stamfunktion5 = antiderivative(ts5, kalibreret_omegas5)

# 2) Lineær regression - Stamfunktion til Vinkel...ish:
"""lin_reg1 = advance_average(ts1, stamfunktion1, hvile1)
lin_reg2 = advance_average(ts2, stamfunktion2, hvile2)
lin_reg3 = advance_average(ts3, stamfunktion3, hvile3)
lin_reg4 = advance_average(ts4, stamfunktion4, hvile4)
lin_reg5 = advance_average(ts5, stamfunktion5, hvile5)"""

# 3) Stamfunktion til Stamfunktion:
"""antiderivative_stamfunktion1 = antiderivative(ts1, stamfunktion1)
antiderivative_stamfunktion2 = antiderivative(ts2, stamfunktion2)
antiderivative_stamfunktion3 = antiderivative(ts3, stamfunktion3)
antiderivative_stamfunktion4 = antiderivative(ts4, stamfunktion4)
antiderivative_stamfunktion5 = antiderivative(ts5, stamfunktion5)"""

# 4) Vi finder stamfunktioner for hvile-intervallerne:
"""mask_hvile1 = (ts1 >= hvile1[0]) & (ts1 <= hvile1[1])
mask_hvile2 = (ts2 >= hvile2[0]) & (ts2 <= hvile2[1])
mask_hvile3 = (ts3 >= hvile3[0]) & (ts3 <= hvile3[1])
mask_hvile4 = (ts4 >= hvile4[0]) & (ts4 <= hvile4[1])
mask_hvile5 = (ts5 >= hvile5[0]) & (ts5 <= hvile5[1])"""

# ektralopering af lineær regression - Stamfunktion til stamfunktion:
# xs_lin_reg1 = np.linspace(np.min(ts1), np.max(ts1), 100)
"""ys_lin_reg1 = lin_reg1.predict(ts1.reshape(-1, 1))"""

# 5) Vi finder offset:
offset1 = advance_average(ts1, stamfunktion1, hvile1).coef_[0]
offset2 = advance_average(ts2, stamfunktion2, hvile2).coef_[0]
offset3 = advance_average(ts3, stamfunktion3, hvile3).coef_[0]
offset4 = advance_average(ts4, stamfunktion4, hvile4).coef_[0]
offset5 = advance_average(ts5, stamfunktion5, hvile5).coef_[0]

# 6) Ægte vinkel:
thetas1 = stamfunktion1 - offset1
thetas2 = stamfunktion2 - offset2
thetas3 = stamfunktion3 - offset3
thetas4 = stamfunktion4 - offset4
thetas5 = stamfunktion5 - offset5


fig, ax = plt.subplots()
ax.plot(ts1, thetas1, color="C0")
ax.set_xlabel("Tid")
ax.set_ylabel("Vinkel")
fig.suptitle("Vinkel 1")
plt.show()

fig, ax = plt.subplots()
ax.plot(ts2, thetas2, color="C0")
ax.set_xlabel("Tid")
ax.set_ylabel("Vinkel")
fig.suptitle("Vinkel 2")
plt.show()

fig, ax = plt.subplots()
ax.plot(ts3, thetas3, color="C0")
ax.set_xlabel("Tid")
ax.set_ylabel("Vinkel")
fig.suptitle("Vinkel 3")
plt.show()

fig, ax = plt.subplots()
ax.plot(ts4, thetas4, color="C0")
ax.set_xlabel("Tid")
ax.set_ylabel("Vinkel")
fig.suptitle("Vinkel 4")
plt.show()

fig, ax = plt.subplots()
ax.plot(ts5, thetas5, color="C0")
ax.set_xlabel("Tid")
ax.set_ylabel("Vinkel")
fig.suptitle("Vinkel 5")
plt.show()
