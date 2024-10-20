"""
### 8 ###
"""
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
import sys
# from scipy import stats
import statsmodels.api as sm


input_file1 = "calibrated_omegas1.npz"
input_file2 = "calibrated_omegas2.npz"
input_file3 = "calibrated_omegas3.npz"
input_file4 = "calibrated_omegas4.npz"
input_file5 = "calibrated_omegas5.npz"


with np.load(input_file1) as data:
    ts1 = data["ts"]
    kalibreret_omegas1 = data["calibrated_omegas"]

"""
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
"""


def antiderivative(xs, ys):
    h = np.diff(xs)
    a1 = ys[:-1]
    a2 = ys[1:]
    trapez_sum = 0.5 * (a1 + a2) * h
    integrals = np.concatenate(([0], np.cumsum(trapez_sum)))
    return integrals


def cubic_bezier(A, B, x):
    xa, ya, dydxa = A
    xb, yb, dydxb = B
    dx = xa - xb
    dy = ya - yb

    term1 = ya + dy / dx * (x - xa)
    term2a = ((dydxa + dydxb) / dx - 2 * dy / dx ** 2)
    term2b = (x - xa) * ((x - xa) ** 2 / dx - dx)
    term2 = term2a * term2b
    term3a = (x - xa) * (x - xb)
    term3b = ((2 * dydxa + dydxb) / dx - 3 * dy / dx ** 2)
    term3 = term3a * term3b

    return term1 + term2 + term3


"""arr = np.array([[4, 4], [7, 10]])

det = np.linalg.det(arr)
print(det)
sys.exit()"""


def point_adjuster(x_target, left_data, right_data, left, right):
    left_x_data, left_y_data = left_data
    right_x_data, right_y_data = right_data

    left_x, left_y, left_dydx = left
    right_x, right_y, right_dydx = right

    def R(x, x_neighbor, y_neighbor, dydx_neighbor, x_target=x_target):
        R = (
            (-(x - x_neighbor)) / (x_neighbor - x_target)
            + 2 * (x - x_neighbor) / (x_neighbor - x_target)**2 * ((x - x_neighbor)**2 / (x_neighbor - x_target) - (x_neighbor - x_target))
            + 3 * (x - x_neighbor) * (x - x_target) / (x_neighbor - x_target)**2
        )
        return R

    def S(x, x_neighbor, y_neighbor, dydx_neighbor, x_target=x_target):
        S = (
            (x - x_neighbor) / (x_neighbor - x_target) * ((x - x_neighbor)**2 / (x_neighbor - x_target) - (x_neighbor - x_target))
            + (x - x_neighbor) * (x - x_target) / (x_neighbor - x_target)
        )
        return S

    def T(x, x_neighbor, y_neighbor, dydx_neighbor, x_target=x_target):
        T = (
            y_neighbor
            + (y_neighbor * (x - x_neighbor)) / (x_neighbor - x_target)
            + (dydx_neighbor / (x_neighbor - x_target) - 2 * y_neighbor / (x_neighbor - x_target)**2) * (x - x_neighbor) * ((x - x_neighbor)**2 / (x_neighbor - x_target) - (x_neighbor - x_target))
            + (x - x_neighbor) * (x - x_target) * (2 * dydx_neighbor / (x_neighbor - x_target) - 3 * y_neighbor / (x_neighbor - x_target)**2)
        )
        return T

    Da1 = np.sum(R(left_x_data, left_x, left_y, left_dydx)**2) + np.sum(R(right_x_data, right_x, right_y, right_dydx)**2)
    Da2 = np.sum(R(left_x_data, left_x, left_y, left_dydx) * S(left_x_data, left_x, left_y, left_dydx)) + np.sum(R(right_x_data, right_x, right_y, right_dydx) * S(right_x_data, right_x, right_y, right_dydx))
    Db1 = Da2  # copy?
    Db2 = np.sum(S(left_x_data, left_x, left_y, left_dydx)**2) + np.sum(S(right_x_data, right_x, right_y, right_dydx)**2)
    Dya1 = np.sum((T(left_x_data, left_x, left_y, left_dydx) - left_y_data) * S(left_x_data, left_x, left_y, left_dydx)) + np.sum((T(right_x_data, right_x, right_y, right_dydx) - right_y_data) * S(right_x_data, right_x, right_y, right_dydx))
    Dya2 = np.sum((T(left_x_data, left_x, left_y, left_dydx) - left_y_data) * R(left_x_data, left_x, left_y, left_dydx)) + np.sum((T(right_x_data, right_x, right_y, right_dydx) - right_y_data) * R(right_x_data, right_x, right_y, right_dydx))
    Dyb1 = Db2
    Dyb2 = Da2
    Ddydxa1 = Da2
    Ddydxa2 = Da1
    Ddydxb1 = Dya1
    Ddydxb2 = Dya2

    arrD = np.array([[Da1, Da2], [Db1, Db2]])
    arrDy = np.array([[Dya1, Dya2], [Dyb1, Dyb2]])
    arrDdydx = np.array([[Ddydxa1, Ddydxa2], [Ddydxb1, Ddydxb2]])

    D = np.linalg.det(arrD)
    Dy = np.linalg.det(arrDy)
    Ddydx = np.linalg.det(arrDdydx)

    y = Dy / D
    dydx = Ddydx / D
    return y, dydx


def initialize(xs, ys, spline_xs):
    spline_ys = np.empty_like(spline_xs)
    spline_dydxs = np.empty_like(spline_xs)
    for i in range(len(spline_xs)):
        X = xs.copy()
        y = ys.copy()
        sigma = 0.1

        weights = np.exp(-((spline_xs[i] - xs) / sigma) ** 2)
        X = sm.add_constant(X)  # If you're doing multiple regression (with more than one predictor variable), results.params will contain all the coefficients in the order they appear in your X matrix.
        model = sm.WLS(y, X, weights=weights)
        results = model.fit()
        intercept = results.params[0]
        slope = results.params[1]

        spline_ys[i] = slope * spline_xs[i] + intercept
        spline_dydxs[i] = slope

        # Print the results
        # print(results.summary())

    return spline_ys, spline_dydxs


ts = np.random.normal(ts1, 0.05)
os = np.random.normal(kalibreret_omegas1, 0.25)
# ts = np.random.normal(ts1, 0.0005)
# os = np.random.normal(kalibreret_omegas1, 0.0025)

fig, ax = plt.subplots()

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

t_width = 0.8

spline_ts = np.arange(np.min(ts), np.max(ts) + t_width, t_width)
spline_ts -= (spline_ts[-1] - np.max(ts)) / 2

spline_os, spline_dodts = initialize(ts, os, spline_ts)


ax.plot(ts, os, "ro")

for spline_x in spline_ts:
    ax.axvline(spline_x)


TARGET = np.argmin(np.abs(100.0 - spline_ts))

spline_os[TARGET] = 0.0
spline_dodts[TARGET] = -5.0


ax.plot(spline_ts, spline_os, 'o', color="black")

for i in range(len(spline_ts) - 1):
    bezier_xs = np.linspace(spline_ts[i], spline_ts[i+1], 100)
    A = (spline_ts[i], spline_os[i], spline_dodts[i])
    B = (spline_ts[i+1], spline_os[i+1], spline_dodts[i+1])
    bezier_ys = cubic_bezier(A, B, bezier_xs)
    ax.plot(bezier_xs, bezier_ys, color="black")

ax.scatter(spline_ts[TARGET], spline_os[TARGET], c="green", s=1000)

ax.set_xlim(spline_ts[TARGET] - 1.5 * t_width, spline_ts[TARGET] + 1.5 * t_width)
mask_T2 = (spline_ts[TARGET] - 1.5 * t_width < ts) & (ts <= spline_ts[TARGET+1] + 1.5 * t_width)
min_y, max_y = np.min(os[mask_T2]), np.max(os[mask_T2])
ax.set_ylim(
    min_y - (max_y - min_y) * 0.03,
    max_y + (max_y - min_y) * 0.03
)


FIT = 0

mask_Tpos = (spline_ts[TARGET] < ts) & (ts <= spline_ts[TARGET+1])

ax.plot(ts[mask_Tpos], os[mask_Tpos], color="magenta")

for _, (x, y_data) in enumerate(zip(ts[mask_Tpos], os[mask_Tpos])):
    Apos = (spline_ts[TARGET], spline_os[TARGET], spline_dodts[TARGET])
    Bpos = (spline_ts[TARGET+1], spline_os[TARGET+1], spline_dodts[TARGET+1])
    y_model = cubic_bezier(Apos, Bpos, x)
    ax.plot([x, x], [y_data, y_model], color="yellow", linewidth=3)
    # print((y_data - y_model) ** 2)
    FIT += (y_data - y_model) ** 2
    plt.pause(0.1)


mask_Tneg = (spline_ts[TARGET-1] < ts) & (ts <= spline_ts[TARGET])

ax.plot(ts[mask_Tneg], os[mask_Tneg], color="blue")
# print()

for _, (x, y_data) in enumerate(zip(ts[mask_Tneg], os[mask_Tneg])):
    Aneg = (spline_ts[TARGET-1], spline_os[TARGET-1], spline_dodts[TARGET-1])
    Bneg = (spline_ts[TARGET], spline_os[TARGET], spline_dodts[TARGET])
    y_model = cubic_bezier(Aneg, Bneg, x)
    ax.plot([x, x], [y_data, y_model], color="orange", linewidth=3)
    # print((y_data - y_model) ** 1)
    FIT += (y_data - y_model) ** 2
    plt.pause(0.1)

print(f"fit: {FIT}")

mask_T_left = (spline_ts[TARGET-1] < ts) & (ts <= spline_ts[TARGET])
mask_T_right = (spline_ts[TARGET] < ts) & (ts <= spline_ts[TARGET+1])


left_data_points = (ts[mask_T_left], os[mask_T_left])
right_data_points = (ts[mask_T_right], os[mask_T_right])
left_neighbor = (spline_ts[TARGET-1], spline_os[TARGET-1], spline_dodts[TARGET-1])
right_neighbor = (spline_ts[TARGET+1], spline_os[TARGET+1], spline_dodts[TARGET+1])

new_y, new_dydx = point_adjuster(
    spline_ts[TARGET],
    left_data_points,
    right_data_points,
    left_neighbor,
    right_neighbor
)

print(f"new y: {new_y}")
print(f"new dydx: {new_dydx}")

plt.show()









fig, ax = plt.subplots()

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

t_width = 0.8

spline_ts = np.arange(np.min(ts), np.max(ts) + t_width, t_width)
spline_ts -= (spline_ts[-1] - np.max(ts)) / 2

spline_os, spline_dodts = initialize(ts, os, spline_ts)


ax.plot(ts, os, "ro")

for spline_x in spline_ts:
    ax.axvline(spline_x)


TARGET = np.argmin(np.abs(100.0 - spline_ts))

spline_os[TARGET] = new_y
spline_dodts[TARGET] = new_dydx


ax.plot(spline_ts, spline_os, 'o', color="black")

for i in range(len(spline_ts) - 1):
    bezier_xs = np.linspace(spline_ts[i], spline_ts[i+1], 100)
    A = (spline_ts[i], spline_os[i], spline_dodts[i])
    B = (spline_ts[i+1], spline_os[i+1], spline_dodts[i+1])
    bezier_ys = cubic_bezier(A, B, bezier_xs)
    ax.plot(bezier_xs, bezier_ys, color="black")

ax.scatter(spline_ts[TARGET], spline_os[TARGET], c="green", s=1000)

ax.set_xlim(spline_ts[TARGET] - 1.5 * t_width, spline_ts[TARGET] + 1.5 * t_width)
mask_T2 = (spline_ts[TARGET] - 1.5 * t_width < ts) & (ts <= spline_ts[TARGET+1] + 1.5 * t_width)
min_y, max_y = np.min(os[mask_T2]), np.max(os[mask_T2])
ax.set_ylim(
    min_y - (max_y - min_y) * 0.03,
    max_y + (max_y - min_y) * 0.03
)


FIT = 0

mask_Tpos = (spline_ts[TARGET] < ts) & (ts <= spline_ts[TARGET+1])

ax.plot(ts[mask_Tpos], os[mask_Tpos], color="magenta")

for _, (x, y_data) in enumerate(zip(ts[mask_Tpos], os[mask_Tpos])):
    Apos = (spline_ts[TARGET], spline_os[TARGET], spline_dodts[TARGET])
    Bpos = (spline_ts[TARGET+1], spline_os[TARGET+1], spline_dodts[TARGET+1])
    y_model = cubic_bezier(Apos, Bpos, x)
    ax.plot([x, x], [y_data, y_model], color="yellow", linewidth=3)
    # print((y_data - y_model) ** 2)
    FIT += (y_data - y_model) ** 2
    plt.pause(0.1)


mask_Tneg = (spline_ts[TARGET-1] < ts) & (ts <= spline_ts[TARGET])

ax.plot(ts[mask_Tneg], os[mask_Tneg], color="blue")
# print()

for _, (x, y_data) in enumerate(zip(ts[mask_Tneg], os[mask_Tneg])):
    Aneg = (spline_ts[TARGET-1], spline_os[TARGET-1], spline_dodts[TARGET-1])
    Bneg = (spline_ts[TARGET], spline_os[TARGET], spline_dodts[TARGET])
    y_model = cubic_bezier(Aneg, Bneg, x)
    ax.plot([x, x], [y_data, y_model], color="orange", linewidth=3)
    # print((y_data - y_model) ** 1)
    FIT += (y_data - y_model) ** 2
    plt.pause(0.1)

print(f"      new fit: {FIT}")

plt.show()














fig, ax = plt.subplots()

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

t_width = 0.8

spline_ts = np.arange(np.min(ts), np.max(ts) + t_width, t_width)
spline_ts -= (spline_ts[-1] - np.max(ts)) / 2

spline_os, spline_dodts = initialize(ts, os, spline_ts)


ax.plot(ts, os, "ro")

for spline_x in spline_ts:
    ax.axvline(spline_x)


TARGET = np.argmin(np.abs(100.0 - spline_ts))

spline_os[TARGET] = new_y - 0.01
spline_dodts[TARGET] = new_dydx + 0.01


ax.plot(spline_ts, spline_os, 'o', color="black")

for i in range(len(spline_ts) - 1):
    bezier_xs = np.linspace(spline_ts[i], spline_ts[i+1], 100)
    A = (spline_ts[i], spline_os[i], spline_dodts[i])
    B = (spline_ts[i+1], spline_os[i+1], spline_dodts[i+1])
    bezier_ys = cubic_bezier(A, B, bezier_xs)
    ax.plot(bezier_xs, bezier_ys, color="black")

ax.scatter(spline_ts[TARGET], spline_os[TARGET], c="green", s=1000)

ax.set_xlim(spline_ts[TARGET] - 1.5 * t_width, spline_ts[TARGET] + 1.5 * t_width)
mask_T2 = (spline_ts[TARGET] - 1.5 * t_width < ts) & (ts <= spline_ts[TARGET+1] + 1.5 * t_width)
min_y, max_y = np.min(os[mask_T2]), np.max(os[mask_T2])
ax.set_ylim(
    min_y - (max_y - min_y) * 0.03,
    max_y + (max_y - min_y) * 0.03
)


FIT = 0

mask_Tpos = (spline_ts[TARGET] < ts) & (ts <= spline_ts[TARGET+1])

ax.plot(ts[mask_Tpos], os[mask_Tpos], color="magenta")

for _, (x, y_data) in enumerate(zip(ts[mask_Tpos], os[mask_Tpos])):
    Apos = (spline_ts[TARGET], spline_os[TARGET], spline_dodts[TARGET])
    Bpos = (spline_ts[TARGET+1], spline_os[TARGET+1], spline_dodts[TARGET+1])
    y_model = cubic_bezier(Apos, Bpos, x)
    ax.plot([x, x], [y_data, y_model], color="yellow", linewidth=3)
    # print((y_data - y_model) ** 2)
    FIT += (y_data - y_model) ** 2
    plt.pause(0.1)


mask_Tneg = (spline_ts[TARGET-1] < ts) & (ts <= spline_ts[TARGET])

ax.plot(ts[mask_Tneg], os[mask_Tneg], color="blue")
# print()

for _, (x, y_data) in enumerate(zip(ts[mask_Tneg], os[mask_Tneg])):
    Aneg = (spline_ts[TARGET-1], spline_os[TARGET-1], spline_dodts[TARGET-1])
    Bneg = (spline_ts[TARGET], spline_os[TARGET], spline_dodts[TARGET])
    y_model = cubic_bezier(Aneg, Bneg, x)
    ax.plot([x, x], [y_data, y_model], color="orange", linewidth=3)
    # print((y_data - y_model) ** 1)
    FIT += (y_data - y_model) ** 2
    plt.pause(0.1)

print(f"brand new fit: {FIT}")

plt.show()
