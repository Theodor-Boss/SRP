"""
### 6 ###
Denne fil beregner vinkelaccelerationen ved at differentiere
de kalibrerede vinkelhastigheder (strengt taget behøver
vinkelhastigheden ikke kalibreres for at finde den afledede.
Offsettet "forsvinder").
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys


input_file1 = "calibrated_omegas1.npz"
input_file2 = "calibrated_omegas2.npz"
input_file3 = "calibrated_omegas3.npz"
input_file4 = "calibrated_omegas4.npz"
input_file5 = "calibrated_omegas5.npz"

output_file = "smooth_fit.npz"


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


def derivative(xs, ys, slope0_idx, slope0):
    ys_dot = np.empty_like(xs, dtype=np.float64)
    ys_dot[slope0_idx] = slope0
    ms = np.diff(ys) / np.diff(xs)  # m[i] er sekanthældnningen fra i til i+1

    # generate positive derivatives:
    for i in range(len(xs) - slope0_idx - 1):
        ys_dot[slope0_idx+1 + (i)] = - ys_dot[slope0_idx + (i)] + 2 * ms[slope0_idx + (i)]

    # generate negative derivatives:
    for i in range(slope0_idx):
        ys_dot[slope0_idx + -(i+1)] = - ys_dot[slope0_idx+1 + -(i+1)] + 2 * ms[slope0_idx + -(i+1)]

    return ys_dot


def get_line_coffs(A, B):
    a = (B[1] - A[1]) / (B[0] - A[0])
    b = A[1] - a * A[0]
    return a, b


def fitness(smooth_curve, data):
    my_xs_nano, my_ys_nano = smooth_curve
    my_x_data, my_y_data = data
    """slopes:"""
    my_lines = np.empty((rezz - 1, 2))
    for i in range(len(my_lines)):
        my_lines[i] = get_line_coffs(
            (my_xs_nano[i], my_ys_nano[i]),
            (my_xs_nano[i+1], my_ys_nano[i+1])
        )

    curviness = np.sum(np.diff(my_lines[:, 0]) ** 2)

    a_vec = my_lines[data_ownership, 0]
    b_vec = my_lines[data_ownership, 1]
    loyalty_vectorized = np.sum(
        ((a_vec * my_x_data + b_vec) - my_y_data) ** 2
    )
    return curviness, loyalty_vectorized


x_data, y_data = ts1[:100], kalibreret_omegas1[:100]
rezz = 170
xs_nano = np.linspace(np.min(x_data), np.max(x_data), rezz)
ys_nano = np.zeros_like(xs_nano)  # np.random.random(len(xs_nano))
data_ownership = np.empty_like(x_data, dtype=int)
for i in range(len(x_data)):
    closest_x = np.argsort(np.abs(xs_nano - x_data[i]))
    left_border = np.minimum(closest_x[0], closest_x[1])
    right_border = np.maximum(closest_x[0], closest_x[1])
    data_ownership[i] = int(left_border)


# fig, ax = plt.subplots()


my_curviness, my_loyalty = fitness((xs_nano, ys_nano), (x_data, y_data))
print(my_loyalty)
print()
# successes = 0

"""
new_ys_nano = ys_nano
randomness = 11
new_ys_nano[randomness] = new_ys_nano[randomness] + 0.3

new_my_curviness, new_my_loyalty = fitness(
    (xs_nano, new_ys_nano), (x_data, y_data)
)

ax.cla()
ax.plot(x_data, y_data, 'o', markersize=1)
ax.plot(xs_nano, ys_nano)
# ax.plot(xs_nano, new_ys_nano, color="red")
plt.show()


sys.exit()
"""

while True:
    randomness = np.random.randint(0, rezz)
    new_ys_nano = ys_nano.copy()
    new_ys_nano[randomness] = new_ys_nano[randomness] + (np.random.random() * 2 - 1) * 0.00314
    # new_ys_nano = ys_nano + np.random.random(len(xs_nano))
    new_my_curviness, new_my_loyalty = fitness(
        (xs_nano, new_ys_nano), (x_data, y_data)
    )
    # print(new_my_loyalty)

    """ax.cla()
    ax.plot(x_data, y_data, 'o', markersize=1)
    ax.plot(xs_nano, ys_nano, color="blue")
    ax.plot(xs_nano, new_ys_nano, "r--")"""

    if new_my_loyalty < my_loyalty:
        my_curviness, my_loyalty = new_my_curviness, new_my_loyalty
        print(my_curviness, my_loyalty)
        ys_nano = new_ys_nano

        np.savez(
            output_file,
            data=[x_data, y_data],
            xs=xs_nano, ys=ys_nano,
            curvature=my_curviness, loyalty=my_loyalty
        )

    """successes += 1
    if successes % 1 == 0:
        plt.pause(0.3)"""





"""for i, x_nano in enumerate(xs_nano):
    mask = data_ownership == i
    (x_data[mask], y_data[mask])
    plt.pause(0.1)

ax.plot(xs_nano, ys_nano, color="black")

plt.show()"""

# my_residual = residual(ts_nano, os_nano, my_data)


sys.exit()

my_xs = np.array([0, 1,   2, 3,   4, 5])
my_ys = np.array([0, 3.5, 6, 7.5, 8, 7.5])
my_derivative = derivative(my_xs, my_ys, 0, 4.)
print(my_derivative)


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


hvile_idx = np.argmin(np.abs(ts1 - np.mean(hvile1)))


my_slope = 0

xlim1, ylim1 = None, None
xlim2, ylim2 = None, None


def update_plot(my_slope):
    global xlim1, ylim1
    if xlim1 is not None:
        xlim1 = ax1.get_xlim()
        ylim1 = ax1.get_ylim()
        xlim2 = ax2.get_xlim()
        ylim2 = ax2.get_ylim()

    alphas1 = derivative(ts1, kalibreret_omegas1, hvile_idx, my_slope)

    alphas1_dot = antiderivative(ts1, alphas1)

    ax1.clear()
    ax2.clear()

    ax1.plot(ts1, alphas1, color="C0")
    ax1.axvline(ts1[hvile_idx], color="C1")
    ax1.axhline(0, color="black")

    ax1.set_xlabel("Tid")
    ax1.set_ylabel("Vinkelacceleration")

    ax2.plot(ts1, kalibreret_omegas1, color="C0")
    ax2.plot(ts1, alphas1_dot + (kalibreret_omegas1[0] - alphas1_dot[0]), color="C1")

    fig.suptitle(f"Vinkelacceleration 1. Slope = {my_slope:.3f}")
    if xlim1 is not None:
        ax1.set_xlim(xlim1)
        ax1.set_ylim(ylim1)
        ax2.set_xlim(xlim2)
        ax2.set_ylim(ylim2)
    else:
        xlim1 = ax1.get_xlim()
        ylim1 = ax1.get_ylim()
        xlim2 = ax2.get_xlim()
        ylim2 = ax2.get_ylim()

    # plt.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1)
    plt.draw()


def on_scroll(event):
    global my_slope
    if event.button == "up":
        my_slope += 0.1
    elif event.button == "down":
        my_slope -= 0.1
    update_plot(my_slope)


fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
update_plot(my_slope)

fig.canvas.mpl_connect("scroll_event", on_scroll)

plt.show()


sys.exit()


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
