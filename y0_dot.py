"""
### 7 ###
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
from scipy import stats


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


def derivative(xs, ys, slope0_idx):
    ms = np.diff(ys) / np.diff(xs)  # m[i] er sekanthældnningen fra i til i+1

    # Først bestemmes slope0 vha. noget avanceret matematik:
    N = slope0_idx
    P = len(xs) - slope0_idx - 1
    pos = 0
    for i in range(P):
        pos += (-1) ** i * (2 * (P - i) - 1) * ms[N + i]

    neg = 0
    for i in range(N):
        neg += (-1) ** i * (2 * (N - i) - 1) * ms[N - i - 1]

    slope0 = (neg + pos) / (N + P)

    ys_dot = np.empty_like(xs, dtype=np.float64)
    ys_dot[slope0_idx] = slope0

    # Dernæst genereres de resterende slopes:
    for i in range(len(xs) - slope0_idx - 1):
        ys_dot[slope0_idx+1 + (i)] = - ys_dot[slope0_idx + (i)] + 2 * ms[slope0_idx + (i)]

    for i in range(slope0_idx):
        ys_dot[slope0_idx + -(i+1)] = - ys_dot[slope0_idx+1 + -(i+1)] + 2 * ms[slope0_idx + -(i+1)]

    return ys_dot


def nabla_derivative(xs, ys, slope0_idx=0):
    ms = np.diff(ys) / np.diff(xs)  # m[i] er sekanthældnningen fra i til i+1

    # Først bestemmes slope0 vha. noget avanceret matematik:
    N = slope0_idx
    P = len(xs) - slope0_idx - 1
    pos = 0
    for i in range(P):
        pos += (-1) ** i * (2 * (P - i) - 1) * ms[N + i]

    neg = 0
    for i in range(N):
        neg += (-1) ** i * (2 * (N - i) - 1) * ms[N - i - 1]

    slope0 = (neg + pos) / (N + P)

    ys_dot = np.empty_like(xs, dtype=np.float64)
    ys_dot[slope0_idx] = slope0

    # Dernæst genereres de resterende slopes:
    for i in range(len(xs) - slope0_idx - 1):
        ys_dot[slope0_idx+1 + (i)] = - ys_dot[slope0_idx + (i)] + 2 * ms[slope0_idx + (i)]

    for i in range(slope0_idx):
        ys_dot[slope0_idx + -(i+1)] = - ys_dot[slope0_idx+1 + -(i+1)] + 2 * ms[slope0_idx + -(i+1)]

    # TODO Kan sagtens optimeres
    dy_dot_dy = np.empty_like(xs)
    dy_dot_dy[0] = 1 / (len(xs) - 1) * (2 * len(xs) - 3) / ms[0]
    dy_dot_dy[-1] = 1 / (len(xs) - 1) * (2 * len(xs) - 3) / ms[-1]
    for T in range(1, len(xs) - 1):
        dy_dot_dy[T] = 1 / (len(xs) - 1) * ((2*len(xs)-2*T-3)/(xs[T]-xs[T+1]) + (2*T-1)/(xs[T]-xs[T-1]))

    dFitNeg_dy = np.empty_like(xs)
    dFitNeg_dy[0] = 0
    dFitNeg_dy[1] = 8 * (ys_dot[1] - ms[0]) * (dy_dot_dy[1] - 1/(xs[1]-xs[0]))
    for T in range(2, len(xs)):
        crazy_sum = sum((-1)**k * (2*(T-(k-1))-1) * ms[T-k] for k in range(2, T+1))
        dFitNeg_dy[T] = 8 * (ys_dot[T] - ms[T-1]) * (dy_dot_dy[T] - 1/(xs[T]-xs[T-1])) + 8 * (dy_dot_dy[T] - 2/(xs[T]-xs[T-1])) * ((T-1)*ys_dot[T] - 2*(T-1)*ms[T-1] + crazy_sum)

    dFitPos_dy = np.empty_like(xs)
    dFitPos_dy[-1] = 0
    dFitPos_dy[-2] = 8 * (ys_dot[-2] - ms[-1]) * (dy_dot_dy[-1] - 1/(xs[-2]-xs[-1]))
    for T in range(len(xs) - 2):
        crazy_mus = sum((-1)**n * (2*(T-(n-1))-1) * ms[T-n] for n in range(2, len(xs)-T))
        dFitPos_dy[T] = 8 * (ys_dot[T] - ms[T]) * (dy_dot_dy[T] - 1/(xs[T]-xs[T+1])) + 8 * (dy_dot_dy[T] - 2/(xs[T]-xs[T+1])) * ((len(xs)-T-2)*ys_dot[T] - 2*(len(xs)-T-2)*ms[T] + crazy_mus)

    dFit_dy = dFitNeg_dy + dFitPos_dy
    # print(dFit_dy[0])
    return dFit_dy, ys_dot


def parabel(x, A, B, c):
    xA, yA = A
    xB, yB = B
    return .5 * (yB - yA) / (xB - xA) * (x - xA) ** 2 + yA * (x - xA) + c

def curvature(xs, ys):
    dy_dx = derivative(xs, ys, 0)
    arc_length = 0
    for i in range(len(xs)-1):
        parable_xs = np.linspace(xs[i], xs[i+1], 100)
        parable_ys = parabel(
            parable_xs,
            (xs[i], dy_dx[i]),
            (xs[i+1], dy_dx[i+1]),
            ys[i]
        )
        arc_length += np.sum(np.sqrt(np.diff(parable_xs) ** 2 + np.diff(parable_ys) ** 2))
    return arc_length


my_xs = np.array([-4.6, 0., 1.,  2.25, 4.,  9., 12., 16., 20.6, 27.7, 33.14, 35.,  37.])
my_ys = np.array([-4.1, 0., 3.5, 6.,   7.5, 8.,  6.4, 4.8, 3.8,  1.1,  0.,   -0.3, -1.2]) # my_xs * 1.2 + 4.3
# my_ys[5] += 40.
# my_ys[6] += -44.8

my_ys_backup = my_ys.copy()

# gradient, dy_dx = nabla_derivative(my_xs, my_ys)
# plt.plot(my_xs, gradient)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# ymin, ymax = np.min(my_ys) - 5, np.max(my_ys) + 5
# xmin2, xmax2 = [], []
# ymin2, ymax2 = [], []

for _ in range(10000):
    ax1.cla()
    ax2.cla()

    parab_dy_dx = derivative(my_xs, my_ys, 0)
    for j in range(len(my_xs)-1):
        parable_xs = np.linspace(my_xs[j], my_xs[j+1], 100)
        parable_ys = parabel(
            parable_xs,
            (my_xs[j], parab_dy_dx[j]),
            (my_xs[j+1], parab_dy_dx[j+1]),
            my_ys[j]
        )
        ax1.plot(parable_xs, parable_ys)

    T_x = np.empty_like(my_xs)
    T_y = np.empty_like(my_xs)
    dys = np.linspace(-20, 20, 100)
    for T in range(len(my_xs)):
        Fits = np.empty_like(dys)
        # print("Fits", Fits)
        for i, dy in enumerate(dys):
            my_ys_copy = my_ys.copy()
            my_ys_copy[T] += dy
            dy_dx = derivative(my_xs, my_ys_copy, 0)
            Fits[i] = np.sum(np.diff(dy_dx) ** 2)
        # print(T, np.min(Fits))
        coefficients = np.polyfit(dys, Fits, 2)
        a, b, c = coefficients
        T_x[T] = - b / (2 * a)
        T_y[T] = - (b ** 2 - 4 * a * c) / (4 * a)

        ax2.scatter(T_x[T], T_y[T])
        ax2.annotate(str(T), (T_x[T], T_y[T]), xytext=(0, 5), textcoords='offset points', ha='center')

    # if xmin2 == []:
    #     xmin2, xmax2 = np.min(T_x), np.max(T_x)
    #     ymin2, ymax2 = np.min(T_y), np.max(T_y)

    change_whom = np.argmin(T_y)
    my_ys[change_whom] += T_x[change_whom] * 1e-1
    ax1.plot(my_xs, my_ys_backup, 'ro')
    ax1.axvline(my_xs[change_whom], color="red")

        # print(T, a, b, c)
        # plt.scatter(T, np.min(Fits), c="black")
    # print(T_x)
    
    # for i, (x_val, y_val) in enumerate(zip(T_x, T_y)):
        

    """y_pred = a*dys**2 + b*dys + c

    ss_res = np.sum((Fits - y_pred) ** 2)
    ss_tot = np.sum((Fits - np.mean(Fits)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.plot(dys, y_pred, 'ro')
    print(T, r_squared)"""

    # plt.tight_layout()
    # ax1.set_ylim(ymin, ymax)
    # ax2.set_xlim(xmin2, xmax2)
    # ax2.set_ylim(ymin2, ymax2)
    plt.pause(0.01)


# plt.legend()
plt.show()
sys.exit()


T = 6
dys = np.linspace(-100, 300, 1000)
Fits = np.empty_like(dys)
for i, dy in enumerate(dys):
    my_ys_copy = my_ys.copy()
    my_ys_copy[T] += dy
    dy_dx = derivative(my_xs, my_ys_copy, 0)
    Fits[i] = np.sum(np.diff(dy_dx) ** 2)
plt.plot(dys, Fits)


plt.show()




my_dy_dx = derivative(my_xs, my_ys, 0)
Fit = np.sum(np.diff(my_dy_dx) ** 2)

print(Fit)

nabla = np.empty_like(my_xs)
for T in range(len(my_xs)-1):
    my_ys_copy = my_ys.copy()
    my_ys_copy[T] += 0.001
    my_temp_dy_dx = derivative(my_xs, my_ys_copy, 0)
    Fit_temp = np.sum(np.diff(my_temp_dy_dx) ** 2)
    nabla[T] = (Fit_temp - Fit) / 0.001

plt.plot(my_xs, nabla, color="red")

plt.axvline(my_xs[5])
plt.show()



for _ in range(100):
    plt.cla()
    dy_dx = derivative(my_xs, my_ys, 0)
    for j in range(len(my_xs)-1):
        parable_xs = np.linspace(my_xs[j], my_xs[j+1], 100)
        parable_ys = parabel(
            parable_xs,
            (my_xs[j], dy_dx[j]),
            (my_xs[j+1], dy_dx[j+1]),
            my_ys[j]
        )
        plt.plot(parable_xs, parable_ys)

    my_ys[6] += 32.8 / 100

    plt.pause(0.1)
    
    
plt.show()

sys.exit()


my_ys_backup = my_ys.copy()

ymin, ymax = np.min(my_ys) - 20, np.max(my_ys) + 20

frame_of_reference = curvature(my_xs, my_ys)

for _ in range(10000):
    plt.cla()

    copy_ys = np.random.normal(my_ys, 0.1)

    dy_dx = derivative(my_xs, copy_ys, 0)
    arc_length = 0
    for j in range(len(my_xs)-1):
        parable_xs = np.linspace(my_xs[j], my_xs[j+1], 100)
        parable_ys = parabel(
            parable_xs,
            (my_xs[j], dy_dx[j]),
            (my_xs[j+1], dy_dx[j+1]),
            copy_ys[j]
        )
        arc_length += np.sum(np.sqrt(np.diff(parable_xs) ** 2 + np.diff(parable_ys) ** 2))
        loyalty = np.sum((copy_ys - my_ys_backup) ** 2)
        plt.plot(parable_xs, parable_ys)
    print(arc_length, loyalty)
    fitness = arc_length + loyalty / 10
    if fitness < frame_of_reference:
        my_ys = copy_ys
        frame_of_reference = fitness

        plt.ylim(ymin, ymax)
        plt.title(frame_of_reference)
        plt.pause(0.1)



sys.exit()

my_new_xs = my_xs.copy()
my_new_ys = my_ys.copy()


fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)


ax1.cla()
ax2.cla()
ax3.cla()

gradient, dy_dx = nabla_derivative(my_new_xs, my_new_ys)
change_whom = np.argmax(np.abs(gradient))
ax2.axvline(my_xs[change_whom])

"""if i < 50:
    my_new_ys[change_whom] -= gradient[change_whom] * 1e-3
else:
    my_new_ys[change_whom] -= gradient[change_whom] * 1e-3
print(i, my_new_ys[5])"""

ax1.plot(my_new_xs, dy_dx)
ax1.set_ylim(-10, 10)

ax2.plot(my_new_xs, gradient)
ax2.set_ylim(-100, 100)
# ax3.plot(my_new_xs, my_new_ys)
ax3.plot(my_new_xs, my_new_ys, "ro")
ax3.set_ylim(-25, 75)

for j in range(len(my_new_xs)-1):
    parable_xs = np.linspace(my_new_xs[j], my_new_xs[j+1], 100)
    parable_ys = parabel(
        parable_xs,
        (my_new_xs[j], dy_dx[j]),
        (my_new_xs[j+1], dy_dx[j+1]),
        my_new_ys[j]
    )
    ax3.plot(parable_xs, parable_ys)


my_dy_dx = derivative(my_xs, my_ys, 0)
Fit = np.sum(np.diff(my_dy_dx) ** 2)

print(Fit)

nabla = np.empty_like(my_xs)
for T in range(len(my_xs)-1):
    my_ys_copy = my_ys.copy()
    my_ys_copy[T] += 0.001
    my_temp_dy_dx = derivative(my_xs, my_ys_copy, 0)
    Fit_temp = np.sum(np.diff(my_temp_dy_dx) ** 2)
    nabla[T] = (Fit_temp - Fit) / 0.001

ax2.plot(my_xs, nabla, color="red")



plt.tight_layout()
plt.show()

sys.exit()


for i in range(len(my_new_xs)):
    my_derivative = derivative(my_new_xs, my_new_ys, i)
    print(my_derivative[0])
    plt.plot(my_new_xs, my_derivative)

    for j in range(len(my_new_xs)-1):
        parable_xs = np.linspace(my_new_xs[j], my_new_xs[j+1], 100)
        parable_ys = parabel(
            parable_xs,
            (my_new_xs[j], my_derivative[j]),
            (my_new_xs[j+1], my_derivative[j+1]),
            my_new_ys[j]
        )
        plt.plot(parable_xs, parable_ys)

plt.plot(my_new_xs, my_new_ys, 'ro')

plt.show()
