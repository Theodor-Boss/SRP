import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


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
room = np.mean(np.diff(x_data)) / 2.
room_y = room * 6.
print("Line before definition of simulation")
x_data = x_data + np.random.uniform(-room, room, x_data.size)
y_data = y_data + np.random.uniform(-room_y, room_y, y_data.size)



rezz = 100
balance = 4000.  # Desto højere, desto mere loyal. Desto lavere, desto glattere

xs_nano = np.linspace(np.min(x_data), np.max(x_data), rezz)
ys_nano = np.zeros_like(xs_nano)

data_ownership = np.empty_like(x_data, dtype=int)
for i in range(len(x_data)):
    closest_x = np.argsort(np.abs(xs_nano - x_data[i]))
    left_border = np.minimum(closest_x[0], closest_x[1])
    right_border = np.maximum(closest_x[0], closest_x[1])
    data_ownership[i] = int(left_border)

print("Another line before definition of simulation")


def simulation(shared_array):
    global rezz, ys_nano
    my_curviness, my_loyalty = fitness((xs_nano, ys_nano), (x_data, y_data))
    my_total_fitness = my_curviness + my_loyalty * balance

    while True:
        # randomness = np.random.randint(0, rezz)
        new_ys_nano = ys_nano.copy()
        k = 0.1
        x0 = np.random.uniform(np.min(xs_nano), np.max(xs_nano))
        y0 = np.random.uniform(-0.0003, 0.0003)
        add_on = np.exp(-((xs_nano - x0) / k) ** 2) * y0
        new_ys_nano = new_ys_nano + add_on
        # new_ys_nano = new_ys_nano + (np.random.random(rezz) * 2 - 1) * 0.000314

        # new_ys_nano[randomness] = new_ys_nano[randomness] + (np.random.random() * 2 - 1) * 0.314

        new_my_curviness, new_my_loyalty = fitness(
            (xs_nano, new_ys_nano), (x_data, y_data)
        )
        new_my_total_fitness = new_my_curviness + new_my_loyalty * balance

        if new_my_total_fitness < my_total_fitness:
            my_curviness = new_my_curviness
            my_loyalty = new_my_loyalty
            my_total_fitness = new_my_total_fitness

            ys_nano = new_ys_nano

            with shared_array.get_lock():
                shared_array[:rezz] = ys_nano
                shared_array[-3:] = [my_curviness, my_loyalty, my_total_fitness]
                # shared_array[-1] = my_total_fitness

        time.sleep(0.001)  # Small delay to prevent excessive CPU usage


def visualization(shared_array):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]}
    )
    line, = ax1.plot(xs_nano, np.zeros_like(xs_nano))
    eurowind, = ax1.plot(x_data, y_data, 'o', markersize=3)
    print("A line inside definition of visualization")
    ax1.set_title("Smooth curve")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Angular velocity")

    text_loyalty = ax2.text(0.1, 0.25, "", transform=ax2.transAxes)
    text_curviness = ax2.text(0.1, 0.5, "", transform=ax2.transAxes)
    text_total_fitness = ax2.text(0.1, 0.75, "", transform=ax2.transAxes)
    ax2.set_title("Fitness Metrics")
    ax2.axis('off')

    """line, = ax1.plot(np.frombuffer(shared_array.get_obj()))"""
    # ax.set_ylim(0, 1)

    def update(frame):
        # Create a copy of the shared array data
        with shared_array.get_lock():
            data = np.frombuffer(shared_array.get_obj()).copy()

        ys = data[:rezz]
        curviness, loyalty, total_fitness = data[-3:]
        # total_fitness = data[-1]

        line.set_ydata(ys)
        text_loyalty.set_text(f"Loyalty: {loyalty:.6f}")
        text_curviness.set_text(f"Curviness: {curviness:.6f}")
        text_total_fitness.set_text(f"Total fitness: {total_fitness:.6f}")

        return line, text_curviness, text_loyalty, text_total_fitness

    ani = FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    shared_array = mp.Array('d', rezz + 3)
    sim_process = mp.Process(target=simulation, args=(shared_array,))
    vis_process = mp.Process(target=visualization, args=(shared_array,))

    sim_process.start()
    vis_process.start()

    sim_process.join()
    vis_process.join()
