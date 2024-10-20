import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from collections import deque 


def load_data():
    input_files = [
        "calibrated_omegas1.npz",
        "calibrated_omegas2.npz",
        "calibrated_omegas3.npz",
        "calibrated_omegas4.npz",
        "calibrated_omegas5.npz"
    ]

    data = []
    for file in input_files:
        with np.load(file) as npz_data:
            data.append((npz_data["ts"], npz_data["calibrated_omegas"]))
    return data


def get_line_coffs(A, B):
    a = (B[1] - A[1]) / (B[0] - A[0])
    b = A[1] - a * A[0]
    return a, b


def fitness(smooth_curve, data, rezz, data_ownership):
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

    """curviness = np.sum(
        np.sqrt(
            np.diff(my_xs_nano) ** 2 + np.diff(my_ys_nano) ** 2
        )
    )"""

    a_vec = my_lines[data_ownership, 0]
    b_vec = my_lines[data_ownership, 1]
    loyalty_vectorized = np.sum(
        ((a_vec * my_x_data + b_vec) - my_y_data) ** 2
    )
    return curviness, loyalty_vectorized


def simulation(shared_array, xs_nano, x_data, y_data, rezz, data_ownership, balance):
    ys_nano = np.full_like(xs_nano, -7.4)
    my_curviness, my_loyalty = fitness((xs_nano, ys_nano), (x_data, y_data), rezz, data_ownership)
    my_total_fitness = my_curviness + my_loyalty * balance


    learning_rate = 0.05
    counter = 0

    ups = -1
    downs = 0

    while True:
        new_ys_nano = ys_nano.copy()

        ###
        add_on = np.random.uniform(-1., 1., new_ys_nano.shape) * learning_rate
        new_ys_nano = new_ys_nano + add_on
        ###

        new_my_curviness, new_my_loyalty = fitness(
            (xs_nano, new_ys_nano), (x_data, y_data), rezz, data_ownership
        )
        new_my_total_fitness = new_my_curviness + new_my_loyalty * balance

        if new_my_total_fitness < my_total_fitness:
            my_curviness = new_my_curviness
            my_loyalty = new_my_loyalty
            my_total_fitness = new_my_total_fitness
            ys_nano = new_ys_nano

            ups = np.sqrt(my_curviness / (rezz - 2))

            with shared_array.get_lock():
                shared_array[:rezz] = ys_nano
                shared_array[-5:] = [my_curviness, my_loyalty, my_total_fitness, ups, downs]

        

        counter += 1

        # time.sleep(2)  # Small delay to prevent excessive CPU usage


def visualization(shared_array, xs_nano, x_data, y_data, rezz):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]}
    )
    line, = ax1.plot(xs_nano, np.zeros_like(xs_nano))
    eurowind, = ax1.plot(x_data, y_data, 'o', markersize=3)
    ax1.set_title("Smooth curve")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Angular velocity")

    text_loyalty = ax2.text(0.1, 0.25, "", transform=ax2.transAxes)
    text_curviness = ax2.text(0.1, 0.5, "", transform=ax2.transAxes)
    text_total_fitness = ax2.text(0.1, 0.75, "", transform=ax2.transAxes)
    text_ups = ax2.text(0.6, 0.25, "", transform=ax2.transAxes)
    text_downs = ax2.text(0.6, 0.5, "", transform=ax2.transAxes)
    ax2.set_title("Fitness Metrics")
    ax2.axis('off')

    """line, = ax1.plot(np.frombuffer(shared_array.get_obj()))"""
    # ax.set_ylim(0, 1)

    def update(frame):
        # Create a copy of the shared array data
        with shared_array.get_lock():
            data = np.frombuffer(shared_array.get_obj()).copy()
            # TODO Data: bad name
        ys = data[:rezz]
        curviness, loyalty, total_fitness, ups, downs = data[-5:]

        line.set_ydata(ys)
        text_loyalty.set_text(f"Loyalty: {loyalty:.6f}")
        text_curviness.set_text(f"Curviness: {curviness:.6f}")
        text_total_fitness.set_text(f"Total fitness: {total_fitness:.6f}")
        text_ups.set_text(f"  Ups: {ups}")
        text_downs.set_text(f"Downs: {downs}")

        return line, text_curviness, text_loyalty, text_total_fitness, text_ups, text_downs

    ani = FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


def main():
    print("Starting main function")

    data = load_data()

    start, stop = 600, 620

    x_data, y_data = data[0][0][start:stop], data[0][1][start:stop]

    """
    room = np.mean(np.diff(x_data)) / 2.
    room_y = room * 6.
    print("Line before definition of simulation")
    x_data = x_data + np.random.uniform(-room, room, x_data.size)
    y_data = y_data + np.random.uniform(-room_y, room_y, y_data.size)
    """

    rezz = 30
    balance = 300.  # Desto højere, desto mere loyal. Desto lavere, desto glattere

    xs_nano = np.linspace(np.min(x_data), np.max(x_data), rezz)

    data_ownership = np.empty_like(x_data, dtype=int)
    for i in range(len(x_data)):
        closest_x = np.argsort(np.abs(xs_nano - x_data[i]))
        left_border = np.minimum(closest_x[0], closest_x[1])
        right_border = np.maximum(closest_x[0], closest_x[1])
        data_ownership[i] = int(left_border)

    print("Data preparation complete")

    shared_array = mp.Array('d', rezz + 5)
    sim_process = mp.Process(target=simulation, args=(shared_array, xs_nano, x_data, y_data, rezz, data_ownership, balance))
    vis_process = mp.Process(target=visualization, args=(shared_array, xs_nano, x_data, y_data, rezz))

    sim_process.start()
    vis_process.start()

    sim_process.join()
    vis_process.join()


if __name__ == "__main__":
    main()
