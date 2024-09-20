import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

input_file = "smooth_fit.npz"

my_loyalty = np.inf


def update_plot(frame):
    global my_loyalty
    try:
        with np.load(input_file) as data:
            x_data, y_data = data["data"]
            xs = data["xs"]
            ys = data["ys"]
            curvature = data["curvature"]
            loyalty = data["loyalty"]
            # evolution_x, evolution_y = data['evolution']
            # error = data['error']

        """
        with np.load(input_file_extra) as data:
            evolution_extra = data['evolution']
            error_extra = data['error']
        """

        if not my_loyalty == loyalty:
            my_loyalty = loyalty

            ax.cla()
            ax.plot(x_data, y_data, 'o', markersize=1)
            ax.plot(xs, ys)

            plt.title(f"Frame: {frame} - Error: {np.round(my_loyalty, 10)}")

            # folder = "GIF4"
            # filename = f"{folder}/ nr {frame}.png"
            # plt.savefig(filename, bbox_inches='tight')
            # plt.tight_layout()
    except Exception as e:
        print(f"Error reading file: {e}")


"""data = [
    (0, 50),
    (15, 338.84766),
    (30, 428.71816),
    (50, 400.18376),
    (60, 360.62972),
    (120, 208.98756),
    (180, 156.81751),
]
x_data = [point[0] for point in data]
y_data = [point[1] for point in data]"""

fig, ax = plt.subplots()

ani = FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False)

plt.show()
