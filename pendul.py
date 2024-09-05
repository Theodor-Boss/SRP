import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Assuming you have your data stored in arrays: angle_data, velocity_data, acceleration_data
# You can replace these arrays with your actual data

# Sample data (replace this with your actual data)
time = np.linspace(0, 10, num=100)  # Assuming time intervals
angle_data = np.sin(time)  # Sample angle data
velocity_data = np.cos(time)  # Sample angular velocity data
acceleration_data = -np.sin(time)  # Sample angular acceleration data

# Function to update the pendulum's position in the animation
def update(frame):
    plt.cla()  # Clear the previous frame
    angle = angle_data[frame]  # Get the angle at the current frame
    x = np.sin(angle)  # Calculate x-coordinate of pendulum
    y = -np.cos(angle)  # Calculate y-coordinate of pendulum
    plt.plot([0, x], [0, y], 'r-o')  # Plot pendulum
    plt.xlim(-1.5, 1.5)  # Adjust x-axis limits
    plt.ylim(-1.5, 0.5)  # Adjust y-axis limits
    plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
    plt.title(f'Frame: {frame}, Time: {time[frame]:.2f}s')  # Display frame number and time

# Create the animation
fig = plt.figure()
animation = FuncAnimation(fig, update, frames=len(time), interval=50)
plt.show()
