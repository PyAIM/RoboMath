import numpy as np
import matplotlib.pyplot as plt

def plot_arm(theta1, theta2, L1=1, L2=1):
    x0, y0 = 0, 0
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)

    plt.plot([x0, x1], [y0, y1], 'b-', linewidth=5) # Link 1
    plt.plot([x1, x2], [y1, y2], 'r-', linewidth=5) # Link 2
    plt.plot(x2, y2, 'ko', markersize=10) # End-effector
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.title(f"2-Link Robot Arm | θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°")
    plt.show()

# Example: Ask students for angles in degrees, convert to radians
theta1 = np.radians(45)
theta2 = np.radians(30)
plot_arm(theta1, theta2)