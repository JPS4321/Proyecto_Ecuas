import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants for the double pendulum
g = 9.81  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
m1 = 1.0  # mass of pendulum 1 in kg
m2 = 1.0  # mass of pendulum 2 in kg

def double_pendulum_ode(t, y):
    theta1, z1, theta2, z2 = y
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1_dot = z1
    z1_dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
              (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    
    theta2_dot = z2
    z2_dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
              m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    
    return theta1_dot, z1_dot, theta2_dot, z2_dot

# Initial conditions
theta1_0 = np.pi / 2  # initial angle for pendulum 1
theta2_0 = np.pi / 2  # initial angle for pendulum 2
y0 = [theta1_0, 0, theta2_0, 0]  # initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt

# Time vector
t_max = 20
t_steps = 200
t = np.linspace(0, t_max, t_steps)

# Solve the ODE
solution = solve_ivp(double_pendulum_ode, [0, t_max], y0, t_eval=t)

# Unpack the solution
theta1, theta2 = solution.y[0], solution.y[2]

# Convert to Cartesian coordinates of the two pendulum bobs
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# Animation function
def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    pt1.set_data(x1[i], y1[i])
    pt2.set_data(x2[i], y2[i])
    return ln1, pt1, pt2

# Set up the figure, the axis, and the plot elements we want to animate
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ln1, = plt.plot([], [], 'o-', lw=2)
pt1, = plt.plot([], [], 'bo', markersize=8)
pt2, = plt.plot([], [], 'ro', markersize=8)

# Define the axes' limits
plt.axis('scaled')
plt.xlim(-2 * (L1+L2), 2 * (L1+L2))
plt.ylim(-2 * (L1+L2), 2 * (L1+L2))
ax.set_aspect('equal', 'box')

# Create the animation
ani = FuncAnimation(fig, animate, frames=t_steps, blit=True, interval=1000 * t_max / t_steps)

plt.show()
