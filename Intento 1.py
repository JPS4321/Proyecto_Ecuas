
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constantes
g = 9.81  # Aceleracion debido a la gravedad

#PENDULO DOBLE  1 
L1 = 1.0  # Largo de la cuerda del pendulo 1 en metros
L2 = 1.0  # Largo de la cuerda del pendulo 2 en metros
m1 = 1.0  # Masa del pendulo 1 en Kg
m2 = 1.0  # Masa del pendulo 2 en Kg

#PENDULO DOBLE 2
L3 = 1.0  # Largo de la cuerda del pendulo 1 en metros
L4 = 1.0  # Largo de la cuerda del pendulo 2 en metros
m3 = 1.0  # Masa del pendulo 1 en Kg
m4 = 1.0  # Masa del pendulo 2 en Kg

# ANGULO INICIAL DE PENDULO DOBLE 1
initial_angle1 = 90  # Angulo Incial del pendulo 1
initial_angle2 = 90   # Angulo Inicial del pendulo 2

# ANGULO INICIAL DE PENDULO DOBLE 2
initial_angle3 = 90 + 0.0001 # Angulo Incial del pendulo 1
initial_angle4 = 90 # Angulo Inicial del pendulo 2

theta1_0 = np.radians(initial_angle1)
theta2_0 = np.radians(initial_angle2)
theta3_0 = np.radians(initial_angle3)
theta4_0 = np.radians(initial_angle4)

def double_pendulum_ode(t, y):
    theta1, z1, theta2, z2, theta3, z3, theta4, z4 = y
    c12, s12 = np.cos(theta1 - theta2), np.sin(theta1 - theta2)
    c34, s34 = np.cos(theta3 - theta4), np.sin(theta3 - theta4)

    theta1_dot = z1
    z1_dot = (m2 * g * np.sin(theta2) * c12 - m2 * s12 * (L1 * z1**2 * c12 + L2 * z2**2) -
              (m1 + m2) * g * np.sin(theta1)) / L1 / (m1 + m2 * s12**2)

    theta2_dot = z2
    z2_dot = ((m1 + m2) * (L1 * z1**2 * s12 - g * np.sin(theta2) + g * np.sin(theta1) * c12) +
              m2 * L2 * z2**2 * s12 * c12) / L2 / (m1 + m2 * s12**2)

    theta3_dot = z3
    z3_dot = (m4 * g * np.sin(theta4) * c34 - m4 * s34 * (L3 * z3**2 * c34 + L4 * z4**2) -
              (m3 + m4) * g * np.sin(theta3)) / L3 / (m3 + m4 * s34**2)

    theta4_dot = z4
    z4_dot = ((m3 + m4) * (L3 * z3**2 * s34 - g * np.sin(theta4) + g * np.sin(theta3) * c34) +
              m4 * L4 * z4**2 * s34 * c34) / L4 / (m3 + m4 * s34**2)

    return theta1_dot, z1_dot, theta2_dot, z2_dot, theta3_dot, z3_dot, theta4_dot, z4_dot


y0 = [theta1_0, 0, theta2_0, 0, theta3_0, 0, theta4_0, 0]

t_max = 60
t_steps = 2000
t = np.linspace(0, t_max, t_steps)

solution = solve_ivp(double_pendulum_ode, [0, t_max], y0, t_eval=t)

theta1, theta2 = solution.y[0], solution.y[2]
theta3, theta4 = solution.y[4], solution.y[6]

x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

x3 = L3 * np.sin(theta3)
y3 = -L3 * np.cos(theta3)
x4 = x3 + L4 * np.sin(theta4)
y4 = y3 - L4 * np.cos(theta4)

def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    pt1.set_data(x1[i], y1[i])
    pt2.set_data(x2[i], y2[i])

    ln2.set_data([0, x3[i], x4[i]], [0, y3[i], y4[i]])
    pt3.set_data(x3[i], y3[i])
    pt4.set_data(x4[i], y4[i])
    
    return ln1, pt1, pt2, ln2, pt3, pt4

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ln1, = plt.plot([], [], 'o-', lw=2, color='blue')
pt1, = plt.plot([], [], 'bo', markersize=8)
pt2, = plt.plot([], [], 'ro', markersize=8)

ln2, = plt.plot([], [], 'o-', lw=2, color='green')
pt3, = plt.plot([], [], 'go', markersize=8)
pt4, = plt.plot([], [], 'yo', markersize=8)

max_length = max(L1 + L2, L3 + L4)
ax.set_xlim(-2 * max_length, 2 * max_length)
ax.set_ylim(-2 * max_length, 2 * max_length)
ax.set_aspect('equal', 'box')

ani = FuncAnimation(fig, animate, frames=t_steps, blit=True, interval=1000 * t_max / t_steps)

plt.show()
