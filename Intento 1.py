
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constantes
g = 9.81  # Aceleracion debido a la gravedad
L1 = 5.0  # Largo de la cuerda del pendulo 1 en metros
L2 = 1.0  # Largo de la cuerda del pendulo 2 en metros
m1 = 1.0  # Masa del pendulo 1 en Kg
m2 = 1.0  # Masa del pendulo 2 en Kg

# Angulos iniciales de los pendulos
AnguloInicial1 = 180  # Angulo Incial del pendulo 1
AnguloInicial2 = 90  # Angulo Inicial del pendulo 2

theta1_0 = np.radians(AnguloInicial1)
theta2_0 = np.radians(AnguloInicial2)

def Pendulo_Doble_Ode(t, y):
    theta1, z1, theta2, z2 = y
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1_punto = z1
    z1_dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1**2 * c + L2 * z2**2) -
              (m1 + m2) * g * np.sin(theta1)) / L1 / (m1 + m2 * s**2)

    theta2_punto = z2
    z2_dot = ((m1 + m2) * (L1 * z1**2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) +
              m2 * L2 * z2**2 * s * c) / L2 / (m1 + m2 * s**2)

    return theta1_punto, z1_dot, theta2_punto, z2_dot

y0 = [theta1_0, 0, theta2_0, 0]

t_max = 20
t_pasos = 200
t = np.linspace(0, t_max, t_pasos)

solution = solve_ivp(Pendulo_Doble_Ode, [0, t_max], y0, t_eval=t)

theta1, theta2 = solution.y[0], solution.y[2]

x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    pt1.set_data(x1[i], y1[i])
    pt2.set_data(x2[i], y2[i])
    return ln1, pt1, pt2

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ln1, = plt.plot([], [], 'o-', lw=2)
pt1, = plt.plot([], [], 'bo', markersize=8)
pt2, = plt.plot([], [], 'ro', markersize=8)

ax.set_xlim(-2 * (L1 + L2), 2 * (L1 + L2))
ax.set_ylim(-2 * (L1 + L2), 2 * (L1 + L2))
ax.set_aspect('equal', 'box')

ani = FuncAnimation(fig, animate, frames=t_pasos, blit=True, interval=1000 * t_max / t_pasos)

plt.show()
