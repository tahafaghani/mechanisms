"""
This program is written by Taha Faghani , 99542321, For Hw2 of Design of Mechanism peresented by Dr.Zabihifar
"""


import math
import numpy as np                     # type: ignore
import matplotlib.pyplot as plt         # type: ignore

t = np.arange(0, np.pi, 0.01)

""""
DEfine the Graphaical Position with Linkes
 We scale the linkages and the angular Velocity to plot them propperly
"""
# link lengths and ang (Velocity)
L1 = 2.5
L2 = 2
L3 = 3
L4 = 3.25
L5 = 2.5
Alpha = np.pi / 4
Teta1 = 0
W = 5
Teta2 = W * t

"""Now to define the mechanism montage mode, We define the sigma
"""
A = 2 * L1 * L4 * np.cos(Teta1) - 2 * L2 * L4 * np.cos(Teta2)
B = 2 * L1 * L4 * np.sin(Teta1) - 2 * L2 * L4 * np.sin(Teta2)
C = L1**2 + L2**2 + L4**2 - L3**2 - 2 * L1 * L2 * (np.cos(Teta1) * np.cos(Teta2) + np.sin(Teta1) * np.sin(Teta2))
sigma = 4 * B**2 - 4 * (C - A) * (C + A)
if np.any(sigma > 0):
    t1 = (-B + np.sqrt(B**2 - C**2 + A**2)) / (C - A)
    t2 = (-B - np.sqrt(B**2 - C**2 + A**2)) / (C - A)

# From the calculation above, Estimate the Theta4,3 and S1,2 and C1,2:
Teta4_1 = 2 * np.arctan(t1)
Teta4_2 = 2 * np.arctan(t2)
Sin1 = (L1 * np.sin(Teta1) + L4 * np.sin(Teta4_1) - L2 * np.sin(Teta2)) / L3
Sin2 = (L1 * np.sin(Teta1) + L4 * np.sin(Teta4_2) - L2 * np.sin(Teta2)) / L3
Cos1 = (L1 * np.cos(Teta1) + L4 * np.cos(Teta4_1) - L2 * np.cos(Teta2)) / L3
Cos2 = (L1 * np.cos(Teta1) + L4 * np.cos(Teta4_2) - L2 * np.cos(Teta2)) / L3
Teta3_1 = np.arctan2(Sin1, Cos1)
Teta3_2 = np.arctan2(Sin2, Cos2)

#Define the Points in the four-bar Mechanism:
P = np.array([L2 * np.cos(Teta2) + L5 * np.cos(Teta3_2 + Alpha), L2 * np.sin(Teta2) + L5 * np.sin(Alpha + Teta3_2)])
Px = P[0, :]
Py = P[1, :]
p1 = np.array([0, 0])

point_A = np.array([L2 * np.cos(Teta2), L2 * np.sin(Teta2)])

point_B= np.array([L1 * np.cos(Teta1) + L4 * np.cos(Teta4_2),L1 * np.sin(Teta1) + L4 * np.sin(Teta4_2)])

point_O = np.array([L1 * np.cos(Teta1), L1 * np.sin(Teta1)])


# With diff, we can compute the Velocity and Accelerations  of each point with respect to time
vx = np.diff(Px) / np.diff(t)
vy = np.diff(Py) / np.diff(t)
v = np.sqrt(vx**2 + vy**2)
ax = np.diff(vx)
ay = np.diff(vy)
a = np.sqrt(ax**2 + ay**2)
fig, ax = plt.subplots()
ax.set_xlim([-3, 5])
ax.set_ylim([-2.5, 4.2])
ax.grid(True)


for i in range(len(t)):
    LA, = ax.plot([p1[0], point_A[0, i]], [p1[1], point_A[1, i]], color='k', linewidth=4)
    LB, = ax.plot([point_A[0, i], point_B[0, i]], [point_A[1, i], point_B[1, i]],color='r', linewidth=4)
    LC, = ax.plot([point_B[0, i], point_O[0]], [point_B[1, i], point_O[1]], color='b', linewidth=4)
    LD, = ax.plot([point_A[0, i], P[0, i]], [point_A[1, i], P[1, i]],color='purple', linewidth=4)
    LE, = ax.plot([P[0, i], point_B[0, i]], [P[1, i], point_B[1, i]],color='y', linewidth=4)

    P5_circle = plt.Circle((P[:, i]), 0.08, color='g', fill=True)
    ax.add_artist(P5_circle)

    #stL1= 'P'
    stL2 = 'A'
    stL3 = 'B'

    P2_text = ax.text(point_A[0, i], point_A[1, i] + 0.4, stL2, fontsize=10)
    P3_text = ax.text(point_B[0, i], point_B[1, i] + 0.4, stL3, fontsize=10)
    #P1_text = ax.text(P[0, i], P[1, i] + 0.4, stL1, fontsize=10)

    plt.pause(0.05)

    if i < len(Teta2) - 1:
        LA.remove()
        LB.remove()
        LC.remove()
        LD.remove()
        LE.remove()
        P3_text.remove()
        P2_text.remove()




#Now plot the results:

#For the velocity:
plt.show()
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(v, linewidth=3)
plt.xlim([0, 100 * np.pi])
plt.ylim([0, 12])
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('velocity (m/s)')

#For the Acceleration:
plt.subplot(2, 1, 2)
plt.plot(a, linewidth=3)
plt.xlim([0, 100 * np.pi])
plt.ylim([0, 0.45])
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('acceleration (m/s^2)')

#For the for Point P:
plt.figure()
plt.plot(Px, linewidth=3)
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('X')

plt.figure()
plt.plot(Py, linewidth=3)
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('Y')
plt.show()