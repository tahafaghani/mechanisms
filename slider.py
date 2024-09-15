import numpy as np
import matplotlib.pyplot as plt

r = 1
n = 1000
theta = np.linspace(0, 2 * np.pi, n)
Ax = r * np.cos(theta)
Ay = r * np.sin(theta)
a = 2.5
e = 0.25
c = 0.5
d = 1
t = 0.06
Bx = Ax + np.sqrt(a**2 - (Ay - e - c / 2)**2)  # Ensure this does not result in complex numbers
By = e + c / 2
axisLimits = [min(Ax) * 1.1, max(Bx) * 1.2, min(Ay) * 1.1, max(Ay) * 1.1]
sliderY = [e, e + c, e + c, e, e]
groundX = [min(Bx) - d / 2, min(Bx) * 1.2, max(Bx) + d / 2, max(Bx) + d / 2, min(Bx) - d / 2]
groundY = [e - t, e, e, e - t, e - t]

for ii in range(n):
    plt.plot(Ax, Ay, '--', color='blue')
    plt.plot(0, 0, 'ko')
    plt.axis('equal')
    sliderX = [Bx[ii] - d / 2, Bx[ii] - d / 2, Bx[ii] + d / 2, Bx[ii] + d / 2, Bx[ii] - d / 2]
    plt.fill(sliderX, sliderY, 'm')
    plt.fill(groundX, groundY, 'g')
    plt.plot([0, Ax[ii]], [0, Ay[ii]], 'k', linewidth=2)
    plt.plot(Ax[ii], Ay[ii], 'ko')
    plt.plot(Bx[ii], By, 'ko')  
    plt.axis(axisLimits)
    plt.plot([Ax[ii], Bx[ii]], [Ay[ii], By], 'k', linewidth=2)
    plt.axis('off')
    plt.pause(0.0005)
    plt.clf()  

plt.show()
