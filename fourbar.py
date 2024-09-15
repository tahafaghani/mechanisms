"""
This program is written by Taha Faghani , 99542321, For Final Exam of Design of Mechanism peresented by Dr.Zabihifar
"""
#Requierment Libraries 
#make sure you have then otherwise, simply install it by ! pip install <lib_name>
import numpy as np                                                  #type:ignore
import matplotlib.pyplot as plt                                         #type:ignore
from pyswarm import pso                                                  #type:ignore

#if you run this:
print("That may be time consuming...Be Patient :)")

#---------------------------Evaluate PSO net------------------------------------------------------------------------ 
class FourBarMechanism:
    def __init__(self, r1, r2, r3, r4, r5, theta1):
     # Initialize the parameters of the four-bar mechanism
        #all the lengths are (cm)

        self.r1 = r1  # Ground link length
        self.r2 = r2  # Input link length
        self.r3 = r3  # Coupler link length
        self.r4 = r4  # Output link length
        self.r5 = r5  # Additional link length
        self.theta1 = theta1  # Input angle

        self.alpha = np.pi / 4
        self.t = np.arange(0, 2 * np.pi, 0.001)

        self.W = 5  # (rad/s), assuming constant angular velocity
        self.Theta2 = self.W * self.t
    
    def get_position(self):
        # Calculate the position of point P based on kinematic equations
        # Define the kinematic equations for the four-bar mechanism
        r1, r2, r3, r4, r5, theta1, Theta2 = self.r1, self.r2, self.r3, self.r4, self.r5, self.theta1, self.Theta2
        alpha = self.alpha
        # Define the kinematic equations for the four-bar mechanism
        A = 2 * r1 * r4 * np.cos(theta1) - 2 * r2 * r4 * np.cos(Theta2)
        B = 2 * r1 * r4 * np.sin(theta1) - 2 * r2 * r4 * np.sin(Theta2)
        C = r1**2 + r2**2 + r4**2 - r3**2 - 2 * r1 * r2 * (np.cos(theta1) * np.cos(Theta2) + np.sin(theta1) * np.sin(Theta2))
        # Correct handling of invalid values in sqrt
        discriminant = B**2 - C**2 + A**2
        discriminant[discriminant < 0] = np.nan  # Set invalid values to NaN

        t1 = (-B + np.sqrt(discriminant)) / (C - A)
        t2 = (-B - np.sqrt(discriminant)) / (C - A)
        # From the calculation above, Estimate the Theta4,3 and S1,2 and C1,2:
        Teta4_1 = 2 * np.arctan(t1)
        Teta4_2 = 2 * np.arctan(t2)
        Sin1 = (r1 * np.sin(theta1) + r4 * np.sin(Teta4_1) - r2 * np.sin(Theta2)) / r3
        Sin2 = (r1 * np.sin(theta1) + r4 * np.sin(Teta4_2) - r2 * np.sin(Theta2)) / r3
        Cos1 = (r1 * np.cos(theta1) + r4 * np.cos(Teta4_1) - r2 * np.cos(Theta2)) / r3
        Cos2 = (r1 * np.cos(theta1) + r4 * np.cos(Teta4_2) - r2 * np.cos(Theta2)) / r3
        Teta3_1 = np.arctan2(Sin1, Cos1)
        Teta3_2 = np.arctan2(Sin2, Cos2)
        # Define the Points in the four-bar Mechanism:
        P = np.array([r2 * np.cos(Theta2) + r5 * np.cos(Teta3_2 + alpha), r2 * np.sin(Theta2) + r5 * np.sin(alpha + Teta3_2)])
        Px = P[0, :]
        Py = P[1, :]
        return Px, Py
    
    # Compute the fitness (deviation from a desired circle)
    def target_function(self):
        # Define the target function to minimize
        # Get the positions
        Px, Py = self.get_position()
        R = 7.5  # Desired radius

        # Circle equation: (x - Px)^2 + (y - Py)^2 = R^2
        center_x, center_y = np.mean(Px), np.mean(Py)  # Estimate the center as the mean of Px and Py

        # Compute the fitness as the deviation from the circle
        distances = np.sqrt((Px - center_x)**2 + (Py - center_y)**2)
        fitness = np.mean((distances - R)**2)

        return fitness

# Define the objective function for PSO
def objective_function(params):
    # Evaluate the fitness of the mechanism configuration
    r1, r2, r3, r4, r5, theta1 = params
    
    # Constraint: r2 + r5 < 15
    if r2 + r5 > 15:
        return 1e6  # Large penalty for violating the constraint

    mechanism = FourBarMechanism(r1, r2, r3, r4, r5, theta1)
    fitness = mechanism.target_function()

    if np.isnan(fitness):
        return 1e6  # Large penalty for invalid configurations

    return fitness

# Define bounds for each parameter
bounds = (
    (2, 20),  # r1
    (0, 10),  # r2
    (0, 10),  # r3
    (0, 10),  # r4
    (0.1, 20),  # r5
    (0, 2*np.pi)  # theta1
)

# Run PSO to find the best parameters
best_params, best_cost = pso(objective_function, [b[0] for b in bounds], [b[1] for b in bounds]
                             , swarmsize=300, maxiter=3000, minstep=1e-8, minfunc=1e-8)


#Now we print the best values for r1, r2, r3, r4, r5 , Theta1
print(f"Best parameters:  r1={best_params[0]:.3f}, r2={best_params[1]:0.3f}, r3={best_params[2]:0.3f}")
print(f"r4={best_params[3]:0.3f}, r5={best_params[4]:0.3f},and Theta1={best_params[5]:0.3f}")
# Now we print the cost score(the more less, the more better)

print("Best cost:", best_cost)

# Plot the path of point P and the desired circle
mechanism = FourBarMechanism(*best_params)
Px, Py = mechanism.get_position()

plt.figure()
plt.plot(Px, Py, label='Path of Point P')
plt.gca().set_aspect('equal', adjustable='box')

# Plot the desired circle
theta = np.linspace(0, 2*np.pi, 100)
circle_x = 7.5 * np.cos(theta)
circle_y = 7.5 * np.sin(theta)
plt.plot(circle_x, circle_y, label='Desired Circle', linestyle='--')

#We compare the path followed by Point P and the desired Circle which must trace
plt.xlabel('Px')
plt.ylabel('Py')
plt.title('Trajectory of Point P')
plt.legend()
plt.grid(True)
plt.show()


#-------------------------------------------------PLOT 4-Bar Mechanism----------------------------------------------------------------------------------
t = np.arange(0, np.pi, 0.01)
L1,L2,L3,L4,L5,Teta1=best_params
Alpha = np.pi /4
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
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
ax.grid(True)

# Visualize the mechanism motion
for i in range(len(t)):
    LA, = ax.plot([p1[0], point_A[0, i]], [p1[1], point_A[1, i]], color='k', linewidth=4)
    LB, = ax.plot([point_A[0, i], point_B[0, i]], [point_A[1, i], point_B[1, i]],color='r', linewidth=4)
    LC, = ax.plot([point_B[0, i], point_O[0]], [point_B[1, i], point_O[1]], color='b', linewidth=4)
    LD, = ax.plot([point_A[0, i], P[0, i]], [point_A[1, i], P[1, i]],color='purple', linewidth=4)
    LE, = ax.plot([P[0, i], point_B[0, i]], [P[1, i], point_B[1, i]],color='y', linewidth=4)

    P5_circle = plt.Circle((P[:, i]), 0.08, color='g', fill=True)
    ax.add_artist(P5_circle)

    stL1= 'P'
    stL2 = 'A'
    stL3 = 'B'

    P2_text = ax.text(point_A[0, i], point_A[1, i] + 0.4, stL2, fontsize=10)
    P3_text = ax.text(point_B[0, i], point_B[1, i] + 0.4, stL3, fontsize=10)
    P1_text = ax.text(P[0, i], P[1, i] + 0.4, stL1, fontsize=10)

    plt.pause(0.05)

    if i < len(Teta2) - 1:
        LA.remove()
        LB.remove()
        LC.remove()
        LD.remove()     
        LE.remove()
        P3_text.remove()
        P2_text.remove()
        P1_text.remove()


