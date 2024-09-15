
# Mechanism Simulation - Four-Bar and Slider-Crank Mechanisms

## Author: Taha Faghani

This project contains codes for simulating two classical mechanisms: the **four-bar mechanism** and the **slider-crank mechanism**. These codes were developed for educational purposes, as part of my work as a Teaching Assistant (TA), and they are implemented in both Python and MATLAB.

## Project Structure

### Python Codes:
- **`fourbar.py`**: Simulates the motion of a four-bar mechanism, optimizes the link lengths using Particle Swarm Optimization (PSO), and visualizes the optimized mechanism's motion.
- **`slider.py`**: Simulates the motion of a slider-crank mechanism in Python using NumPy and Matplotlib, providing real-time visualizations of the mechanism's dynamics.

### MATLAB Codes:
- **`slider.m`**: Simulates the motion of a slider-crank mechanism in MATLAB, showcasing the real-time plotting and animation of the crank and slider motion.

## Code Descriptions

### Four-Bar Mechanism (`fourbar.py`)
This code simulates a four-bar mechanism, which consists of four rigid links connected by revolute joints. The goal is to trace the path of a specific point on the coupler link. The optimization of link lengths is performed using **Particle Swarm Optimization (PSO)** to minimize the deviation from a target path (a circular trajectory in this case).

#### Key Features:
- Uses PSO to find optimal link lengths.
- Calculates the position, velocity, and acceleration of the coupler point.
- Visualizes the mechanism in motion and compares the actual path of the coupler with the target circular path.

### Slider-Crank Mechanism in Python (`slider.py`)
This code simulates a **slider-crank mechanism**, which consists of a rotating crank connected to a sliding block by a connecting rod. The goal is to simulate the slider motion as the crank rotates.

#### Key Features:
- Simulates the motion of the crank and slider.
- Uses **Matplotlib** for visualization.
- Real-time plotting of the slider and crank positions during the simulation.

### Slider-Crank Mechanism in MATLAB (`slider.m`)
This MATLAB script also simulates the motion of a **slider-crank mechanism**. It produces a real-time visualization of the crank and slider moving.

#### Key Features:
- Simulates the slider-crank motion using built-in MATLAB functions.
- Real-time plotting and animation of the mechanism’s movement.

## How to Run the Codes

### Python:
1. Ensure you have the required dependencies installed:

   ```bash
   pip install numpy matplotlib pyswarm
   ```

2. To run the **four-bar mechanism** simulation:

   ```bash
   python fourbar.py
   ```

3. To run the **slider-crank mechanism** simulation:

   ```bash
   python slider.py
   ```

### MATLAB:
1. Open MATLAB.
2. Navigate to the directory containing `slider.m`.
3. Run the `slider.m` script by typing:

   ```matlab
   slider
   ```

## Visualization

Both the Python and MATLAB codes include real-time visualizations of the mechanisms, allowing you to see the motion of the links and the paths traced by specific points in the mechanism.


<p align="center">
  <img src="https://github.com/tahafaghani/mechanisms/blob/main/4-bar.png" width="45%" alt="4bar"/>
</p>


<p align="center">
  <img src="https://github.com/tahafaghani/mechanisms/blob/main/slider.png" width="45%" alt="Slider"/>
</p>


