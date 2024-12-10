"""
settings.py
---------------------------------------------------------------------------------------------------------
This file contains global configurations and constants for the project.
Modify these settings to adjust the parameters of the Chaotic Boltzmann Machines.
---------------------------------------------------------------------------------------------------------
Purpose:

Contains global settings and constants shared across the project, such as:
Number of units (N).
Temperature (T).
Simulation time span (𝑡_span).
---------------------------------------------------------------------------------------------------------
"""

# Number of units in the Chaotic Boltzmann Machine
NUM_UNITS = 10

# Temperature of the system
TEMPERATURE = 1.0

# Simulation time span
TIME_SPAN = (0, 10)

# Number of time steps for evaluation
TIME_EVAL_STEPS = 500
