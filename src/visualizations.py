"""
visualizations.py
---------------------------------------------------------------------------------------------------------
This file contains functions for visualizing results, such as energy evolution,
binary state transitions, and performance comparisons.
---------------------------------------------------------------------------------------------------------
Purpose:

Provides plotting functions to visualize results, such as:
Energy evolution over time.
State transitions.
Comparison of results with traditional Boltzmann Machines.
---------------------------------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt

def plot_energy(t_eval, energies):
    """
    Plot the energy evolution over time.
    """
    plt.figure()
    plt.plot(t_eval, energies, label="Energy")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy Evolution in CBM")
    plt.legend()
    plt.show()
