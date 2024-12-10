"""
energy_calculator.py
---------------------------------------------------------------------------------------------------------
This file provides functions to calculate the energy of the system.
The energy function is minimized during the optimization process of the Chaotic Boltzmann Machines.
---------------------------------------------------------------------------------------------------------
Purpose:

Functions for calculating the total energy of the system based on its states, weights, and biases.
---------------------------------------------------------------------------------------------------------
"""

import numpy as np

def calculate_energy(states, weights, biases):
    """
    Calculate the total energy of the system.
    """
    energy = -np.dot(biases, states) - np.sum(np.triu(np.outer(states, states) * weights))
    return energy
