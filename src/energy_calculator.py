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

def solve_max_cut_sbm(num_nodes, weight_matrix, biases, temperature, num_iterations=1000):
    """
    Solves the Max-Cut problem using Stochastic Boltzmann Machines (SBM) via Gibbs sampling.
    """
    states = np.random.choice([0, 1], size=num_nodes).astype(float)
    best_cut_value = -np.inf
    best_states = states.copy()

    for _ in range(num_iterations):
        for i in range(num_nodes):
            input_i = biases[i] + np.dot(weight_matrix[i, :], states)
            prob_1 = 1 / (1 + np.exp(-input_i / temperature))
            states[i] = 1 if np.random.rand() < prob_1 else 0

        current_cut_value = -calculate_energy(states, weight_matrix, biases)
        if current_cut_value > best_cut_value:
            best_cut_value = current_cut_value
            best_states = states.copy()

    return best_cut_value, best_states

