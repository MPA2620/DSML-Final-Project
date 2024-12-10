"""
chaotic_bm.py
---------------------------------------------------------------------------------------------------------
This file implements the Chaotic Boltzmann Machines (CBM).
It includes functions to initialize states, update states deterministically,
and integrate the system dynamics over time.
---------------------------------------------------------------------------------------------------------
Purpose:

Implements the core of Chaotic Boltzmann Machines.
Handles differential equations, state updates, and numerical integration.
---------------------------------------------------------------------------------------------------------
"""

import numpy as np
from scipy.integrate import solve_ivp
from config.settings import NUM_UNITS, TEMPERATURE, TIME_SPAN, TIME_EVAL_STEPS

def initialize_states(num_units):
    """
    Initialize the internal continuous states and binary states of the system.
    """
    internal_states = np.random.rand(num_units)  # Continuous states
    binary_states = (internal_states > 0.5).astype(float)  # Binary states
    return internal_states, binary_states

def chaotic_dynamics(t, x, b, W, T):
    """
    Defines the differential equations for the CBM dynamics.
    """
    s = (x > 0.5).astype(float)  # Binary states
    z = b + np.dot(W, s)  # Input to each unit
    exp_term = np.exp(np.clip((1 - 2 * s) * z / T, -50, 50))  # Limit exponential to avoid overflow
    dx_dt = (1 - 2 * s) * (1 + exp_term)
    return dx_dt

def run_simulation(biases, weights):
    """
    Runs the CBM simulation over the specified time span.
    """
    x0, _ = initialize_states(NUM_UNITS)
    t_eval = np.linspace(TIME_SPAN[0], TIME_SPAN[1], TIME_EVAL_STEPS)

    print("Initial states:", x0)
    print("Biases:", biases)
    print("Weights:", weights)

    solution = solve_ivp(
        chaotic_dynamics,
        TIME_SPAN,
        x0,
        args=(biases, weights, TEMPERATURE),
        t_eval=t_eval
    )
    print("Final states:", solution.y[:, -1])
    return solution

