"""
main.py
--------
Main script to run the Chaotic Boltzmann Machines simulation.
"""
from matplotlib import pyplot as plt
from src.chaotic_bm import run_simulation
from src.graph_utils import create_weight_matrix
from src.energy_calculator import calculate_energy
from src.visualizations import plot_energy
import numpy as np
import pandas as pd
import os


def save_results_to_csv(t_eval, states, energies, filename="data/results/simulation_results.csv"):
    """
    Save the results of the simulation to a CSV file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = {
        "time": t_eval,
        "energy": energies,
    }
    for i in range(states.shape[1]):
        data[f"state_unit_{i}"] = states[:, i]

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def plot_states_over_time(t_eval, states):
    """
    Plot the binary states of each unit over time.
    """
    plt.figure(figsize=(10, 6))
    for i in range(states.shape[1]):
        plt.plot(t_eval, states[:, i] + i, label=f"Unit {i}")  # Offset to separate lines
    plt.xlabel("Time")
    plt.ylabel("States")
    plt.title("State Evolution of Units Over Time")
    plt.yticks(range(states.shape[1]), labels=[f"Unit {i}" for i in range(states.shape[1])])
    plt.legend(loc='upper right')
    plt.show()



def main():
    # Generate a random weight matrix
    num_nodes = 10
    edge_probability = 0.5
    weight_matrix = create_weight_matrix(num_nodes, edge_probability, weight_range=(-5, 5))
    biases = np.random.uniform(-1, 1, num_nodes)

    # Run simulation
    solution = run_simulation(biases, weight_matrix)

    # Calculate energy at each time step
    states = (solution.y.T > 0.5).astype(float)
    energies = [calculate_energy(state, weight_matrix, biases) for state in states]

    # Save results to file
    save_results_to_csv(solution.t, states, energies)

    # Visualize results
    plot_energy(solution.t, energies)


if __name__ == "__main__":
    main()
