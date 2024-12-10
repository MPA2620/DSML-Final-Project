"""
main.py
--------
Main script to run the Chaotic Boltzmann Machines simulation.
"""
import time
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from src.chaotic_bm import run_simulation
from src.graph_utils import create_weight_matrix
from src.energy_calculator import calculate_energy, solve_max_cut_sbm


def save_results_to_csv(results_summary, filename="data/results/simulation_results.csv"):
    """
    Save the results of CBM and SBM comparisons to a CSV file using Pandas.
    """
    print("Saving results to CSV...")
    print("Results Summary:", results_summary)

    # Convert results to a Pandas DataFrame
    data = {
        "Graph Size": results_summary["graph_sizes"],
        "CBM Cut Value": [r["cut_value"] for r in results_summary["CBM"]],
        "CBM Time (s)": [r["time"] for r in results_summary["CBM"]],
        "SBM Cut Value": [r["cut_value"] for r in results_summary["SBM"]],
        "SBM Time (s)": [r["time"] for r in results_summary["SBM"]],
    }
    df = pd.DataFrame(data)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def has_converged(energies, threshold=1e-3):
    if len(energies) > 10:
        return np.std(energies[-10:]) < threshold
    return False

def solve_max_cut_final(num_nodes, edge_probability, weight_range=(-5, 5), temperature=1.0):
    """
    Solves the Max-Cut problem using Chaotic Boltzmann Machines (CBM).
    """
    from src.graph_utils import create_weight_matrix

    # Step 1: Create a random weight matrix
    weight_matrix = create_weight_matrix(num_nodes, edge_probability, weight_range)
    biases = np.random.uniform(-1, 1, num_nodes)

    # Step 2: Run the CBM simulation
    solution = run_simulation(
        biases, weight_matrix, num_nodes, temperature=0.8, t_span=(0, 5), time_eval_steps=150
    )

    # Step 3: Extract binary states from the solution
    final_states = (solution.y[:, -1] > 0.5).astype(float)

    # Step 4: Calculate energy at each step
    energies = [calculate_energy(state, weight_matrix, biases) for state in solution.y.T]

    # Early stop if converged
    if has_converged(energies):
        print("CBM converged successfully.")

    # Step 5: Calculate final energy and cut value
    final_energy = calculate_energy(final_states, weight_matrix, biases)
    cut_value = -final_energy

    return cut_value, final_states, final_energy, solution.t, energies

def test_cbm_sbm(graph_sizes, edge_probability, weight_range=(-10, 10), temperature=1.0):
    """
    Tests CBM and SBM on graphs of varying sizes and generates performance comparisons.
    """
    results_summary = {"CBM": [], "SBM": [], "graph_sizes": []}

    for idx, size in enumerate(graph_sizes):
        print(f"\nTesting graph with {size} nodes...")

        # Generate graph
        weight_matrix = create_weight_matrix(size, edge_probability, weight_range)
        num_nodes = weight_matrix.shape[0]
        biases = np.zeros(num_nodes)

        # CBM
        start_time_cbm = time.time()
        cut_value_cbm, states_cbm, energy_cbm, t_eval, energies = solve_max_cut_final(
            size, edge_probability, weight_range, temperature
        )
        time_cbm = time.time() - start_time_cbm

        # Plot energy only for the first graph
        if idx == 0:
            plot_energy_over_time(t_eval, energies)

        # SBM
        start_time_sbm = time.time()
        cut_value_sbm, states_sbm = solve_max_cut_sbm(
            size, weight_matrix, biases, temperature, num_iterations=1000
        )
        time_sbm = time.time() - start_time_sbm

        # Save results
        results_summary["CBM"].append({"cut_value": cut_value_cbm, "time": time_cbm})
        results_summary["SBM"].append({"cut_value": cut_value_sbm, "time": time_sbm})
        results_summary["graph_sizes"].append(size)

        print(f"CBM -> Cut Value: {cut_value_cbm}, Time: {time_cbm:.4f} sec")
        print(f"SBM -> Cut Value: {cut_value_sbm}, Time: {time_sbm:.4f} sec")

    return results_summary

def plot_results(results_summary):
    """
    Plots the results of CBM and SBM comparisons.
    """
    sizes = results_summary["graph_sizes"]
    cbm_values = [r["cut_value"] for r in results_summary["CBM"]]
    cbm_times = [r["time"] for r in results_summary["CBM"]]
    sbm_values = [r["cut_value"] for r in results_summary["SBM"]]
    sbm_times = [r["time"] for r in results_summary["SBM"]]

    # Plot Cut Values
    plt.figure()
    plt.plot(sizes, cbm_values, label="CBM Cut Value", marker="o")
    plt.plot(sizes, sbm_values, label="SBM Cut Value", marker="x")
    plt.xlabel("Graph Size")
    plt.ylabel("Max-Cut Value")
    plt.title("Max-Cut Value Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Times
    plt.figure()
    plt.plot(sizes, cbm_times, label="CBM Time", marker="o")
    plt.plot(sizes, sbm_times, label="SBM Time", marker="x")
    plt.xlabel("Graph Size")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time Comparison")
    plt.legend()
    plt.grid()
    plt.show()


def plot_energy_over_time(t_eval, energies):
    """
    Plots the energy of CBM over time.
    """
    plt.figure()
    plt.plot(t_eval, energies, label="Energy")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy Evolution in CBM")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    graph_sizes = [10, 30]
    edge_probability = 0.6
    temperature = 0.5
    results = test_cbm_sbm(graph_sizes, edge_probability, weight_range=(-10, 10))
    save_results_to_csv(results, filename="data/results/simulation_results.csv")
    plot_results(results)
