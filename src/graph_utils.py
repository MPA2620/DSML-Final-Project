"""
graph_utils.py
---------------------------------------------------------------------------------------------------------
This file contains utility functions for handling graph-related operations.
Includes graph creation, weight matrix generation, and plotting utilities.
---------------------------------------------------------------------------------------------------------
Purpose:

Functions for creating and manipulating graphs for optimization problems, such as the Maximum Cut Problem.
---------------------------------------------------------------------------------------------------------
"""

import numpy as np


def create_weight_matrix(num_nodes, edge_probability, weight_range=(1, 10)):
    """
    Create a random weight matrix for a graph.

    Parameters:
    - num_nodes: int, number of nodes in the graph.
    - edge_probability: float, probability of an edge existing between two nodes.
    - weight_range: tuple, the range of weights for the edges.

    Returns:
    - weight_matrix: np.ndarray, symmetric matrix with weights.
    """
    # Generate a random adjacency matrix
    adjacency_matrix = np.random.rand(num_nodes, num_nodes) < edge_probability

    # Assign random weights to the edges
    weights = np.random.randint(weight_range[0], weight_range[1], size=(num_nodes, num_nodes))

    # Create a symmetric weight matrix
    weight_matrix = adjacency_matrix * weights
    weight_matrix = np.triu(weight_matrix, 1)  # Keep only the upper triangle
    weight_matrix += weight_matrix.T  # Symmetrize the matrix

    return weight_matrix