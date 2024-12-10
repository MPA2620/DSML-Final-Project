# DSML-Final-Project

---

# Chaotic Boltzmann Machines for Optimization

## Project Overview
This project investigates the use of **Chaotic Boltzmann Machines (CBM)** as a deterministic alternative to 
traditional Boltzmann Machines (BM). CBM leverages chaotic dynamics to mimic the stochastic behavior of BM, 
offering potential advantages in computational efficiency and hardware implementation. 
The project focuses on applying CBM to solve optimization tasks, specifically the **Maximum Cut Problem**, 
and compares its performance to traditional BM.

---

## Mathematical Explanation

### Chaotic Boltzmann Machines
Chaotic Boltzmann Machines are deterministic systems governed by chaotic dynamics, designed to emulate the 
stochastic behavior of traditional Boltzmann Machines. The model operates as follows:

1. **State Evolution**:  
   Each unit in the CBM evolves according to a differential equation:
   $$
   \frac{dx_i}{dt} = (1 - 2s_i) \cdot \left(1 + \exp\left[(1 - 2s_i) z_i / T\right]\right)
   $$
   where:
   - \(x_i\): The internal continuous variable of unit \(i\), oscillating between 0 and 1.
   - \(s_i in \{0, 1\}\): The binary state of unit \(i\).
   - \(T\): The system temperature, influencing the randomness of transitions.
   - \(z_i\): The total input to unit \(i\), defined as:
     $$
     z_i = b_i + \sum_{j} W_{ij} s_j
     $$
     with \(b_i\) as the bias term and \(W_{ij}\) as the weight matrix.

2. **State Updates**:  
   The binary state \(s_i\) changes deterministically based on the value of \(x_i\):
   $$
   s_i = 
   \begin{cases} 
   0, & \text{if } x_i \leq 0 \\
   1, & \text{if } x_i \geq 1 
   \end{cases}
   $$

3. **Energy Function**:  
   Similar to traditional BM, the CBM minimizes a global energy function:
   $$
   E(s) = - \sum_{i} b_i s_i - \sum_{i < j} W_{ij} s_i s_j
   $$
   where \(s = (s_1, s_2, ..., s_N)\) represents the states of all units.

### Advantages of CBM
- Eliminates the need for random number generation, which is computationally expensive.
- Supports parallelism, making it more suitable for hardware implementations.
- Offers deterministic dynamics while maintaining comparable performance to stochastic models.

---

## Python Simulations

The CBM model is implemented and tested using numerical simulations. The key steps include:

1. **Model Initialization**:  
   - Define the number of units \(N\), temperature \(T\), weight matrix \(W\), and bias vector \(b\).
   - Initialize the internal states \(x_i\) and binary states \(s_i\).

2. **Solving the Differential Equations**:  
   The CBM dynamics are simulated by integrating the differential equations over a defined time period. 
This approach ensures the system evolves toward a minimum energy configuration.

3. **Optimization Task**:  
   The **Maximum Cut Problem** is used to evaluate the CBM. The graph structure is represented by the weight matrix \(W\), 
and the CBM identifies the optimal partition of the graph's nodes.

4. **Energy Monitoring**:  
   During the simulation, the system's energy is calculated to verify convergence and evaluate performance.

5. **Performance Comparison**:  
   The results are compared to those of traditional Boltzmann Machines, focusing on:
   - Accuracy of the solutions.
   - Computational efficiency (runtime and resource usage).
   - Scalability to larger problems.

---

## Results and Insights

### Observations
1. **Convergence**:  
   CBM successfully minimizes the energy function, providing solutions to the Maximum Cut Problem that are comparable to traditional BM.

2. **Efficiency**:  
   By avoiding stochastic processes, CBM demonstrates significant computational advantages, 
especially in scenarios where random number generation is a bottleneck.

3. **Scalability**:  
   The deterministic nature of CBM and its compatibility with parallel computation make it suitable for larger networks and 
hardware implementation.

---

## Future Work
- Extend simulations to larger and more complex graphs to assess scalability.
- Investigate the implementation of CBM on parallel hardware architectures.
- Explore additional applications of CBM in machine learning and optimization tasks.
