## SCDAA Assignment 2023-24

### Team:
* Ananya Jaishankar (s2538071) - $\frac{1}{3}$

* Frances McKnight (s2535018) - $\frac{1}{3}$

* Luz Pascual (s2571924) - $\frac{1}{3}$

## Description
This repository contains the code for the SCDAA Assignment, which aims to implement numerical algorithms to solve a stochastic control problem (Linear Quadratic Regulator) using a torch tensor structure, the “deep Galerkin method” for solving the linear PDE, combined with policy iteration. The chunk of our algorithm formulation and estimation is found in the Jupyter Notebook **report.ipynb**.

## Jupyter Notebook Breakdown

### 1. Solution of the Linear Quadratic Regulator problem
For replication purposes, the user must first run the codes under subsection *Exercise 1.1*, in which our *LQR class* is defined, which uses numerical methods from the package *scipy.integrate* to solve the Ricatti ODE and find the optimal Markov controls and value function.

An example usage was provided using example values to initiate the input matrices and batches needed to apply the previously defined *LQR class*. The user is free to change this example with their own values for their iniation of the LQR problem, all as torch tensors:
  - D, H, C, M: 2x2 matrices in Ricatti ODE,
  - R: Final condition for Ricatti ODE solution,
  - sigma: volatility parameter,
  - T: final time,
  - t_batch: batch of initial times to solve the Ricatti ODE with,
  - x_batch: batch of initial conditions of 2-dimensional X.

We then create a function to run Monte Carlo simulations of the process X, for given initial conditions, and run these simulations while varying (separately) number of samples to generate of X and number of time steps to use in X sample generation, under *Exercise 1.2*. The results are compared to the numerical solutions obtained in *Exercise 1.1* with the same initial conditions.

### 2. NNs as an alternative to estimate Markov Control and Value Function
Under *Exercise 2* the notebook includes two training functions, one per Neural Network (NN), that seek to estimate the value function and Markov control, as numerically estimated in *Exercise 1.1*. The training loss of each NN is then plotted against the epochs of training it endured. To run this, first run *Exercise 1.1* defining the LQR class and the initial conditions (which should be consistent throughout) in *Exercise 1.2*.
We point out that for computational power reasons, these functions were only tested using a small number of generated batches (100) of the X process, however the user is encouraged to modify this as necessary and run it for larger number of samples to get a more accurate estimation.

### 3. Applying the Deep Galerkin method to approximate the linear PDE
In *Exercise 3* we apply the Deep Galerkin method to approximate the linear Bellman PDE. To run this segment of the notebook, first run *Exercise 1.1*, the initial conditions (which should be consistent throughout) in *Exercise 1.2* and define the function *generate_x* which is found below *Exercise 1.2*.
The training loss for the DGM is then plotted against the epochs. We point out that for computational power reasons, these functions were only tested using a small number of generated batches (100) of the X process, however the user is encouraged to modify this as necessary and run it for larger number of samples to get a more accurate estimation.

### 4. Policy iteration
Under *Exercise 4*, we seek to implement the policy iteration algorithm and check that it converges to the results obtained with numerical methods. To run this segment of the notebook, first run *Exercise 1.1*.
The training loss for the DGM is then plotted against the epochs. We point out that for computational power reasons, these functions were only tested using a small number of generated batches (100) of the X process, however the user is encouraged to modify this as necessary and run it for larger number of samples to get a more accurate estimation.

