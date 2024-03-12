import numpy as np
from scipy.linalg import solve_continuous_are


def lqr(A, B, Q, R):
    """
    Solves the continuous time LQR controller.
    dx/dt = Ax + Bu
    cost = integral x.T*Q*x + u.T*R*u dt
    """
    # Solve Riccati equation
    P = solve_continuous_are(A, B, Q, R)
    
    # Compute the LQR gain
    K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
    
    return K


A = np.array([[0, 1], [-1, -1]])  # Example system dynamics matrix
B = np.array([[0], [1]])  # Example input matrix
Q = np.array([[1, 0], [0, 1]])  # State cost matrix
R = np.array([[1]])  # Input cost matrix

K = lqr(A, B, Q, R)
print("Optimal K:", K)
