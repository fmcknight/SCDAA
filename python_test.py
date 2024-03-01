import numpy as np
import matplotlib.pyplot as plt

# Define system matrices
A = np.array([[1, 0.1, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0.1],
              [0, 0, 0, 1]])
B = np.array([[0, 0],
              [0, 0],
              [0, 0],
              [0, 0.1]])
C = np.eye(4)
D = np.zeros((4,2))

# Define cost matrices
Q = np.eye(4)
R = np.eye(2)

# Solve algebraic Riccati equation
P = np.matrix(np.zeros((4,4)))
max_iter = 150
tolerance = 1e-6
for _ in range(max_iter):
    P_new = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    if np.max(np.abs(P_new - P)) < tolerance:
        break
    P = P_new

K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

print("Feedback Gain (K):", K)

# Simulation
t = np.linspace(0, 10, 100)
x0 = np.array([1, 0, 1, 0])  # initial state

# Simulate the system with the calculated control gain
x = np.zeros((4, len(t)))
u = np.zeros((2, len(t)))
x[:, 0] = x0
for i in range(1, len(t)):
    u[:, i-1] = -K @ x[:, i-1]
    x[:, i] = A @ x[:, i-1] + B @ u[:, i-1]

# Plot results
plt.figure()
plt.subplot(211)
plt.plot(t, x[0], label='x1')
plt.plot(t, x[2], label='x2')
plt.ylabel('States')
plt.legend()

plt.subplot(212)
plt.plot(t[:-1], u[0], label='u1')
plt.plot(t[:-1], u[1], label='u2')
plt.ylabel('Control Inputs')
plt.xlabel('Time')
plt.legend()
plt.show()
