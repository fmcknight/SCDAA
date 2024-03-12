import numpy as np
from numpy.linalg import inv
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class LQR:
    def __init__(self, H, M, sigma, alpha, alpha_s, C, D, R, T) -> None:
        self.H = H 
        self.M = M
        self.sigma = sigma
        self.alpha = alpha 
        self.alpha_s = alpha_s
        self.C = C 
        self.D = D 
        self.R = R
        self.T = T

    
    def ricatti_ode(self, S, t):
        return -2*(np.transpose(self.H))*S+S*(self.M)*(inv(self.D))*(self.M)*S-self.C
    

    def solve_lqr(self):
        """
        Solves the LQR problem by integrating the Riccati equation backwards in time.
        """
        t_points = np.linspace(0, self.T, 1000)[::-1]  # Time points (reversed for backward integration)
        ST = np.zeros(self.R.shape)  # Terminal condition for S
        S = odeint(self.ricatti_ode, ST.ravel(), len(t_points))
        return S.reshape(-1, self.R.shape)
    
    
    def visualize_results(self, S):
        """
        Visualizes the solution of the Riccati equation or the state/control trajectories.
        """
        t_points = np.linspace(0, self.T, len(S))
        plt.plot(t_points, S[:, 0, 0])  # Example: Plotting first element of S over time
        plt.xlabel('Time')
        plt.ylabel('S[0,0]')
        plt.title('Solution of Riccati Equation over Time')
        plt.show()
    


# Example usage
H = np.array([[0, 0], [0, 0]])
M = np.array([[0, 0], [0, 0]])
sigma = np.array([[0, 0], [0, 0]])
alpha = np.array([[0, 0], [0, 0]])
alpha_s = np.array([[0, 0], [0, 0]])
C = np.array([[0, 0], [0, 0]])
D = np.array([[0, 0], [0, 0]])
R = np.array([[0, 0], [0, 0]])
T = np.array([[0, 0], [0, 0]])

lqr_system = LQR(H, M, sigma, alpha, alpha_s, C, D, R, T)
S_solution = lqr_system.solve_lqr()
lqr_system.visualize_results(S_solution)

