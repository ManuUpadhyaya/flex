# src/problems/cournot_nash.py

import numpy as np
from .base_problem import Problem

class CournotNashProblem(Problem):
    """
    Cournot-Nash Equilibrium Problem for Oligopolistic Markets with Quadratic Costs.
    """

    def __init__(self, n, seed=None):
        """
        Initializes the Cournot-Nash Problem with generated data.

        Parameters:
            n (int): Number of players.
            seed (int, optional): Seed for random number generators.
        """
        self.n = n

        rng = np.random.default_rng(seed)

        valid = False
        while not valid:
        
            self.m_i_price = rng.uniform(150, 250, size=n)
        
            self.b_i_cost = rng.uniform(30, 50, size=n)
        
            self.T_i = rng.uniform(3, 7, size=n)
        
            self.d_i = rng.uniform(5, 20, size=n)
            # Sort d_i in increasing order
            self.d_i.sort()
            
            u_i = rng.uniform(-10, -5, size=n)
            
            self.a_i = self.d_i / u_i
            # Sort a_i in decreasing order
            self.a_i.sort()
            self.a_i = self.a_i[::-1]  # Reverse to get decreasing order

            # Check validity condition
            valid = True
            for i in range(n):
                if self.b_i_cost[i] < -2 * self.a_i[i] * self.T_i[i] or self.m_i_price[i] <= self.b_i_cost[i] or self.d_i[i] <= (-1) * self.a_i[i] :
                    valid = False
                    break  # Exit the loop if the condition is not met

        # Compute A and b for F(z) = A z + b
        self.A = np.zeros((n, n))
        for i in range(n):
            self.A[i, i] = 2 * (self.a_i[i] + self.d_i[i])
            for j in range(n):
                if j != i:
                    self.A[i, j] = self.d_i[i]

        # Compute b_i = b_i_cost - m_i_price
        self.b = self.b_i_cost - self.m_i_price

        # Compute Lipschitz constant L = ||A||_2 (spectral norm)
        L = np.linalg.norm(self.A, 2)

        # Initialize the base class
        super().__init__(dim_z=n, L=L)

    def F(self, z):
        """
        Evaluates the operator F at a given point z.

        Parameters:
            z (np.ndarray): The point at which to evaluate F.

        Returns:
            np.ndarray: The value of F at z.
        """
        if z.shape[0] != self.dim_z:
            raise ValueError(f"Input z must be of dimension {self.dim_z}.")

        return self.A @ z + self.b

    def prox_g(self, z, gamma):
        """
        Computes the proximal operator of gamma * g at point z.

        Since g is the indicator function of [0, T_i], the proximal operator is projection onto [0, T_i].

        Parameters:
            z (np.ndarray): The point at which to compute the proximal operator.
            gamma (float): Step size parameter (not used here).

        Returns:
            np.ndarray: The result of the proximal operator at z.
        """
        if z.shape[0] != self.dim_z:
            raise ValueError(f"Input z must be of dimension {self.dim_z}.")

        # Project z onto [0, T_i]
        return np.minimum(np.maximum(z, 0), self.T_i)

    def get_plot_settings(self):
        """
        Returns the plotting settings specific to this problem.

        Returns:
            dict: A dictionary containing plot labels and titles.
        """
        return {
            'y_label': r'$\| r^k \|$',  # Updated y-label
            'x_label_iterations': 'Iteration $k$',
            'x_label_F_evals': 'Number of $F$ evaluations',
            'x_label_operator_evals': 'Number of operator evaluations',
            'x_label_time': 'Time (s)',
            'title_prefix': 'Convergence on'
        }
