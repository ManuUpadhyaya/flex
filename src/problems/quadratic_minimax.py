# src/problems/quadratic_minimax.py

import numpy as np
from numpy.random import default_rng
from .base_problem import Problem

class QuadraticMinimaxProblem(Problem):
    """
    Quadratic Convex-Concave Minimax Problem.

    The problem involves solving:
        min_x max_y L(x, y) = 0.5 * (x - x_*)^T A (x - x_*) + (x - x_*)^T C (y - y_*) - 0.5 * (y - y_*)^T B (y - y_*)

    where A and B are positive semidefinite matrices, and C is a real matrix.

    Attributes:
        n_single (int): Dimension of x and y individually.
        A (np.ndarray): Positive semidefinite matrix for x.
        B (np.ndarray): Positive semidefinite matrix for y.
        C (np.ndarray): Coupling matrix between x and y.
        x_star (np.ndarray): Optimal x value.
        y_star (np.ndarray): Optimal y value.
    """

    def __init__(self, n: int, alpha: float = 0.5, seed: int = None):
        """
        Initializes the QuadraticMinimaxProblem with generated data.

        Parameters:
            n (int): Dimension of x and y individually (each in R^n).
            alpha (float): Scaling parameter in [0, 1] for matrices A and B.
            seed (int, optional): Seed for random number generators.
        """
        self.n_single = n  # Dimension of x and y individually
        self.alpha = alpha  # Scaling parameter for A and B

        # Initialize random number generator
        rng = default_rng(seed)

        # Generate x_star and y_star
        self.x_star = rng.standard_normal(n)
        self.y_star = rng.standard_normal(n)

        # Generate matrix A
        self.A = self.alpha * self._generate_pd_matrix(n, rng)

        # Generate matrix B
        self.B = self.alpha * self._generate_pd_matrix(n, rng)

        # Generate matrix C 
        self.C = self._generate_matrix(n, rng)

        # Compute the Lipschitz constant L_F
        self.L = self._compute_lipschitz_constant()

        # Total dimension is 2n (for z = (x, y))
        super().__init__(dim_z=2 * n, L=self.L)

    def F(self, z: np.ndarray) -> np.ndarray:
        """
        Evaluates the operator F at a given point z = (x, y).

        Parameters:
            z (np.ndarray): The point at which to evaluate F, must be of dimension 2n.

        Returns:
            np.ndarray: The value of F at z, also of dimension 2n.

        Raises:
            ValueError: If the dimension of z does not match dim_z.
        """
        if z.shape[0] != self.dim_z:
            raise ValueError(f"Input z must be of dimension {self.dim_z} (which is 2n).")

        n = self.n_single  # Dimension of x and y individually
        x = z[:n]          # First n elements are x
        y = z[n:]          # Last n elements are y

        x_minus_x_star = x - self.x_star
        y_minus_y_star = y - self.y_star

        # Compute gradients with respect to x and y
        grad_x_L = self.A @ x_minus_x_star + self.C @ y_minus_y_star
        grad_y_L = - self.B @ y_minus_y_star + self.C.T @ x_minus_x_star

        # Compute F(z) = [grad_x_L; -grad_y_L]
        F_z = np.concatenate([grad_x_L, -grad_y_L])

        return F_z

    def Jacobian_F(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian matrix of the operator F at point z.

        For this quadratic problem, the Jacobian is constant and given by:
            J_F = [[A, C],
                [-C^T, B]]

        Parameters:
            z (np.ndarray): The point at which to evaluate the Jacobian (unused since it's constant).

        Returns:
            np.ndarray: The Jacobian matrix of shape (2n, 2n).
        """

        # Construct the Jacobian matrix J_F
        top_row = np.hstack([self.A, self.C])
        bottom_row = np.hstack([-self.C.T, self.B])
        J_F = np.vstack([top_row, bottom_row])

        return J_F

    def _generate_pd_matrix(self, n: int, rng) -> np.ndarray:
        """
        Generates a random positive definite (PD) matrix.

        Parameters:
            n (int): Dimension of the matrix.
            rng: Random number generator instance.

        Returns:
            np.ndarray: An n x n PD matrix.
        """
        S = self._generate_matrix(n, rng)
        S = (S + S.T) / 2

        min_eig = np.min(np.linalg.eigvalsh(S))
        S += (abs(min_eig) + 1) * np.eye(n)

        return S

    def _generate_matrix(self, n: int, rng) -> np.ndarray:
        """
        Generates a random matrix.

        Parameters:
            n (int): Dimension of the matrix.
            rng: Random number generator instance.

        Returns:
            np.ndarray: An n x n matrix.
        """
        M = rng.standard_normal((n, n)) / np.sqrt(n)
        return M

    def _compute_lipschitz_constant(self) -> float:
        """
        Computes the Lipschitz constant L_F of the operator F.

        Returns:
            float: The Lipschitz constant L_F.
        """
        top_row = np.hstack([self.A, self.C])
        bottom_row = np.hstack([-self.C.T, self.B])
        K = np.vstack([top_row, bottom_row])  # K is a (2n x 2n) matrix

        L_F = np.linalg.norm(K, 2)

        return L_F

    def get_plot_settings(self) -> dict:
        """
        Returns the plotting settings specific to this problem.

        Returns:
            dict: A dictionary containing plot labels and titles.
        """
        return {
            'y_label': r'$\|F(z^k)\|$',  # Specific to Quadratic Minimax Problem
            'x_label_iterations': 'Iteration $k$',
            'x_label_F_evals': 'Number of $F$ evaluations',
            'x_label_operator_evals': 'Number of operator evaluations', 
            'x_label_time': 'Time (s)',
            'title_prefix': 'Convergence on'
        }
