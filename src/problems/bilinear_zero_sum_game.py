# src/problems/bilinear_zero_sum_game.py

import numpy as np
from .base_problem import Problem

class BilinearZeroSumGameProblem(Problem):
    """
    Bilinear Zero-Sum Game Problem.

    Considers the problem:
        min_{x ∈ Δ^n} max_{y ∈ Δ^n} x^T A y

    where A ∈ R^{n x n} is the payoff matrix, and Δ^n is the probability simplex in R^n.

    The operator F(z) is given by:
        F(z) = [A y; -A^T x]

    where z = (x, y) ∈ R^{2n}.
    """

    def __init__(self, n, seed=None):
        """
        Initializes the BilinearZeroSumGameProblem with generated data.

        Parameters:
            n (int): Dimension of x and y individually (each in R^n).
            seed (int, optional): Seed for random number generators.
        """
        self.n_single = n  # Dimension of x and y individually

        # Initialize random number generator
        rng = np.random.default_rng(seed)

        # Generate payoff matrix A
        S = rng.normal(loc=0.0, scale=1.0, size=(n, n))
        self.A = S - S.T

        # Compute the Lipschitz constant L_F = || [0, A; -A^T, 0] ||_2
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

        # Compute F(z)
        F_x = self.A @ y
        F_y = - self.A.T @ x

        F_z = np.concatenate([F_x, F_y])

        return F_z

    def prox_g(self, z: np.ndarray, gamma: float) -> np.ndarray:
        """
        Computes the proximal operator of gamma * g at point z.

        Since g is the indicator function of Δ^n x Δ^n, the proximal operator is projection onto Δ^n x Δ^n.

        Parameters:
            z (np.ndarray): The point at which to compute the proximal operator, of shape (2n,).
            gamma (float): Step size parameter (not used here).

        Returns:
            np.ndarray: The result of the proximal operator at z, of shape (2n,).
        """
        n = self.n_single
        x = z[:n]
        y = z[n:]

        # Project x and y onto the probability simplex Δ^n
        x_proj = self._project_onto_simplex(x)
        y_proj = self._project_onto_simplex(y)

        z_proj = np.concatenate([x_proj, y_proj])

        return z_proj
    
    def _project_onto_simplex(self, v: np.ndarray) -> np.ndarray:
        """
        Projects a vector v onto the probability simplex Δ^n.

        Parameters:
            v (np.ndarray): Input vector of shape (n,).

        Returns:
            np.ndarray: The projection of v onto the simplex Δ^n.
        """
        # Algorithm from:
        # "Efficient Projections onto the l1-Ball for Learning in High Dimensions" (Duchi et al., ICML 2008)

        n = v.shape[0]
        u = np.sort(v)[::-1]  # Sort v in descending order
        cssv = np.cumsum(u) - 1
        ind = np.arange(n) + 1
        cond = u - cssv / ind > 0
        if not np.any(cond):
            # All entries are projected to zero
            w = np.zeros(n)
        else:
            rho = ind[cond][-1]
            theta = cssv[cond][-1] / rho
            w = np.maximum(v - theta, 0)
        return w

    def _compute_lipschitz_constant(self) -> float:
        """
        Computes the Lipschitz constant L_F of the operator F.

        Returns:
            float: The Lipschitz constant L_F.
        """
        # Construct the block matrix K = [0, A; -A^T, 0]
        zero_block = np.zeros((self.n_single, self.n_single))
        top_row = np.hstack([zero_block, self.A])
        bottom_row = np.hstack([-self.A.T, zero_block])
        K = np.vstack([top_row, bottom_row])

        # Compute the spectral norm (largest singular value) of K
        L_F = np.linalg.norm(K, 2)

        return L_F

    def get_plot_settings(self) -> dict:
        """
        Returns the plotting settings specific to this problem.

        Returns:
            dict: A dictionary containing plot labels and titles.
        """
        return {
            'y_label': r'$\|r^k\|$',
            'x_label_iterations': 'Iteration $k$',
            'x_label_F_evals': 'Number of $F$ evaluations',
            'x_label_operator_evals': 'Number of operator evaluations',
            'x_label_time': 'Time (s)',
            'title_prefix': 'Convergence on'
        }
