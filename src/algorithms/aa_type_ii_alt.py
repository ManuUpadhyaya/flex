# src/algorithms/aa_type_ii_alt.py

import numpy as np
from .direction import Direction

class AAII_alt(Direction):
    """
    Type-II Anderson Acceleration (Now inherits from Direction).

    Based on Algorithm 1 in:
    - "Anderson Acceleration of Proximal Gradient Methods"
      by Vien V. Mai and Mikael Johansson.
    """

    def __init__(self, dim_z, m_memory, fix_point_operator):
        """
        Initializes Type-II Anderson Acceleration.

        Parameters:
            dim_z (int): Dimension of iterate z.
            m_memory (int): Memory parameter for Anderson Acceleration.
            fix_point_operator (str): "EG" or "FB" for the fixed-point operator.
        """
        super().__init__()
        self.dim_z = dim_z
        self.m_memory = m_memory
        if fix_point_operator not in ("EG", "FB"):
            raise ValueError("fix_point_operator must be 'EG' or 'FB'.")
        self.fix_point_operator = fix_point_operator

        self.m_k = -1
        self.R_k = np.empty((self.dim_z, 0)) 
        self.gs = np.empty((self.dim_z, 0))

    def compute_direction(self, zk, zbar, wk, Fzk, Fzbar, iteration):
        """
        Computes the direction d^k using Anderson Acceleration Type-II.

        Parameters (following the Direction interface):
            zk (np.ndarray): Current iterate z^k.
            zbar (np.ndarray): \bar{z}^k = z^k - gamma * F(z^k).
            wk (np.ndarray): w^k = z^k - gamma * F(\bar{z}^k).
            Fzk (np.ndarray): F(z^k).
            Fzbar (np.ndarray): F(\bar{z}^k).
            iteration (int): Current iteration number.

        Returns:
            (d, success):
                d (np.ndarray): The direction d^k (same dimension as zk).
                success (bool): True if acceleration succeeded, False otherwise.
        """
        acceleration_failed = False

        if self.m_k < self.m_memory - 1:
            self.m_k += 1
        else:
            self.gs = self.gs[:, 1:]
            self.R_k = self.R_k[:, 1:]

        if self.fix_point_operator == "EG":
            g = wk
            r = wk - zk
        else:
            g = zbar
            r = zbar - zk

        self.gs = np.column_stack((self.gs, g))
        self.R_k = np.column_stack((self.R_k, r))

        if self.R_k.shape[1] == 0:
            d = None
            success = False
        else:
            mat = self.R_k.T @ self.R_k
            try:
                ones_vec = np.ones(self.R_k.shape[1])
                alpha = np.linalg.solve(mat, ones_vec)
                alpha = alpha / np.sum(alpha)
                g_aa = self.gs @ alpha
                d = g_aa - zk
                success = True
            except np.linalg.LinAlgError:
                self.m_k = -1
                self.R_k = np.empty((self.dim_z, 0))
                self.gs = np.empty((self.dim_z, 0))
                acceleration_failed = True
                d = None
                success = False

        if acceleration_failed:
            success = False
            d = None

        return d, success

    def post_step_update(self, zk, Fzk, zbar, Fzbar, wk,
                         zk_plus_1, Fzk_plus_1, zbar_plus_1,
                         Fzbar_plus_1, wk_plus_1, iteration):
        """
        Post-step update for Type-II Anderson Acceleration.

        Parameters (following the Direction interface):
            zk, Fzk, zbar, Fzbar, wk: Variables from the current iteration.
            zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1: Next iteration variables.
            iteration (int): Current iteration number.
        """
        pass
