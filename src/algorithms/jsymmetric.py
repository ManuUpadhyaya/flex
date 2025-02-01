# src/algorithms/jsymmetric.py

import numpy as np
from .direction import Direction
import warnings

class Jsymmetric(Direction):
    """
    J-symmetric quasi-Newton direction strategy.
    
    References:
        - Asl, Lu, Yang. "A J-symmetric quasi-Newton method for minimax problems."
    """

    def __init__(self, dim_x, dim_z):
        """
        Initializes the J-symmetric direction logic.

        Parameters:
            dim_x (int): Dimension of the x part in the variable z = (x, y).
            dim_z (int): Total dimension of z (dim_x + dim_y).
        """
        super().__init__()

        if dim_x > dim_z:
            raise ValueError("dim_x must be <= dim_z.")

        self.dim_x = dim_x
        self.dim_y = dim_z - dim_x
        self.dim_z = dim_z

        I_x = np.eye(self.dim_x)
        I_y = np.eye(self.dim_y)
        self.J = np.block([[I_x,                                np.zeros((self.dim_x, self.dim_y))],
                           [np.zeros((self.dim_y, self.dim_x)), -I_y]
                           ])

        # sign_vector = [1,...,1, -1,...,-1] (dim_x ones, dim_y -ones)
        self.sign_vector = np.concatenate([np.ones(self.dim_x), -np.ones(self.dim_y)])

        self.H_k = np.eye(self.dim_z)
        self.B_k = np.eye(self.dim_z) # !!!!!
        self.I_z = np.eye(self.dim_z)

    def compute_direction(self, zk, zbar, wk, Fzk, Fzbar, iteration):
        """
        Computes the direction d^k using the J-symmetric quasi-Newton strategy.

        Parameters (Direction interface):
            zk (np.ndarray): Current iterate z^k.
            zbar (np.ndarray): Extrapolated point z^k - gamma * F(z^k).
            wk (np.ndarray): Intermediate point z^k - gamma * F(zbar^k).
            Fzk (np.ndarray): F(z^k).
            Fzbar (np.ndarray): F(zbar^k).
            iteration (int): Current iteration number.

        Returns:
            (d, success):
                d (np.ndarray): The direction d^k (size = dim_z).
                success (bool): True if direction was successfully computed.
        """
        d = -self.H_k @ Fzk
        success = True
        return d, success

    def post_step_update(self, zk, Fzk, zbar, Fzbar, wk,
                         zk_plus_1, Fzk_plus_1, zbar_plus_1,
                         Fzbar_plus_1, wk_plus_1, iteration):
        """
        Updates the quasi-Newton matrices H_k and B_k using the J-symmetric update.

        Parameters (Direction interface):
            zk, Fzk, zbar, Fzbar, wk: Variables from the current iteration.
            zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1: Next iteration variables.
            iteration (int): Current iteration number.
        """
        try:
            with warnings.catch_warnings(record=True) as w:
                # Compute s_k and y_k
                s_k = zk_plus_1 - zk      # shape: (dim_z,) !!!!!
                y_k = Fzk_plus_1 - Fzk      # shape: (dim_z,)

                r = y_k - self.B_k @ s_k  # shape: (dim_z,) !!!!!

                # s_k^T s_k
                sTs = np.dot(s_k, s_k)  # scalar

                # Update B_k !!!!!
                self.B_k += (
                    (self.sign_vector[:, None] * np.outer(s_k, r) * self.sign_vector[None, :]) / sTs
                    + np.outer(r, s_k) / sTs
                    - (np.dot(self.sign_vector * s_k, r) * np.outer(self.sign_vector * s_k, s_k)) / (sTs ** 2)
                )

                # J-symmetric update for H_k
                I_minus_ssT = self.I_z - np.outer(s_k, s_k) / sTs
                H_J_P_J = self.H_k @ (self.sign_vector[:, None] * I_minus_ssT * self.sign_vector[None, :])

                denom_1 = sTs + np.dot(s_k, H_J_P_J @ r)
                Qinv = self.H_k - (H_J_P_J @ np.outer(r, s_k) @ self.H_k) / denom_1

                denom_2 = sTs + np.dot(self.sign_vector * r, Qinv @ (self.sign_vector * s_k))
                self.H_k = Qinv - (
                    (Qinv @ (self.sign_vector[:, None] * np.outer(s_k, self.sign_vector * r)) @ Qinv) / denom_2
                )
        except RuntimeWarning as e:
            self.H_k = np.eye(self.dim_z)
            self.B_k = np.eye(self.dim_z)
