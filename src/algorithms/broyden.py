# src/algorithms/broyden.py

import numpy as np
from .direction import Direction

class Broyden(Direction):
    """
    Broyden's scheme with Powell's modification
    
    References:
    - Themelis & Patrinos. "SuperMann: A superlinearly convergent algorithm for 
      finding fixed points of nonexpansive operators."
    """

    def __init__(self, dim_z, gamma, fix_point_operator, theta_bar, 
                 algorithm_variant='g_zero'):
        """
        Initializes the Broyden direction strategy with Powell's modification.
        
        Parameters:
            dim_z (int): Dimension of the problem variable z.
            gamma (float): Step size parameter (in (0, 1/L_F)).
            fix_point_operator (str): "EG" or "FB", specifying the fixed-point operator.
            theta_bar (float): Parameter for Powell's modification.
            algorithm_variant (str): Either "g_zero" or "g_nonzero".
        """
        super().__init__()
        self.dim_z = dim_z
        self.gamma = gamma
        self.fix_point_operator = fix_point_operator
        self.theta_bar = theta_bar
        if algorithm_variant not in ("g_zero", "g_nonzero"):
            raise ValueError("algorithm_variant must be 'g_zero' or 'g_nonzero'.")
        self.algorithm_variant = algorithm_variant

        #self.Hk = np.eye(self.dim_z)
        np.random.seed(42)  
        self.Hk = np.diag(np.random.uniform(0, 1, self.dim_z)) 


    def compute_direction(self, zk, zbar, wk, Fzk, Fzbar, iteration):
        """
        Computes the direction d^k using Broyden's scheme with Powell's modification.

        Parameters (Direction interface):
            zk (np.ndarray): Current iterate z^k.
            zbar (np.ndarray): \bar{z}^k = z^k - gamma * F(z^k).
            wk (np.ndarray): w^k = z^k - gamma * F(\bar{z}^k).
            Fzk (np.ndarray): F(z^k).
            Fzbar (np.ndarray): F(\bar{z}^k).
            iteration (int): Current iteration number.

        Returns:
            (d, success):
                d (np.ndarray): Direction d^k (size = dim_z).
                success (bool): True if the direction was successfully computed.
        """
        # Broyden direction depends on algorithm_variant and fix_point_operator
        if self.algorithm_variant == "g_zero":
            if self.fix_point_operator == "EG":
                # For "EG": direction = -Hk @ [gamma * Fzbar]
                d = -self.Hk @ (self.gamma * Fzbar)
            elif self.fix_point_operator == "FB":
                # For "FB": direction = -Hk @ [gamma * Fzk]
                d = -self.Hk @ (self.gamma * Fzk)
            else:
                raise ValueError("fix_point_operator must be 'EG' or 'FB'.")
        else:  # algorithm_variant == "g_nonzero"
            if self.fix_point_operator == "EG":
                # direction = -Hk @ (zk - wk)
                d = -self.Hk @ (zk - wk)
            elif self.fix_point_operator == "FB":
                # direction = -Hk @ (zk - zbar)
                d = -self.Hk @ (zk - zbar)
            else:
                raise ValueError("fix_point_operator must be 'EG' or 'FB'.")

        success = True
        return d, success

    def post_step_update(self, zk, Fzk, zbar, Fzbar, wk,
                         zk_plus_1, Fzk_plus_1, zbar_plus_1,
                         Fzbar_plus_1, wk_plus_1, iteration):
        """
        Updates the quasi-Newton matrix Hk using Powell's modification.

        Parameters (Direction interface):
            zk (np.ndarray): z^k
            Fzk (np.ndarray): F(z^k)
            zbar (np.ndarray): \bar{z}^k
            Fzbar (np.ndarray): F(\bar{z}^k)
            wk (np.ndarray): w^k
            zk_plus_1 (np.ndarray): z^{k+1}
            Fzk_plus_1 (np.ndarray): F(z^{k+1})
            zbar_plus_1 (np.ndarray): \bar{z}^{k+1}
            Fzbar_plus_1 (np.ndarray): F(\bar{z}^{k+1})
            wk_plus_1 (np.ndarray): w^{k+1}
            iteration (int): Current iteration number.
        """
        s = zk_plus_1 - zk  # step s^k

        # Compute y based on fix_point_operator and algorithm_variant
        if self.algorithm_variant == "g_zero":
            if self.fix_point_operator == "EG":
                # y = gamma * F(\bar{z}^{k+1}) - gamma * F(\bar{z}^{k})
                y = self.gamma * Fzbar_plus_1 - self.gamma * Fzbar
            elif self.fix_point_operator == "FB":
                # y = gamma * F(z^{k+1}) - gamma * F(z^k)
                y = self.gamma * Fzk_plus_1 - self.gamma * Fzk
            else:
                raise ValueError("fix_point_operator must be 'EG' or 'FB'.")
        else:  # algorithm_variant == "g_nonzero"
            if self.fix_point_operator == "EG":
                # y = (z^{k+1}-w^{k+1}) - (z^k - w^k)
                y = (zk_plus_1 - wk_plus_1) - (zk - wk)
            elif self.fix_point_operator == "FB":
                # y = (z^{k+1}-zbar^{k+1}) - (z^k - zbar^k)
                y = (zk_plus_1 - zbar_plus_1) - (zk - zbar)
            else:
                raise ValueError("fix_point_operator must be 'EG' or 'FB'.")

        H_y = self.Hk @ y
        s_norm_squared = np.dot(s, s)

        gamma_ = np.dot(H_y, s) / s_norm_squared

        if abs(gamma_) >= self.theta_bar:
            theta = 1.0
        elif gamma_ == 0:
            theta = (1 - self.theta_bar) / (1 - gamma_) 
        else:
            theta = (1 - np.sign(gamma_) * self.theta_bar) / (1 - gamma_)

        H_y_tilde = (1 - theta) * s + theta * H_y

        numerator = np.outer(s - H_y_tilde, self.Hk.T @ s)
        denominator = np.dot(H_y_tilde, s)

        self.Hk += numerator / denominator
