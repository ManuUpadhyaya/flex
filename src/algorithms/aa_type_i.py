import numpy as np
from .direction import Direction

class AAI(Direction):
    """
    Type-I Anderson Acceleration strategy.

    Based on Section 2.3 in:
    - Globally Convergent Type-I Anderson Acceleration for Non-Smooth Fixed-Point Iterations
    - By Junzi Zhang, Brendan O'Donoghue, and Stephen Boyd.
    """

    def __init__(self, dim_z, m_memory, fix_point_operator):
        """
        Initializes Type-I Anderson Acceleration.

        Parameters:
            dim_z (int): Dimension of the iterate z.
            m_memory (int): Memory parameter for Anderson Acceleration.
            fix_point_operator (str): "EG" or "FB" to choose the fixed-point operator.
        """
        super().__init__()
        
        self.dim_z = dim_z
        self.m_memory = m_memory
        if fix_point_operator not in ("EG", "FB"):
            raise ValueError("fix_point_operator must be 'EG' or 'FB'.")
        self.fix_point_operator = fix_point_operator

        self.m_k = -1
        self.Y_k = np.empty((self.dim_z, 0))
        self.S_k = np.empty((self.dim_z, 0))
        self.I = np.eye(self.dim_z)
        self.g_prev = None
        self.z_prev = None

    def compute_direction(self, zk, zbar, wk, Fzk, Fzbar, iteration):
        """
        Computes the direction d^k using Anderson Acceleration Type-I.

        Parameters (following the Direction interface):
            zk (np.ndarray): Current iterate z^k.
            zbar (np.ndarray): \bar{z}^k = z^k - gamma * F(z^k).
            wk (np.ndarray): w^k = z^k - gamma * F(\bar{z}^k).
            Fzk (np.ndarray): F(z^k).
            Fzbar (np.ndarray): F(\bar{z}^k).
            iteration (int): Current iteration number.

        Returns:
            (d, success):
                d (np.ndarray): The direction d^k (size = dim_z).
                success (bool): True if acceleration was successful; False otherwise.
        """
        if self.fix_point_operator == "EG":
            gk = zk - wk        
        else:
            gk = zk - zbar

        acceleration_failed = False

        if iteration > 1 and self.m_memory > 0:
            if self.m_k < self.m_memory - 1:
                self.m_k += 1
            else:
                self.Y_k = self.Y_k[:, 1:]
                self.S_k = self.S_k[:, 1:]

            yk = gk - self.g_prev
            sk = zk - self.z_prev

            self.Y_k = np.column_stack((self.Y_k, yk))
            self.S_k = np.column_stack((self.S_k, sk))

            mat = self.S_k.T @ self.Y_k
            try:
                inv_mat = np.linalg.solve(mat, self.S_k.T)
                H_k = self.I + (self.S_k - self.Y_k) @ inv_mat
                d = -H_k @ gk
                success = True
            except np.linalg.LinAlgError:
                self.m_k = -1
                self.Y_k = np.empty((self.dim_z, 0))
                self.S_k = np.empty((self.dim_z, 0))
                success = False
                d = None
        else:
            acceleration_failed = True
            success = False
            d = None

        if acceleration_failed:
            success = False
            d = None

        return d, success

    def post_step_update(self, zk, Fzk, zbar, Fzbar, wk,
                         zk_plus_1, Fzk_plus_1, zbar_plus_1,
                         Fzbar_plus_1, wk_plus_1, iteration):
        """
        Post-step update for Anderson Acceleration (store the new iterate and residual).

        Parameters (following the Direction interface):
            zk, Fzk, zbar, Fzbar, wk: Variables from the current iteration.
            zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1: Next iteration variables.
            iteration (int): Current iteration number.
        """
        self.z_prev = zk.copy()

        if self.fix_point_operator == "EG":
            self.g_prev = zk - wk
        else: 
            self.g_prev = zk - zbar
