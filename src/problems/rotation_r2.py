# src/problems/rotation_r2.py

import numpy as np
from .base_problem import Problem

class RotationR2Problem(Problem):
    """
    Rotation in R^2 Monotone Operator.

    Defines an operator F: R^2 -> R^2 where F(z) = A z, and A is a rotation matrix:
        A = [[0, 1],
             [-1, 0]]

    This operator represents a rotation by 90 degrees and is monotone.
    """
    
    def __init__(self):
        """
        Initializes the RotationR2Problem.
        
        Sets dim_z=2, L=1 (Lipschitz constant).
        """
        dim_z = 2
        L = 1.0  # Lipschitz constant as given
        super().__init__(dim_z=dim_z, L=L)
        
        # Define the rotation matrix A
        self.A = np.array([[0, 1],
                           [-1, 0]])
        
    def F(self, z: np.ndarray) -> np.ndarray:
        """
        Evaluates F(z) = A z.
        
        Parameters:
            z (np.ndarray): Point in R^2.
        
        Returns:
            np.ndarray: F(z) = A z.
        """
        if z.shape[0] != self.dim_z:
            raise ValueError(f"Input z must be of dimension {self.dim_z}.")
        
        return self.A @ z
    
    def Jacobian_F(self, z: np.ndarray) -> np.ndarray:
        """
        Returns the Jacobian of F, which is constant and equal to A.
        
        Parameters:
            z (np.ndarray): Point in R^2 (ignored).
        
        Returns:
            np.ndarray: The matrix A.
        """
        return self.A
    
    def get_plot_settings(self) -> dict:
        """
        Returns the plotting settings specific to this problem.
        
        Returns:
            dict: A dictionary containing plot labels and titles.
        """
        return {
            'y_label': r'$\|F(z^k)\|$',
            'x_label_iterations': 'Iteration $k$',
            'x_label_F_evals': 'Number of $F$ evaluations',
            'x_label_operator_evals': 'Number of operator evaluations',  # Add this line
            'x_label_time': 'Time (s)',
            'title_prefix': 'Convergence on'
        }