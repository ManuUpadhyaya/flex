# src/problems/base_problem.py

from abc import ABC, abstractmethod
import numpy as np

class Problem(ABC):
    """
    Base class for defining monotone inclusion problems.

    Attributes:
        dim_z (int): Dimension of the space R^dim_z.
        L (float): Lipschitz constant of the operator F.
    """

    def __init__(self, dim_z: int, L: float):
        """
        Initializes the Problem with a given dimension and Lipschitz constant.

        Parameters:
            dim_z (int): Dimension of the space R^dim_z.
            L (float): Lipschitz constant of the operator F.
        """
        self.dim_z = dim_z  # Dimension of the space R^dim_z
        self.L = L  # Lipschitz constant of the operator F

    @abstractmethod
    def F(self, z: np.ndarray) -> np.ndarray:
        """
        Evaluates the operator F at a given point z.

        Parameters:
            z (np.ndarray): The point at which to evaluate F.

        Returns:
            np.ndarray: The value of F at z.

        Raises:
            ValueError: If the dimension of z does not match dim_z.
        """
        if z.shape[0] != self.dim_z:
            raise ValueError(f"Input z must be of dimension {self.dim_z}.")
        pass  # Must be implemented by subclasses

    def g(self, z: np.ndarray) -> float:
        """
        Evaluates the function g at a given point z.

        Default implementation assumes g(z) = 0 for all z.

        Parameters:
            z (np.ndarray): The point at which to evaluate g.

        Returns:
            float: The value of g at z.
        """
        if z.shape[0] != self.dim_z:
            raise ValueError(f"Input z must be of dimension {self.dim_z}.")
        return 0.0  # Default g(z) = 0

    def prox_g(self, z: np.ndarray, gamma: float) -> np.ndarray:
        """
        Computes the proximal operator of gamma * g at point z.

        Default implementation returns z, assuming g is the zero function.

        Parameters:
            z (np.ndarray): The point at which to compute the proximal operator.
            gamma (float): Step size parameter.

        Returns:
            np.ndarray: The result of the proximal operator at z.
        """
        if z.shape[0] != self.dim_z:
            raise ValueError(f"Input z must be of dimension {self.dim_z}.")
        return z  # Default prox_g is the identity mapping
