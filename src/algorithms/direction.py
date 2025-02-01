# src/algorithms/direction.py

from abc import ABC, abstractmethod

class Direction(ABC):
    """
    Abstract Base Class for direction strategies used in FLEX.

    Subclasses must implement `compute_direction` and `post_step_update`.
    """

    @abstractmethod
    def compute_direction(self, zk, zbar, wk, Fzk, Fzbar, iteration):
        """
        Computes the direction `d^k` for the update step.

        Parameters:
            zk (np.ndarray): Current iterate.
            zbar (np.ndarray): Extrapolated point.
            wk (np.ndarray): Intermediate point.
            Fzk (np.ndarray): Operator evaluated at zk.
            Fzbar (np.ndarray): Operator evaluated at zbar.
            iteration (int): Current iteration number.

        Returns:
            tuple:
                - d (np.ndarray): Computed direction.
                - success (bool): Indicates if the direction was successfully computed.
        """
        pass

    @abstractmethod
    def post_step_update(self, zk, Fzk, zbar, Fzbar, wk, zk_plus_1, Fzk_plus_1,
                         zbar_plus_1, Fzbar_plus_1, wk_plus_1, iteration):
        """
        Performs any necessary updates after the main iteration steps.

        Parameters:
            zk, Fzk, zbar, Fzbar, wk: Variables from the current iteration.
            zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1: Variables for the next iteration.
            iteration (int): Current iteration number.
        """
        pass
