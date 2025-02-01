# src/algorithms/eag_c.py

import numpy as np
from .base_algorithm import IterativeAlgorithm
import time

class EAGC(IterativeAlgorithm):
    """
    Extra Anchored Gradient with Constant Step-size (EAG-C) method.

    Reference:
    "Accelerated Algorithms for Smooth Convex-Concave Minimax Problems with O(1/k^2) Rate on Squared Gradient Norm"
    TaeHo Yoon, Ernest K. Ryu
    https://arxiv.org/abs/2102.07922
    """

    def __init__(self, problem, z0, max_iter, performance_evaluator, stopping_criterion, tol, ignore_starting_point=True, callback=None):
        """
        Initializes the EAG-C method.

        Parameters:
            problem (Problem): The problem instance.
            z0 (np.ndarray): Initial point, must have dimension equal to problem.dim_z.
            max_iter (int): Maximum number of iterations.
            performance_evaluator (callable): Function to evaluate performance metrics.
            stopping_criterion (callable): Function to determine convergence.
            tol (float): Tolerance for convergence.
            callback (callable, optional): Optional callback function.
        """
        super().__init__(problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point, callback)
        self.alpha = 1 / (8 * problem.L)  # Step size [Largest in Corollary 1.]
        self.z0_anchor = z0.copy()
        self.tol = tol

    def run(self):
        """
        Executes the EAG-C algorithm.

        Returns:
            np.ndarray: The final iterate after convergence or reaching the maximum number of iterations.
        """
        zk = self.z0.copy()

        self.start_time = time.time()

        # Compute F(zk)
        Fzk = self.problem.F(zk)
        self.total_F_evals = 1  # F evaluated once

        if not self.ignore_starting_point:
            # Compute initial performance metric
            perf_metric = self.performance_evaluator(zk=zk, Fzk=Fzk)
            self.record_performance(iteration=0, perf_metric=perf_metric)

            # Check convergence at iteration 0
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    EAG-C converged at iteration {0}")
                return zk

        for k in range(1, self.max_iter + 1):
            # Compute the next iterate using EAG-C update rule
            zk_bar = zk + (1 / (k + 2)) * (self.z0_anchor - zk) - self.alpha * Fzk

            # Compute F(zk_bar)
            Fzk_bar = self.problem.F(zk_bar)
            self.total_F_evals += 1  # F evaluated once

            # Compute the next iterate
            zk_next = zk + (1 / (k + 2)) * (self.z0_anchor - zk) - self.alpha * Fzk_bar

            # Compute F(zk_next)
            Fzk_next = self.problem.F(zk_next)
            self.total_F_evals += 1  # F evaluated once

            # Compute performance metric
            perf_metric = self.performance_evaluator(zk=zk_next, Fzk=Fzk_next)
            self.record_performance(iteration=k, perf_metric=perf_metric)

            # Check convergence
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    EAG-C converged at iteration {k}")
                zk = zk_next
                break

            # Update variables for next iteration
            zk = zk_next
            Fzk = Fzk_next

        return zk
