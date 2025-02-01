# src/algorithms/graal.py

import numpy as np
from .base_algorithm import IterativeAlgorithm
import time

class GRAAL(IterativeAlgorithm):
    """
    GRAAL (Golden Ratio Algorithm for Variational Inequalities) Method.

    References:
    - "Golden Ratio Algorithms for Variational Inequalities" by Yura Malitsky
      https://arxiv.org/abs/1803.08832
    - "Beyond the Golden Ratio for Variational Inequality Algorithms" by Ahmet Alacaoglu et al.
      https://arxiv.org/abs/2212.13955
    """

    def __init__(self, problem, z0, z_bar_m1, max_iter, performance_evaluator, stopping_criterion, tol, ignore_starting_point=True, callback=None):
        """
        Initializes the GRAAL method.

        Parameters:
            problem (Problem): The problem instance.
            z0 (np.ndarray): Initial point, must have dimension equal to problem.dim_z.
            z_bar_m1 (np.ndarray): Initial value for z_bar (can be set to z0).
            max_iter (int): Maximum number of iterations.
            performance_evaluator (callable): Function to evaluate performance metrics.
            stopping_criterion (callable): Function to determine convergence.
            tol (float): Tolerance for convergence.
            callback (callable, optional): Optional callback function.
        """
        super().__init__(problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point, callback)
        self.z_bar_m1 = z_bar_m1.copy()
        self.tol = tol

        # Using the updated phi and alpha as per the latest research (second paper, Corollary 2)
        self.phi = 2
        self.alpha = 0.999 / problem.L  # Step size

    def run(self):
        """
        Executes the GRAAL algorithm.

        Returns:
            np.ndarray: The final iterate after convergence or reaching the maximum number of iterations.
        """
        zk = self.z0.copy()
        z_bar = self.z_bar_m1.copy()

        self.start_time = time.time()

        # Compute F(zk)
        Fzk = self.problem.F(zk)
        self.total_F_evals = 1  # F evaluated once

        if not self.ignore_starting_point:
            # Compute initial performance metric
            perf_metric = self.performance_evaluator(zk=zk, Fzk=Fzk)  # **CHANGED**
            self.record_performance(iteration=0, perf_metric=perf_metric)

            # Check convergence at iteration 0
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    GRAAL converged at iteration {0}")
                return zk

        for k in range(1, self.max_iter + 1):
            # Update z_bar using the Golden Ratio
            z_bar = ((self.phi - 1) * zk + z_bar) / self.phi

            # Compute the next iterate using the proximal operator
            zk_next = self.problem.prox_g(z_bar - self.alpha * Fzk, self.alpha)
            self.total_prox_g_evals += 1  # Increment prox_g evaluations

            # Compute F(zk_next)
            Fzk_next = self.problem.F(zk_next)
            self.total_F_evals += 1  # F evaluated once

            # Compute performance metric
            perf_metric = self.performance_evaluator(zk=zk_next, Fzk=Fzk_next)  # **CHANGED**
            self.record_performance(iteration=k, perf_metric=perf_metric)

            # Check convergence
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    GRAAL converged at iteration {k}")
                zk = zk_next
                break

            # Update variables for the next iteration
            zk = zk_next
            Fzk = Fzk_next

        return zk
