# src/algorithms/agraal.py

import numpy as np
from .base_algorithm import IterativeAlgorithm
import time

class AGRAAL(IterativeAlgorithm):
    """
    Adaptive GRAAL Method for solving variational inequalities.

    Reference:
    "Beyond the Golden Ratio for Variational Inequality Algorithms"
    Ahmet Alacaoglu, Axel Böhm, Yura Malitsky
    https://arxiv.org/abs/2212.13955
    """

    def __init__(self, problem, z0, max_iter, performance_evaluator, stopping_criterion, tol, ignore_starting_point=True, callback=None):
        """
        Initializes the aGRAAL method.

        Parameters:
            problem (Problem): The problem instance.
            z0 (np.ndarray): Initial point.
            max_iter (int): Maximum number of iterations.
            performance_evaluator (callable): Function to evaluate performance.
            stopping_criterion (callable): Function to determine stopping condition.
            tol (float): Tolerance for convergence.
            callback (callable, optional): Optional callback function.
        """
        super().__init__(problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point, callback)
        self.tol = tol

    def run(self):
        """
        Executes the aGRAAL algorithm.

        Returns:
            np.ndarray: The final iterate after convergence or reaching the maximum number of iterations.
        """
        zk = self.z0.copy()
        z_bar = self.z0.copy()

        # Initialize parameters
        var_phi = (np.sqrt(5) + 1) / 2  # Golden ratio
        phi = (1 + var_phi) / 2
        gamma = 1 / phi + 1 / phi ** 2
        alpha = 0.1
        theta = phi

        self.start_time = time.time()

        # Compute F(zk)
        Fzk = self.problem.F(zk)
        self.total_F_evals = 1  # F evaluated

        if not self.ignore_starting_point:
            # Compute initial performance metric
            perf_metric = self.performance_evaluator(zk=zk, Fzk=Fzk)
            self.record_performance(iteration=0, perf_metric=perf_metric)

            # Check convergence at iteration 0
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    aGRAAL converged at iteration {0}")
                return zk

        # Iteration k = 1
        zk_next = self.problem.prox_g(zk - alpha *Fzk, alpha)
        self.total_prox_g_evals += 1  # Increment prox_g evaluations


        # Compute F(zk_next)
        Fzk_next = self.problem.F(zk_next)
        self.total_F_evals += 1  # F evaluated

        # Compute initial performance metric
        perf_metric = self.performance_evaluator(zk=zk_next, Fzk=Fzk_next)
        self.record_performance(iteration=1, perf_metric=perf_metric)

        # Check convergence at iteration 1
        if self.has_converged(perf_metric=perf_metric, tol=self.tol):
            print(f"    aGRAAL converged at iteration {1}")
            return zk_next

        # Prepare for next iteration
        z_prev = zk.copy()
        Fzk_prev = Fzk.copy()
        z_bar_prev = z_bar.copy()

        zk = zk_next
        Fzk = Fzk_next

        for k in range(2, self.max_iter + 1):

            alpha_prev = alpha
            theta_prev = theta

            # Compute residuals
            z_res = np.linalg.norm(zk - z_prev)
            F_res = np.linalg.norm(Fzk - Fzk_prev)

            # Compute alpha with safeguards
            if F_res != 0:
                alpha = min(
                    gamma * alpha_prev,
                    (phi * theta_prev / (4 * alpha_prev)) * (z_res ** 2 / F_res ** 2),
                    1e10
                )
            else:
                alpha = min(gamma * alpha, 1e10)

            # Update z_bar using the Golden Ratio
            z_bar = ((phi - 1) * zk + z_bar_prev) / phi

            # Compute the next iterate using the proximal operator
            zk_next = self.problem.prox_g(z_bar - alpha * Fzk, alpha)
            self.total_prox_g_evals += 1  # Increment prox_g evaluations

            # Compute F(zk_next)
            Fzk_next = self.problem.F(zk_next)
            self.total_F_evals += 1  # F evaluated once

            # Compute performance metric
            perf_metric = self.performance_evaluator(zk=zk_next, Fzk=Fzk_next)
            self.record_performance(iteration=k, perf_metric=perf_metric)

            # Check convergence
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    aGRAAL converged at iteration {k}")
                return zk_next

            # Update variables for the next iteration
            z_prev = zk.copy()
            Fzk_prev = Fzk.copy()
            z_bar_prev = z_bar.copy()
            zk = zk_next
            Fzk = Fzk_next

            theta = (alpha / alpha_prev) * phi

        return zk
