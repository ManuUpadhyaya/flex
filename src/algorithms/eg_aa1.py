# src/algorithms/anderson_accelerated_eg.py

import numpy as np
from .base_algorithm import IterativeAlgorithm
import time

class EGAA(IterativeAlgorithm):
    """
    EG-Anderson(1)

    Reference:
    "An extra gradient Anderson-accelerated algorithm for pseudomonotone variational inequalities"
    Xin Qu, Wei Bian, Xiaojun Chen
    """

    def __init__(self, problem, z0, max_iter, performance_evaluator, stopping_criterion, tol,
                 omega=30, M=5000, tau=0.6, rho=0.8, mu=0.5, t=1, ignore_starting_point=True, callback=None):
        """
        Initializes EG-Anderson(1).

        Parameters:
            problem: The problem instance.
            z0: Initial point.
            max_iter: Maximum number of iterations.
            performance_evaluator: Function to evaluate performance.
            stopping_criterion: Function to determine stopping condition.
            tol: Tolerance for convergence.
            omega: Non-negative algorithm parameter.
            M: Positive large algorithm parameter.
            tau: Algorithm parameter greater than 1/2.
            rho: Algorithm parameter in the interval (0,1).
            mu: Algorithm parameter in the interval (0,1).
            t: Step size parameter greater than 0.
            callback: Optional callback function.
        """
        super().__init__(problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point, callback)
        self.t = t  # Step size parameter, can be set constant if L is known, which we assume
        self.sigma = 1
        self.omega = omega
        self.M = M
        self.tau = tau
        self.rho = rho
        self.mu = mu
        self.tol = tol

    def run(self):
        zk = self.z0.copy()
        self.start_time = time.time()

        # Compute F(zk)
        Fzk = self.problem.F(zk)
        self.total_F_evals = 1  # F evaluated once

        if not self.ignore_starting_point:
            # Compute initial performance metric
            perf_metric = self.performance_evaluator(zk=zk, Fzk=Fzk)
            self.record_performance(iteration=0, perf_metric=perf_metric)

            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    EG-AA converged at iteration {0}")
                return zk

        for k in range(1, self.max_iter + 1):
            # Extrapolation step
            y_half = self.problem.prox_g(zk - self.t * Fzk, self.t)
            self.total_prox_g_evals += 1  # Increment prox_g evaluations

            Fy_half = self.problem.F(y_half)
            self.total_F_evals += 1  # F evaluated once

            # Compute full EG step
            y_one = self.problem.prox_g(zk - self.t * Fy_half, self.t)
            self.total_prox_g_evals += 1  # Increment prox_g evaluations

            # Compute differences
            F_t_z = y_half - zk
            F_tilde_t_z = y_one - zk

            norm_F_tilde_t_z = np.linalg.norm(F_tilde_t_z)
            norm_F_t_z = np.linalg.norm(F_t_z)

            # Compute alpha
            if norm_F_tilde_t_z < min(norm_F_t_z, self.omega * self.sigma ** (-self.tau)):
                numerator = np.dot(F_tilde_t_z, F_tilde_t_z - F_t_z)
                denominator = np.linalg.norm(F_tilde_t_z - F_t_z) ** 2
                alpha = numerator / denominator
            else:
                alpha = self.M + 1

            # Update zk
            if np.abs(alpha) <= self.M:
                zk_next = alpha * zk + (1 - alpha) * y_one
                self.sigma += 1
            else:
                zk_next = y_one

            Fzk_next = self.problem.F(zk_next)
            self.total_F_evals += 1  # F evaluated once

            # Compute performance metric
            perf_metric = self.performance_evaluator(zk=zk_next, Fzk=Fzk_next)
            self.record_performance(iteration=k, perf_metric=perf_metric)

            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    EG-AA converged at iteration {k}")
                zk = zk_next
                break

            # Update variables for next iteration
            zk = zk_next
            Fzk = Fzk_next

        return zk
