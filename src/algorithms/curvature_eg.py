# src/algorithms/curvature_eg.py

import numpy as np
from .base_algorithm import IterativeAlgorithm
import time

class CurvatureEG(IterativeAlgorithm):
    """
    Curvature-Aware Extra Gradient Method.

    Reference:
    "Escaping limit cycles: Global convergence for constrained nonconvex-nonconcave minimax problems"
    Thomas Pethick, Puya Latafat, Panos Patrinos, Olivier Fercoq, Volkan Cevher
    https://arxiv.org/abs/2302.09831
    """

    def __init__(self, problem, z0, max_iter, performance_evaluator, stopping_criterion, tol, nu, tau, ignore_starting_point=True, callback=None):
        """
        Initializes the Curvature-Aware Extra Gradient method.

        Parameters:
            problem: The problem instance.
            z0: Initial point.
            max_iter: Maximum number of iterations.
            performance_evaluator: Function to evaluate performance.
            stopping_criterion: Function to determine stopping condition.
            tol: Tolerance for convergence.
            nu: Parameter nu in (0,1).
            tau: Parameter tau in (0,1).
            callback: Optional callback function.
        """
        super().__init__(problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point, callback)
        self.nu = nu
        self.tau = tau
        self.tol = tol

    def run(self):
        zk = self.z0.copy()
        self.start_time = time.time()

        # Compute F(z0)
        Fzk = self.problem.F(zk)
        self.total_F_evals = 1  # F evaluated once

        if not self.ignore_starting_point:
            # Compute initial performance metric
            perf_metric = self.performance_evaluator(zk=zk, Fzk=Fzk)
            self.record_performance(iteration=0, perf_metric=perf_metric)

            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    CurvatureEG converged at iteration {0}")
                return zk

        for k in range(1, self.max_iter + 1):
            # Compute gamma using the norm of the Jacobian
            JFzk_norm = np.linalg.norm(self.problem.Jacobian_F(zk), 2)
            gamma = self.nu / JFzk_norm

            # Extrapolation step
            z_bar = self.problem.prox_g(zk - gamma * Fzk, gamma)
            self.total_prox_g_evals += 1  # Increment prox_g evaluations
            
            Fz_bar = self.problem.F(z_bar)
            self.total_F_evals += 1  # F evaluated once

            # Backtracking loop
            while gamma * np.linalg.norm(Fz_bar - Fzk) > self.nu * np.linalg.norm(z_bar - zk):
                gamma *= self.tau
                z_bar = self.problem.prox_g(zk - gamma * Fzk, gamma)  # Corrected
                self.total_prox_g_evals += 1  # Increment prox_g evaluations
                Fz_bar = self.problem.F(z_bar)
                self.total_F_evals += 1  # F evaluated once

            # Compute alpha | We pick lambda_k = 1 and delta_k = 0
            Hz_bar = z_bar - gamma * Fz_bar
            Hzk = zk - gamma * Fzk
            numerator = np.dot(z_bar - zk, Hz_bar - Hzk)
            denominator = np.linalg.norm(Hz_bar - Hzk) ** 2
            alpha = numerator / denominator

            # Update zk
            zk_next = zk + alpha * (Hz_bar - Hzk)
            Fzk_next = self.problem.F(zk_next)
            self.total_F_evals += 1  # F evaluated once

            # Compute performance metric
            perf_metric = self.performance_evaluator(zk=zk_next, Fzk=Fzk_next)
            self.record_performance(iteration=k, perf_metric=perf_metric)

            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    CurvatureEG converged at iteration {k}")
                zk = zk_next
                break

            # Update variables for next iteration
            zk = zk_next
            Fzk = Fzk_next

        return zk
