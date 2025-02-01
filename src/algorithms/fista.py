# src/algorithms/fista.py

import numpy as np
from .base_algorithm import IterativeAlgorithm
import time

class FISTA(IterativeAlgorithm):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for solving:
        min_x f(x) + g(x)
    where f is smooth and convex, and g is convex with an easy proximal operator.

    Reference:
    "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems"
    Amir Beck and Marc Teboulle
    SIAM Journal on Imaging Sciences, 2009
    """

    def __init__(self, problem, z0, max_iter, performance_evaluator, stopping_criterion, L_f, tol=1e-8, ignore_starting_point=True, callback=None):
        """
        Initializes the FISTA algorithm.

        Parameters:
            problem: The problem instance.
            z0 (np.ndarray): Initial point.
            max_iter: Maximum number of iterations.
            performance_evaluator: Function to evaluate performance.
            stopping_criterion: Function to determine stopping condition.
            L_f (float): Lipschitz constant of the gradient of f.
            tol (float): Tolerance for convergence.
            callback: Optional callback function.
        """
        super().__init__(problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point, callback)
        self.L_f = L_f
        self.tol = tol
        self.z0 = z0.copy()

    def run(self):
        """
        Executes the FISTA algorithm.

        Returns:
            np.ndarray: The final iterate after convergence or reaching the maximum number of iterations.
        """
        z_k = self.z0.copy()
        y_k = z_k.copy()
        t_k = 1

        self.start_time = time.time()

        # Compute initial gradient
        F_yk = self.problem.F(y_k)
        self.total_F_evals = 1  # F evaluated once

        if not self.ignore_starting_point:
            # Compute performance metric
            perf_metric = self.performance_evaluator(zk=z_k, Fzk=F_yk)
            self.record_performance(iteration=0, perf_metric=perf_metric)

            # Check convergence at iteration 0
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    FISTA converged at iteration {0}")
                return z_k

        for k in range(1, self.max_iter + 1):
            # Gradient step
            z_next = self.problem.prox_g(y_k - (1 / self.L_f) * F_yk, 1 / self.L_f)
            self.total_prox_g_evals += 1

            # Update t_{k+1}
            t_next = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2

            # Update y_{k+1}
            y_k = z_next + ((t_k - 1) / t_next) * (z_next - z_k)

            # Update variables
            z_k = z_next
            t_k = t_next

            # Compute gradient at new y_k
            F_yk = self.problem.F(y_k)
            self.total_F_evals += 1

            # Compute performance metric
            F_zk = self.problem.F(z_k)
            self.total_F_evals += 1
            perf_metric = self.performance_evaluator(zk=z_k, Fzk=F_zk)
            self.record_performance(iteration=k, perf_metric=perf_metric)

            # Check convergence
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    FISTA converged at iteration {k}")
                break

        return z_k
