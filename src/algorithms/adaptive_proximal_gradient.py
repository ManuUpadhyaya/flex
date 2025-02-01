# src/algorithms/adaptive_proximal_gradient.py

import numpy as np
from .base_algorithm import IterativeAlgorithm
import time

class AdaptiveProximalGradient(IterativeAlgorithm):
    """
    Adaptive Proximal Gradient Method for solving:
        min_x f(x) + g(x)
    where f is smooth and convex, and g is convex with an easy proximal operator.

    Reference:
    The adaptive proximal gradient algorithm provided.
    """

    def __init__(self, problem, z0, max_iter, performance_evaluator, stopping_criterion,
                 tol=1e-8, sigma0=1.0, theta0=1/3, ignore_starting_point=True, callback=None):
        """
        Initializes the Adaptive Proximal Gradient Method.

        Parameters:
            problem: The problem instance.
            z0 (np.ndarray): Initial point.
            max_iter (int): Maximum number of iterations.
            performance_evaluator (callable): Function to evaluate performance.
            stopping_criterion (callable): Function to determine convergence.
            tol (float): Tolerance for convergence.
            sigma0 (float): Initial step size σ0 > 0.
            theta0 (float): Initial θ0 = 1/3.
            ignore_starting_point (bool): Whether to ignore the starting point in performance recording.
            callback (callable, optional): Optional callback function.
        """
        super().__init__(problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point, callback)
        self.tol = tol
        self.z0 = z0.copy()
        self.sigma0 = sigma0
        self.theta0 = theta0

    def run(self):
        """
        Executes the Adaptive Proximal Gradient Method.

        Returns:
            np.ndarray: The final iterate after convergence or reaching the maximum number of iterations.
        """
        z_km1 = self.z0.copy()  # z^{k-1}
        theta_km1 = self.theta0  # θ_{k-1}
        sigma_km1 = self.sigma0  # σ_{k-1}

        # Initial step
        self.start_time = time.time()

        # Compute F(z^{0})
        F_zkm1 = self.problem.F(z_km1)
        self.total_F_evals = 1

        # Compute z^{1}
        z_k = self.problem.prox_g(z_km1 - sigma_km1 * F_zkm1, sigma_km1)
        self.total_prox_g_evals = 1

        if not self.ignore_starting_point:
            # Compute performance metric
            F_zk = self.problem.F(z_k)
            self.total_F_evals += 1
            perf_metric = self.performance_evaluator(zk=z_k, Fzk=F_zk)
            self.record_performance(iteration=0, perf_metric=perf_metric)

            # Check convergence at iteration 0
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    Adaptive Proximal Gradient Method converged at iteration {0}")
                return z_k

        # Main loop
        for k in range(1, self.max_iter + 1):
            # Compute F(z^{k})
            F_zk = self.problem.F(z_k)
            self.total_F_evals += 1

            # Compute L_k
            delta_F = F_zk - F_zkm1
            delta_z = z_k - z_km1

            norm_delta_F = np.linalg.norm(delta_F)
            norm_delta_z = np.linalg.norm(delta_z)

            if norm_delta_z == 0:
                L_k = np.inf
            else:
                L_k = norm_delta_F / norm_delta_z

            # Compute σ_k
            term1 = np.sqrt(2/3 + theta_km1) * sigma_km1
            denom_squared = 2 * sigma_km1 ** 2 * L_k ** 2 - 1
            denom = np.sqrt(max(denom_squared, 0))

            if denom == 0:
                term2 = np.inf  # Division by zero leads to infinity
            else:
                term2 = sigma_km1 / denom

            sigma_k = min(term1, term2)

            # Update z^{k+1}
            z_kp1 = self.problem.prox_g(z_k - sigma_k * F_zk, sigma_k)
            self.total_prox_g_evals += 1

            # Update θ_k
            theta_k = sigma_k / sigma_km1

            # Compute performance metric
            F_zkp1 = self.problem.F(z_kp1)
            self.total_F_evals += 1
            perf_metric = self.performance_evaluator(zk=z_kp1, Fzk=F_zkp1)
            self.record_performance(iteration=k, perf_metric=perf_metric)

            # Check convergence
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    Adaptive Proximal Gradient Method converged at iteration {k}")
                break

            # Prepare for next iteration
            z_km1 = z_k.copy()
            z_k = z_kp1.copy()
            sigma_km1 = sigma_k
            theta_km1 = theta_k
            F_zkm1 = F_zk.copy()  # Store F(z_k) for next iteration

        return z_kp1
