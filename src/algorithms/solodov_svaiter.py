# src/algorithms/solodov_svaiter.py

import numpy as np
from .base_algorithm import IterativeAlgorithm
import time
import warnings
from numpy import linalg

class SolodovSvaiter(IterativeAlgorithm):
    """
    Algorithm 2.1

    Reference:
    "A Globally Convergent Inexact Newton Method for Systems of Monotone Equations"
    Michael V. Solodov and Benar F. Svaiter
    https://link.springer.com/chapter/10.1007/978-1-4757-6388-1_18
    """

    def __init__(self, problem, z0, max_iter, performance_evaluator, stopping_criterion, tol, beta=0.5, lambda_param=0.5, ignore_starting_point=True, callback=None):
        """
        Algorithm 2.1 by Solodov and Svaiter

        Parameters:
            problem (Problem): The problem instance.
            z0 (np.ndarray): Initial point, must have dimension equal to problem.dim_z.
            max_iter (int): Maximum number of iterations.
            performance_evaluator (callable): Function to evaluate performance.
            stopping_criterion (callable): Function to determine stopping condition.
            tol (float): Tolerance for convergence.
            beta (float, optional): Step-size scaling parameter for line search, default is 0.5.
            lambda_param (float, optional): Scaling factor in line-search criterion, default is 0.5.
            callback (callable, optional): Optional callback function.
        """
        super().__init__(problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point, callback)
        self.beta = beta
        self.lambda_param = lambda_param
        self.tol = tol

    def run(self):
        """
        Executes the Solodov-Svaiter algorithm.

        Returns:
            np.ndarray: The final iterate after convergence or reaching the maximum number of iterations.
        """
        z = self.z0.copy()
        dim_z = z.size  # Updated from 'n' to 'dim_z'

        self.start_time = time.time()

        # Compute F(z0)
        Fz = self.problem.F(z)
        self.total_F_evals = 1  # F evaluated once

        if not self.ignore_starting_point:
            # Compute initial performance metric
            perf_metric = self.performance_evaluator(zk=z, Fzk=Fz) 
            self.record_performance(iteration=0, perf_metric=perf_metric)

            # Check convergence at iteration 0
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    Solodov-Svaiter converged at iteration {0}")
                return z

        for k in range(1, self.max_iter + 1):
            norm_Fz = np.linalg.norm(Fz)
            mu = max(norm_Fz, np.sqrt(norm_Fz))
            rho = min(0.5, np.sqrt(norm_Fz))

            G = self.problem.Jacobian_F(z)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress potential warnings during matrix solve
                # Solve (G + mu I) d = -F(z)
                M = G + mu * np.eye(dim_z)
                try:
                    d = linalg.solve(M, -Fz)
                except np.linalg.LinAlgError as e:
                    print(f"    Linear algebra error at iteration {k}: {e}")
                    break  # Exit the loop if the system is singular

            norm_d = np.linalg.norm(d)

            # Line search to satisfy the condition
            m = 0
            while True:
                step_size = self.beta ** m
                z_bar = z + step_size * d
                Fz_bar = self.problem.F(z_bar)
                self.total_F_evals += 1  # F evaluated once

                condition = -np.dot(Fz_bar.T, d) >= self.lambda_param * (1 - rho) * mu * norm_d ** 2
                if condition:
                    break
                else:
                    m += 1
                    if m > 50:  # Prevent infinite loop
                        print(f"    Line search failed to satisfy condition at iteration {k}.")
                        break

            # Update z using the computed direction
            numerator = np.dot(Fz_bar.T, z - z_bar)
            denominator = np.linalg.norm(Fz_bar) ** 2
            if denominator == 0:
                print(f"    Denominator is zero at iteration {k}.")
                break  # Avoid division by zero

            z_next = z - (numerator / denominator) * Fz_bar
            Fz_next = self.problem.F(z_next)
            self.total_F_evals += 1  # F evaluated once

            # Compute performance metric
            perf_metric = self.performance_evaluator(zk=z_next, Fzk=Fz_next)  # **CHANGED**
            self.record_performance(iteration=k, perf_metric=perf_metric)

            # Check convergence
            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    Solodov-Svaiter converged at iteration {k}")
                z = z_next
                break

            # Update variables for next iteration
            z = z_next
            Fz = Fz_next

        return z
