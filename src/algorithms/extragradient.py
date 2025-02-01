# src/algorithms/extragradient.py

import numpy as np
import time
from .base_algorithm import IterativeAlgorithm

class Extragradient(IterativeAlgorithm):
    def __init__(self, problem, z0, max_iter, performance_evaluator, stopping_criterion, gamma, tol, ignore_starting_point=True, callback=None):
        super().__init__(problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point, callback)
        self.gamma = gamma
        self.tol = tol 

    def run(self):
        """
        Executes the Extragradient algorithm.

        Returns:
            np.ndarray: The final iterate after convergence or reaching the maximum number of iterations.
        """
        zk = self.z0.copy()
        self.start_time = time.time()

        Fzk = self.problem.F(zk)
        self.total_F_evals = 1 

        if not self.ignore_starting_point:
            perf_metric = self.performance_evaluator(zk=zk, Fzk=Fzk, k=0)

            self.record_performance(
                iteration=0,
                perf_metric=perf_metric
            )

            if self.has_converged(
                perf_metric=perf_metric,
                zk=zk,
                Fzk=Fzk,
                k=0,
                tol=self.tol
            ):
                print(f"    EG converged at iteration 0")
                return zk

        for k in range(1, self.max_iter + 1):
            zk_bar = self.problem.prox_g(zk - self.gamma * Fzk, self.gamma)
            self.total_prox_g_evals += 1 

            Fzk_bar = self.problem.F(zk_bar)
            self.total_F_evals += 1  

            zk_prev = zk.copy()
            zk = self.problem.prox_g(zk - self.gamma * Fzk_bar, self.gamma)
            self.total_prox_g_evals += 1

            Fzk = self.problem.F(zk)
            self.total_F_evals += 1

            perf_metric = self.performance_evaluator(
                zk=zk,
                Fzk=Fzk,
                zk_bar=zk_bar,
                Fzk_bar=Fzk_bar,
                zk_prev=zk_prev,
                k=k
            )

            self.record_performance(
                iteration=k,
                perf_metric=perf_metric
            )

            if self.has_converged(
                perf_metric=perf_metric,
                zk=zk,
                Fzk=Fzk,
                zk_bar=zk_bar,
                Fzk_bar=Fzk_bar,
                zk_prev=zk_prev,
                k=k,
                tol=self.tol
            ):
                print(f"    EG converged at iteration {k}")
                break

        return zk
