# src/algorithms/iflex.py

import numpy as np
import time
from .base_algorithm import IterativeAlgorithm
from .direction import Direction

class IFLEX(IterativeAlgorithm):
    """
    Class for the I-FLEX methods.
    """

    def __init__(
        self,
        problem,
        z0,
        max_iter,
        performance_evaluator,
        stopping_criterion,
        gamma,
        sigma,
        beta,
        tol,
        direction: Direction,
        ignore_starting_point=True,
        callback=None
    ):
        """
        Initializes the I-FLEX method.

        Parameters:
            problem: The problem instance.
            z0 (np.ndarray): Initial point.
            max_iter (int): Maximum number of iterations.
            performance_evaluator (callable): Function to evaluate performance.
            stopping_criterion (callable): Function to determine stopping condition.
            gamma (float): Step size parameter, must be in (0, 1/L_F).
            sigma (float): Parameter in (0,1) for line search condition.
            beta (float): Parameter in (0,1) for line search step size reduction.
            tol (float): Tolerance for convergence.
            direction (Direction): An instance of a Direction subclass.
            ignore_starting_point (bool, optional): Skip performance metric at iteration 0 if True.
            callback (callable, optional): Optional callback function.
        """
        super().__init__(problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point, callback)
        self.gamma = gamma
        self.sigma = sigma
        self.beta = beta
        self.tol = tol
        self.zk = z0.copy()

        self.direction = direction

    def run(self):
        """
        Executes I-FLEX using the provided Direction strategy.
        """
        zk = self.zk.copy()
        self.start_time = time.time()

        Fzk = self.problem.F(zk)
        self.total_F_evals += 1

        if not self.ignore_starting_point:
            perf_metric = self.performance_evaluator(zk=zk, Fzk=Fzk)
            self.record_performance(iteration=0, perf_metric=perf_metric)

            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    {self.__class__.__name__} converged at iteration 0")
                return zk

        zbar = zk - self.gamma * Fzk
        Fzbar = self.problem.F(zbar)
        self.total_F_evals += 1

        wk = zk - self.gamma * Fzbar

        for k in range(1, self.max_iter + 1):

            dk, success = self.direction.compute_direction(zk, zbar, wk, Fzk, Fzbar, k)

            if not success:
                zk_plus_1 = wk
                Fzk_plus_1 = self.problem.F(zk_plus_1)
                self.total_F_evals += 1

                perf_metric = self.performance_evaluator(zk=zk_plus_1, Fzk=Fzk_plus_1)
                self.record_performance(iteration=k, perf_metric=perf_metric)

                if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                    print(f"    {self.__class__.__name__} converged at iteration {k}")
                    break

                zbar_plus_1 = zk_plus_1 - self.gamma * Fzk_plus_1
                Fzbar_plus_1 = self.problem.F(zbar_plus_1)
                self.total_F_evals += 1

                wk_plus_1 = zk_plus_1 - self.gamma * Fzbar_plus_1

                self.direction.post_step_update(
                    zk, Fzk, zbar, Fzbar, wk,
                    zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1, k
                )

                zk, Fzk, zbar, Fzbar, wk = zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1
                continue

            zk_dk = zk + dk

            rhs = (
                np.linalg.norm(Fzk) ** 2
                - self.sigma * (1 - self.gamma ** 2 * self.problem.L ** 2)
                * np.linalg.norm(Fzk - Fzbar) ** 2
            )

            m = 0
            while True:
                tau_m = self.beta ** m
                zk_plus_1 = (1 - tau_m) * wk + tau_m * zk_dk
                Fzk_plus_1 = self.problem.F(zk_plus_1)
                self.total_F_evals += 1

                lhs = np.linalg.norm(Fzk_plus_1) ** 2

                if lhs <= rhs:
                    break
                else:
                    m += 1

            perf_metric = self.performance_evaluator(zk=zk_plus_1, Fzk=Fzk_plus_1)
            self.record_performance(iteration=k, perf_metric=perf_metric)

            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    {self.__class__.__name__} converged at iteration {k}")
                break

            zbar_plus_1 = zk_plus_1 - self.gamma * Fzk_plus_1
            Fzbar_plus_1 = self.problem.F(zbar_plus_1)
            self.total_F_evals += 1

            wk_plus_1 = zk_plus_1 - self.gamma * Fzbar_plus_1

            self.direction.post_step_update(
                zk, Fzk, zbar, Fzbar, wk,
                zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1, k
            )

            zk, Fzk, zbar, Fzbar, wk = zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1

        return zk
