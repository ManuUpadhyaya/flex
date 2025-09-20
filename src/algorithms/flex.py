import numpy as np
import time

from .base_algorithm import IterativeAlgorithm
from .direction import Direction

class FLEX(IterativeAlgorithm):
    """
    Class for the FLEX methods.
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
        M,
        beta,
        rho,
        tol,
        direction: Direction,
        ignore_starting_point=True,
        callback=None
    ):
        """
        Initializes the FLEX method.

        Parameters:
            problem: The problem instance.
            z0 (np.ndarray): Initial point.
            max_iter (int): Maximum number of iterations.
            performance_evaluator (callable): Function to evaluate performance.
            stopping_criterion (callable): Function to determine stopping condition.
            gamma (float): Step size parameter, must be in (0, 1/L_F).
            sigma (float): Parameter in (0,1) for line search condition.
            M (int): Maximum number of backtracking steps in line search.
            beta (float): Parameter in (0,1) for line search step size reduction.
            rho (float): Parameter in (0,1) for checking contraction.
            tol (float): Tolerance for convergence.
            direction (Direction): An instance of a Direction subclass.
            ignore_starting_point (bool, optional): If True, skip performance metric at iteration 0.
            callback (callable, optional): Optional callback function.
        """
        super().__init__(problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point, callback)
        self.gamma = gamma
        self.sigma = sigma
        self.M = M
        self.beta = beta
        self.rho = rho
        self.tol = tol

        self.zk = z0.copy()

        self.direction = direction
        self._tau_history = []

    def run(self):
        """
        Executes the FLEX meta-algorithm using the provided direction strategy.
        """
        zk = self.zk.copy()
        self.start_time = time.time()

        Fzk = self.problem.F(zk)
        self.total_F_evals += 1

        if not self.ignore_starting_point:
            perf_metric = self.performance_evaluator(zk=zk, Fzk=Fzk)
            self.record_performance(iteration=0, perf_metric=perf_metric)
            self._tau_history.append(None)

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
                tau_k = 0.0
                zk_plus_1 = wk
                Fzk_plus_1 = self.problem.F(zk_plus_1)
                self.total_F_evals += 1

                perf_metric = self.performance_evaluator(zk=zk_plus_1, Fzk=Fzk_plus_1)
                self.record_performance(iteration=k, perf_metric=perf_metric)
                self._tau_history.append(tau_k)

                if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                    print(f"    {self.__class__.__name__} converged at iteration {k}")
                    break

                zbar_plus_1 = zk_plus_1 - self.gamma * Fzk_plus_1
                Fzbar_plus_1 = self.problem.F(zbar_plus_1)
                self.total_F_evals += 1

                wk_plus_1 = zk_plus_1 - self.gamma * Fzbar_plus_1

                self.direction.post_step_update(
                    zk, Fzk, zbar, Fzbar, wk,
                    zk_plus_1, Fzk_plus_1, zbar_plus_1,
                    Fzbar_plus_1, wk_plus_1, k
                )

                zk, Fzk, zbar, Fzbar, wk = zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1
                continue

            zk_dk = zk + dk
            Fzk_dk = self.problem.F(zk_dk)
            self.total_F_evals += 1

            # Test linear contraction
            if np.linalg.norm(Fzk_dk) <= self.rho * np.linalg.norm(Fzk):
                # Accept the direction directly
                tau_k = 1.0
                zk_plus_1 = zk_dk
                Fzk_plus_1 = self.problem.F(zk_plus_1)
                self.total_F_evals += 1

            else:
                # Perform line search
                rhs = (
                    np.linalg.norm(Fzk) ** 2
                    - self.sigma * (1 - self.gamma ** 2 * self.problem.L ** 2)
                    * np.linalg.norm(Fzk - Fzbar) ** 2
                )
                m = 1
                while m <= self.M:
                    tau_k = self.beta ** m
                    zk_plus_1 = (1 - tau_k) * wk + tau_k * zk_dk
                    Fzk_plus_1 = self.problem.F(zk_plus_1)
                    self.total_F_evals += 1

                    lhs = np.linalg.norm(Fzk_plus_1) ** 2

                    if lhs <= rhs:
                        break
                    else:
                        m += 1

                # If line search fails (m > M), fallback to wk
                if m > self.M:
                    tau_k = 0
                    zk_plus_1 = wk
                    Fzk_plus_1 = self.problem.F(zk_plus_1)
                    self.total_F_evals += 1

            perf_metric = self.performance_evaluator(zk=zk_plus_1, Fzk=Fzk_plus_1)
            self.record_performance(iteration=k, perf_metric=perf_metric)
            self._tau_history.append(tau_k)

            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    {self.__class__.__name__} converged at iteration {k}")
                break

            zbar_plus_1 = zk_plus_1 - self.gamma * Fzk_plus_1
            Fzbar_plus_1 = self.problem.F(zbar_plus_1)
            self.total_F_evals += 1

            wk_plus_1 = zk_plus_1 - self.gamma * Fzbar_plus_1

            self.direction.post_step_update(
                zk, Fzk, zbar, Fzbar, wk,
                zk_plus_1, Fzk_plus_1, zbar_plus_1,
                Fzbar_plus_1, wk_plus_1, k
            )

            zk, Fzk, zbar, Fzbar, wk = zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1

        return zk

    def get_results(self):
        results = super().get_results()
        results['tau'] = self._tau_history
        return results
