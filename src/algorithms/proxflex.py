import numpy as np
import time
from .base_algorithm import IterativeAlgorithm
from .direction import Direction

class ProxFLEX(IterativeAlgorithm):
    """
    Class for the ProxFLEX methods.
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
        Initializes the ProxFLEX method.

        Parameters:
            problem: The problem instance (must provide .F(...) and .prox_g(...), etc.).
            z0 (np.ndarray): Initial point.
            max_iter (int): Maximum number of iterations.
            performance_evaluator (callable): Function to evaluate performance.
            stopping_criterion (callable): Function to determine stopping condition.
            gamma (float): Step size parameter, must be in (0, 1/L_F).
            sigma (float): Parameter in (0,1) for line search condition.
            M (int): Maximum number of backtracking steps in line search.
            beta (float): Parameter in (0,1) for line search step size reduction.
            rho (float): Parameter in (0,1) for checking contraction (V_{k+1} <= rho^2 * V_k).
            tol (float): Tolerance for convergence.
            direction (Direction): Strategy object that implements compute_direction and post_step_update.
            ignore_starting_point (bool, optional): If True, skip performance metric at iteration 0.
            callback (callable, optional): Optional callback function.
        """
        super().__init__(
            problem, z0, max_iter, performance_evaluator,
            stopping_criterion, ignore_starting_point, callback
        )
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
        Executes ProxFLEX meta-algorithm using the provided Direction strategy.
        """
        zk = self.zk.copy()
        gamma = self.gamma
        sigma = self.sigma
        M = self.M
        beta = self.beta
        L_F = self.problem.L
        rho = self.rho

        self.start_time = time.time()
        self.total_F_evals = 0
        self.total_prox_g_evals = 0

        Fzk = self.problem.F(zk)
        self.total_F_evals += 1

        if not self.ignore_starting_point:
            perf_metric = self.performance_evaluator(zk=zk, Fzk=Fzk)
            self.record_performance(iteration=0, perf_metric=perf_metric)
            self._tau_history.append(None)

            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    {self.__class__.__name__} converged at iteration 0")
                return zk

        zbar = self.problem.prox_g(zk - gamma * Fzk, gamma)
        self.total_prox_g_evals += 1

        Fzbar = self.problem.F(zbar)
        self.total_F_evals += 1

        wk = self.problem.prox_g(zk - gamma * Fzbar, gamma)
        self.total_prox_g_evals += 1
        
        V_k = self.compute_V(zk, zbar, wk, Fzk, Fzbar)

        for k in range(1, self.max_iter + 1):
            dk, success = self.direction.compute_direction(zk, zbar, wk, Fzk, Fzbar, k)

            if not success:
                tau_k = 0.0
                zk_plus_1 = wk
                Fzk_plus_1 = self.problem.F(zk_plus_1)
                self.total_F_evals += 1

                zbar_plus_1 = self.problem.prox_g(zk_plus_1 - gamma * Fzk_plus_1, gamma)
                self.total_prox_g_evals += 1

                Fzbar_plus_1 = self.problem.F(zbar_plus_1)
                self.total_F_evals += 1

                wk_plus_1 = self.problem.prox_g(zk_plus_1 - gamma * Fzbar_plus_1, gamma)
                self.total_prox_g_evals += 1

                perf_metric = self.performance_evaluator(zk=zk_plus_1, Fzk=Fzk_plus_1)
                self.record_performance(iteration=k, perf_metric=perf_metric)
                self._tau_history.append(tau_k)

                if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                    print(f"    {self.__class__.__name__} converged at iteration {k}")
                    break

                self.direction.post_step_update(
                    zk, Fzk, zbar, Fzbar, wk,
                    zk_plus_1, Fzk_plus_1, zbar_plus_1,
                    Fzbar_plus_1, wk_plus_1, k
                )

                zk, Fzk, zbar, Fzbar, wk = (
                    zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1
                )
                V_k = self.compute_V(zk, zbar, wk, Fzk, Fzbar)

                continue

            zk_dk = zk + dk

            Fzk_dk = self.problem.F(zk_dk)
            self.total_F_evals += 1

            zbar_from_zk_dk = self.problem.prox_g(zk_dk - gamma * Fzk_dk, gamma)
            self.total_prox_g_evals += 1

            Fzbar_from_zk_dk = self.problem.F(zbar_from_zk_dk)
            self.total_F_evals += 1

            w_from_zk_dk = self.problem.prox_g(zk_dk - gamma * Fzbar_from_zk_dk, gamma)
            self.total_prox_g_evals += 1

            V_from_zk_dk = self.compute_V(
                zk_dk, zbar_from_zk_dk, w_from_zk_dk, Fzk_dk, Fzbar_from_zk_dk
            )

            # Test linear contraction: V_{k+1} <= (rho^2) * V_k
            if V_from_zk_dk <= rho**2 * V_k:
                tau_k = 1.0
                zk_plus_1 = zk_dk
                Fzk_plus_1 = Fzk_dk
                zbar_plus_1 = zbar_from_zk_dk
                Fzbar_plus_1 = Fzbar_from_zk_dk
                wk_plus_1 = w_from_zk_dk
                V_k_plus_1 = V_from_zk_dk
            else:
                # Perform line search
                sigma_term = (
                    sigma * (1 - gamma**2 * L_F**2)
                    * gamma**(-2)
                    * np.linalg.norm(wk - zbar) ** 2
                )
                m = 1
                while m <= M:
                    tau_k = beta**m
                    zk_plus_1 = (1 - tau_k) * wk + tau_k * zk_dk

                    Fzk_plus_1 = self.problem.F(zk_plus_1)
                    self.total_F_evals += 1

                    zbar_plus_1 = self.problem.prox_g(zk_plus_1 - gamma * Fzk_plus_1, gamma)
                    self.total_prox_g_evals += 1

                    Fzbar_plus_1 = self.problem.F(zbar_plus_1)
                    self.total_F_evals += 1

                    wk_plus_1 = self.problem.prox_g(zk_plus_1 - gamma * Fzbar_plus_1, gamma)
                    self.total_prox_g_evals += 1

                    V_k_plus_1 = self.compute_V(
                        zk_plus_1, zbar_plus_1, wk_plus_1, Fzk_plus_1, Fzbar_plus_1
                    )

                    if V_k_plus_1 <= V_k - sigma_term:
                        break
                    m += 1

                if m > M:
                    # If the line search exhausts M steps
                    tau_k = 0.0
                    
                    zk_plus_1 = wk

                    Fzk_plus_1 = self.problem.F(zk_plus_1)
                    self.total_F_evals += 1

                    zbar_plus_1 = self.problem.prox_g(zk_plus_1 - gamma * Fzk_plus_1, gamma)
                    self.total_prox_g_evals += 1

                    Fzbar_plus_1 = self.problem.F(zbar_plus_1)
                    self.total_F_evals += 1

                    wk_plus_1 = self.problem.prox_g(zk_plus_1 - gamma * Fzbar_plus_1, gamma)
                    self.total_prox_g_evals += 1

                    V_k_plus_1 = self.compute_V(
                        zk_plus_1, zbar_plus_1, wk_plus_1, Fzk_plus_1, Fzbar_plus_1
                    )

            perf_metric = self.performance_evaluator(zk=zk_plus_1, Fzk=Fzk_plus_1)
            self.record_performance(iteration=k, perf_metric=perf_metric)
            self._tau_history.append(tau_k)

            if self.has_converged(perf_metric=perf_metric, tol=self.tol):
                print(f"    {self.__class__.__name__} converged at iteration {k}")
                break

            self.direction.post_step_update(
                zk, Fzk, zbar, Fzbar, wk,
                zk_plus_1, Fzk_plus_1, zbar_plus_1,
                Fzbar_plus_1, wk_plus_1, k
            )

            zk, Fzk, zbar, Fzbar, wk = (
                zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1
            )
            V_k = V_k_plus_1

        return zk

    def get_results(self):
        results = super().get_results()
        results['tau'] = self._tau_history
        return results

    def compute_V(self, zk, zbar, wk, Fzk, Fzbar):
        """
        Computes the value of V_k for the line search condition in ProxFLEX.

        Parameters:
            zk (np.ndarray): Current iterate.
            zbar (np.ndarray): Extrapolated point (prox).
            wk (np.ndarray): Intermediate point (prox).
            Fzk (np.ndarray): Operator evaluated at zk.
            Fzbar (np.ndarray): Operator evaluated at zbar.

        Returns:
            float: The value of V_k for contraction checks.
        """
        gamma = self.gamma
        z_minus_w = zk - wk
        F_diff = Fzk - Fzbar
        w_minus_zbar = wk - zbar
        V_k = (
            2 * gamma**(-1) * np.dot(z_minus_w, F_diff)
            + gamma**(-2) * np.linalg.norm(w_minus_zbar) ** 2
            + gamma**(-2) * np.linalg.norm(z_minus_w) ** 2
        )
        return V_k
