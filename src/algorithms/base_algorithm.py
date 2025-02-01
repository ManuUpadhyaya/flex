# src/algorithms/base_algorithm.py

import time
import numpy as np

class IterativeAlgorithm:
    def __init__(self, problem, z0, max_iter, performance_evaluator, stopping_criterion, ignore_starting_point=True, callback=None):
        """
        Base class for iterative algorithms.

        Attributes:
            problem: The problem instance to solve.
            z0 (np.ndarray): Initial point.
            max_iter (int): Maximum number of iterations.
            performance_evaluator (callable): Function to evaluate performance metrics.
            stopping_criterion (callable): Function to determine convergence.
            ignore_starting_point (bool): If True, the starting point is ignored in performance recording.
            callback (callable, optional): Optional callback function executed at each iteration.
        """
        self.problem = problem
        self.z0 = z0
        self.max_iter = max_iter
        self.performance_evaluator = performance_evaluator
        self.stopping_criterion = stopping_criterion
        self.callback = callback
        self.ignore_starting_point = ignore_starting_point

        # Initialize performance tracking variables
        self.iterations = []
        self.num_F_evals = []
        self.num_prox_g_evals = []
        self.num_operator_evals = []
        self.times = []
        self.performance = []

        self.start_time = None
        self.total_F_evals = 0
        self.total_prox_g_evals = 0

    def run(self):
        """
        Runs the iterative algorithm.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def record_performance(self, iteration, perf_metric, **kwargs):
        """
        Records performance metrics at the current iteration.

        Parameters:
            iteration (int): Current iteration number.
            perf_metric: The performance metric computed at the current iteration.
            **kwargs: Additional keyword arguments.
        """
        current_time = time.time() - self.start_time

        # Record metrics
        self.iterations.append(iteration)
        self.num_F_evals.append(self.total_F_evals)
        self.num_prox_g_evals.append(self.total_prox_g_evals)
        self.num_operator_evals.append(self.total_F_evals + self.total_prox_g_evals)
        self.times.append(current_time)
        self.performance.append(perf_metric)

        # Optional callback
        if self.callback is not None:
            self.callback(iteration, perf_metric, **kwargs)

    def has_converged(self, perf_metric, **kwargs):
        """
        Determines whether the stopping criterion has been met.

        Parameters:
            perf_metric: The performance metric computed at the current iteration.
            **kwargs: Additional keyword arguments required by the stopping criterion.

        Returns:
            bool: True if the algorithm has converged, False otherwise.
        """
        return self.stopping_criterion(perf_metric, **kwargs)

    def get_results(self):
        """
        Retrieves the recorded performance metrics and evaluation counts.

        Returns:
            dict: A dictionary containing:
                - 'iterations': List of iteration numbers.
                - 'num_F_evals': List of cumulative counts of F evaluations.
                - 'num_prox_g_evals': List of cumulative counts of prox_g evaluations.
                - 'num_operator_evals': List of cumulative counts of total operator evaluations.
                - 'times': List of cumulative times at each iteration.
                - 'performance': List of performance metric values at each iteration.
        """
        return {
            'iterations': self.iterations,
            'num_F_evals': self.num_F_evals,
            'num_prox_g_evals': self.num_prox_g_evals,
            'num_operator_evals': self.num_operator_evals,
            'times': self.times,
            'performance': self.performance
        }
