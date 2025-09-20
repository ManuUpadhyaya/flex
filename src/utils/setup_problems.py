# src/utils/setup_problems.py

import numpy as np

from .data_utils import get_num_datapoints

from problems import (
    QuadraticMinimaxProblem,
    RotationR2Problem,
    CournotNashProblem,
    LogisticRegressionProblem,
    BilinearZeroSumGameProblem
)

from algorithms import (
    Extragradient, EAGC, GRAAL, AGRAAL, EGAA, FISTA,
    FLEX, IFLEX, ProxFLEX
)

from algorithms import (
    AAI, AAII, Jsymmetric, RegularizedNewton
)


def configure_algorithms(problem, algorithm_configs):
    """
    Instantiates algorithms based on provided configurations for a given problem.

    Parameters:
        problem: The problem instance.
        algorithm_configs (list): A list of dictionaries, each containing:
            - 'label': (str) Algorithm label.
            - 'class': Algorithm class reference.
            - 'force_run': (bool) Whether to force rerun the algorithm.
            - 'params': (dict) Parameters to initialize the algorithm.

    Returns:
        list: A list of dictionaries containing algorithm label, instance, and force_run flag.
    """
    algorithms = []
    
    for algo_conf in algorithm_configs:
        label = algo_conf['label']
        algo_class = algo_conf['class']
        force_run = algo_conf.get('force_run', False)  # Default to False if not specified
        params = algo_conf['params']
        
        # Instantiate the algorithm
        try:
            algo_instance = algo_class(**params)
            algorithms.append({
                'label': label,
                'instance': algo_instance,
                'force_run': force_run
            })
        except Exception as e:
            print(f"    Error initializing algorithm '{label}' for problem '{problem}'.")
            print(f"    Exception message: {e}\n")
    
    return algorithms

def setup_problem_qmp_n20_alpha0(reg_newton_only=0):
    # -----------------------------------------------------------------------------------
    # Quadratic minimax problem: n = 20, alpha = 0
    # -----------------------------------------------------------------------------------
    problem_label = 'quadratic minimax problem'

    # Initialize the problem
    n = 20
    alpha = 0
    seed = 42
    problem = QuadraticMinimaxProblem(n=n, alpha=alpha, seed=seed)
    z0 = np.zeros(problem.dim_z)
    
    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_qmp(zk, Fzk, **kwargs):
        return np.linalg.norm(Fzk)

    def stopping_criterion_qmp(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 100000
    tol = 1e-8
    ignore_starting_point = False
    FLEX_M = 2
    FLEX_sigma = 0.1
    FLEX_beta = 0.3
    FLEX_rho = 0.99
    fix_point_operator = 'FB' # 'EG' or 'FB'
    m_memory_AAI = 20
    m_memory_AAII = 20
    algorithm_configs = [
        {
            'label': 'FLEX-AAI',
            'class': FLEX,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M,         
                'beta': FLEX_beta,    
                'rho': FLEX_rho,     
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=m_memory_AAI,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'FLEX-AAII',
            'class': FLEX,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M,
                'beta': FLEX_beta,
                'rho': FLEX_rho,
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=m_memory_AAII,
                    fix_point_operator=fix_point_operator
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'FLEX-Jsym',
            'class': FLEX,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M, 
                'beta': FLEX_beta,
                'rho': FLEX_rho,     
                'tol': tol,
                'direction': Jsymmetric(
                    dim_x=n, 
                    dim_z=problem.dim_z,
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EAG-C',
            'class': EAGC,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'aGRAAL',
            'class': AGRAAL,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': 1e-8,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG-AA',
            'class': EGAA,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                't': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
    ]

    if reg_newton_only == 1:
        reg_newton_algorithms = [
            {
                'label': 'FLEX-RegNewton',
                'class': FLEX,
                'force_run': False,
                'params': {
                    'problem': problem,
                    'z0': z0,
                    'max_iter': max_iter,
                    'performance_evaluator': performance_evaluator_qmp,
                    'stopping_criterion': stopping_criterion_qmp,
                    'gamma': 0.9 / problem.L,
                    'sigma': FLEX_sigma,
                    'M': FLEX_M,
                    'beta': FLEX_beta,
                    'rho': FLEX_rho,
                    'tol': tol,
                    'direction': RegularizedNewton(problem=problem),
                    'ignore_starting_point': ignore_starting_point
                }
            }
        ]

        algorithm_configs.extend(reg_newton_algorithms)

        allowed_labels = {'EG', 'FLEX-RegNewton'}
        if any(cfg['label'] == 'IFLEX-RegNewton' for cfg in algorithm_configs):
            allowed_labels.add('IFLEX-RegNewton')
        algorithm_configs = [
            cfg for cfg in algorithm_configs if cfg['label'] in allowed_labels
        ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)
    
    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'n': n, 
            'alpha': alpha
        },
        'problem_instance': problem,  
        'count_operator_evals': False
    }

    return config


def setup_problem_qmp_n20_alpha_pos(reg_newton_only=0):
    # -----------------------------------------------------------------------------------
    # Quadratic minimax problem: n = 20, alpha = 0.0001
    # -----------------------------------------------------------------------------------
    problem_label = 'quadratic minimax problem'

    # Initialize the problem
    n = 20
    alpha = 0.0001
    seed = 42
    problem = QuadraticMinimaxProblem(n=n, alpha=alpha, seed=seed)
    z0 = np.zeros(problem.dim_z)
    
    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_qmp(zk, Fzk, **kwargs):
        return np.linalg.norm(Fzk)

    def stopping_criterion_qmp(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 100000
    tol = 1e-8
    ignore_starting_point = False
    FLEX_M = 2
    FLEX_sigma = 0.1
    FLEX_beta = 0.3
    FLEX_rho = 0.99
    FLEX_m_memory = 20
    fix_point_operator = 'FB'
    IFLEX_sigma = 0.1
    IFLEX_beta = 0.01
    _IFLEX_sigma = 0.99
    _IFLEX_beta = 0.01
    IFLEX_m_memory = 20
    force_run = False
    force_run_other = False
    algorithm_configs = [
        {
            'label': 'FLEX-AAI',
            'class': FLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,   
                'M': FLEX_M,         
                'beta': FLEX_beta,    
                'rho': FLEX_rho,     
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=FLEX_m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'FLEX-AAII',
            'class': FLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M,
                'beta': FLEX_beta,
                'rho': FLEX_rho,
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=FLEX_m_memory,
                    fix_point_operator=fix_point_operator
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'FLEX-Jsym',
            'class': FLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M, 
                'beta': FLEX_beta,
                'rho': FLEX_rho,     
                'tol': tol,
                'direction': Jsymmetric(
                    dim_x=n, 
                    dim_z=problem.dim_z,
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'IFLEX-AAI',
            'class': IFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': IFLEX_sigma,
                'beta': IFLEX_beta,  
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=IFLEX_m_memory,
                    fix_point_operator=fix_point_operator
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'IFLEX-AAII',
            'class': IFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': IFLEX_sigma,
                'beta': IFLEX_beta,
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=IFLEX_m_memory,
                    fix_point_operator=fix_point_operator
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'IFLEX-Jsym',
            'class': IFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': IFLEX_sigma,
                'beta': IFLEX_beta,
                'tol': tol,
                'direction': Jsymmetric(
                    dim_x=n, 
                    dim_z=problem.dim_z,
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EAG-C',
            'class': EAGC,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'aGRAAL',
            'class': AGRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': 1e-8,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG-AA',
            'class': EGAA,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                't': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
    ]

    if reg_newton_only == 1:
        reg_newton_algorithms = [
            {
                'label': 'FLEX-RegNewton',
                'class': FLEX,
                'force_run': force_run,
                'params': {
                    'problem': problem,
                    'z0': z0,
                    'max_iter': max_iter,
                    'performance_evaluator': performance_evaluator_qmp,
                    'stopping_criterion': stopping_criterion_qmp,
                    'gamma': 0.9 / problem.L,
                    'sigma': FLEX_sigma,
                    'M': FLEX_M,
                    'beta': FLEX_beta,
                    'rho': FLEX_rho,
                    'tol': tol,
                    'direction': RegularizedNewton(problem=problem),
                    'ignore_starting_point': ignore_starting_point
                }
            },
            {
                'label': 'IFLEX-RegNewton',
                'class': IFLEX,
                'force_run': force_run,
                'params': {
                    'problem': problem,
                    'z0': z0,
                    'max_iter': max_iter,
                    'performance_evaluator': performance_evaluator_qmp,
                    'stopping_criterion': stopping_criterion_qmp,
                    'gamma': 0.9 / problem.L,
                    'sigma': IFLEX_sigma,
                    'beta': IFLEX_beta,
                    'tol': tol,
                    'direction': RegularizedNewton(problem=problem),
                    'ignore_starting_point': ignore_starting_point
                }
            }
        ]

        algorithm_configs.extend(reg_newton_algorithms)

        allowed_labels = {'EG', 'FLEX-RegNewton', 'IFLEX-RegNewton'}
        algorithm_configs = [
            cfg for cfg in algorithm_configs if cfg['label'] in allowed_labels
        ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)
    
    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'n': n, 
            'alpha': alpha
        },
        'problem_instance': problem,  
        'count_operator_evals': False
    }

    return config

def setup_problem_qmp_n500_alpha0(reg_newton_only=0):
    # -----------------------------------------------------------------------------------
    # Quadratic minimax problem: n = 500, alpha = 0
    # -----------------------------------------------------------------------------------
    problem_label = 'quadratic minimax problem'

    # Initialize the problem
    n = 500
    alpha = 0
    seed = 42
    problem = QuadraticMinimaxProblem(n=n, alpha=alpha, seed=seed)
    z0 = np.zeros(problem.dim_z)
    
    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_qmp(zk, Fzk, **kwargs):
        return np.linalg.norm(Fzk)

    def stopping_criterion_qmp(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 100000
    tol = 1e-8
    ignore_starting_point = False
    FLEX_M = 2
    FLEX_sigma = 0.1
    FLEX_beta = 0.3
    FLEX_rho = 0.99
    fix_point_operator = 'FB' # 'EG' or 'FB'
    m_memory_AAI = 20
    m_memory_AAII = 20
    algorithm_configs = [
        {
            'label': 'FLEX-AAI',
            'class': FLEX,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M,         
                'beta': FLEX_beta,    
                'rho': FLEX_rho,     
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=m_memory_AAI,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'FLEX-AAII',
            'class': FLEX,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M,
                'beta': FLEX_beta,
                'rho': FLEX_rho,
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=m_memory_AAII,
                    fix_point_operator=fix_point_operator
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'FLEX-Jsym',
            'class': FLEX,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M, 
                'beta': FLEX_beta,
                'rho': FLEX_rho,     
                'tol': tol,
                'direction': Jsymmetric(
                    dim_x=n, 
                    dim_z=problem.dim_z,
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EAG-C',
            'class': EAGC,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'aGRAAL',
            'class': AGRAAL,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': 1e-8,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG-AA',
            'class': EGAA,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                't': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
    ]

    if reg_newton_only == 1:
        reg_newton_algorithms = [
            {
                'label': 'FLEX-RegNewton',
                'class': FLEX,
                'force_run': False,
                'params': {
                    'problem': problem,
                    'z0': z0,
                    'max_iter': max_iter,
                    'performance_evaluator': performance_evaluator_qmp,
                    'stopping_criterion': stopping_criterion_qmp,
                    'gamma': 0.9 / problem.L,
                    'sigma': FLEX_sigma,
                    'M': FLEX_M,
                    'beta': FLEX_beta,
                    'rho': FLEX_rho,
                    'tol': tol,
                    'direction': RegularizedNewton(problem=problem),
                    'ignore_starting_point': ignore_starting_point
                }
            }
        ]

        algorithm_configs.extend(reg_newton_algorithms)

        allowed_labels = {'EG', 'FLEX-RegNewton'}
        if any(cfg['label'] == 'IFLEX-RegNewton' for cfg in algorithm_configs):
            allowed_labels.add('IFLEX-RegNewton')
        algorithm_configs = [
            cfg for cfg in algorithm_configs if cfg['label'] in allowed_labels
        ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)
    
    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'n': n, 
            'alpha': alpha
        },
        'problem_instance': problem,  
        'count_operator_evals': False
    }

    return config

def setup_problem_qmp_n500_alpha_pos(reg_newton_only=0):
    # -----------------------------------------------------------------------------------
    # Quadratic minimax problem: n = 500, alpha = 0.0001
    # -----------------------------------------------------------------------------------
    problem_label = 'quadratic minimax problem'

    # Initialize the problem
    n = 500
    alpha = 0.0001
    seed = 42
    problem = QuadraticMinimaxProblem(n=n, alpha=alpha, seed=seed)
    z0 = np.zeros(problem.dim_z)
    
    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_qmp(zk, Fzk, **kwargs):
        return np.linalg.norm(Fzk)

    def stopping_criterion_qmp(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 100000
    tol = 1e-8
    ignore_starting_point = False
    FLEX_M = 2
    FLEX_sigma = 0.1
    FLEX_beta = 0.3
    FLEX_rho = 0.99
    FLEX_m_memory = 20
    fix_point_operator = 'FB'
    IFLEX_sigma = 0.1
    IFLEX_beta = 0.01
    _IFLEX_sigma = 0.99
    _IFLEX_beta = 0.01
    IFLEX_m_memory = 20
    force_run = False
    force_run_other = False
    algorithm_configs = [
        {
            'label': 'FLEX-AAI',
            'class': FLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,   
                'M': FLEX_M,         
                'beta': FLEX_beta,    
                'rho': FLEX_rho,     
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=FLEX_m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'FLEX-AAII',
            'class': FLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M,
                'beta': FLEX_beta,
                'rho': FLEX_rho,
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=FLEX_m_memory,
                    fix_point_operator=fix_point_operator
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'FLEX-Jsym',
            'class': FLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M, 
                'beta': FLEX_beta,
                'rho': FLEX_rho,     
                'tol': tol,
                'direction': Jsymmetric(
                    dim_x=n, 
                    dim_z=problem.dim_z,
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'IFLEX-AAI',
            'class': IFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': IFLEX_sigma,
                'beta': IFLEX_beta,  
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=IFLEX_m_memory,
                    fix_point_operator=fix_point_operator
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'IFLEX-AAII',
            'class': IFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': IFLEX_sigma,
                'beta': IFLEX_beta,
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=IFLEX_m_memory,
                    fix_point_operator=fix_point_operator
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'IFLEX-Jsym',
            'class': IFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'gamma': 0.9 / problem.L,
                'sigma': IFLEX_sigma,
                'beta': IFLEX_beta,
                'tol': tol,
                'direction': Jsymmetric(
                    dim_x=n, 
                    dim_z=problem.dim_z,
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EAG-C',
            'class': EAGC,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'aGRAAL',
            'class': AGRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': 1e-8,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG-AA',
            'class': EGAA,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_qmp,
                'stopping_criterion': stopping_criterion_qmp,
                'tol': tol,
                't': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]

    if reg_newton_only == 1:
        reg_newton_algorithms = [
            {
                'label': 'FLEX-RegNewton',
                'class': FLEX,
                'force_run': force_run,
                'params': {
                    'problem': problem,
                    'z0': z0,
                    'max_iter': max_iter,
                    'performance_evaluator': performance_evaluator_qmp,
                    'stopping_criterion': stopping_criterion_qmp,
                    'gamma': 0.9 / problem.L,
                    'sigma': FLEX_sigma,
                    'M': FLEX_M,
                    'beta': FLEX_beta,
                    'rho': FLEX_rho,
                    'tol': tol,
                    'direction': RegularizedNewton(problem=problem),
                    'ignore_starting_point': ignore_starting_point
                }
            },
            {
                'label': 'IFLEX-RegNewton',
                'class': IFLEX,
                'force_run': force_run,
                'params': {
                    'problem': problem,
                    'z0': z0,
                    'max_iter': max_iter,
                    'performance_evaluator': performance_evaluator_qmp,
                    'stopping_criterion': stopping_criterion_qmp,
                    'gamma': 0.9 / problem.L,
                    'sigma': IFLEX_sigma,
                    'beta': IFLEX_beta,
                    'tol': tol,
                    'direction': RegularizedNewton(problem=problem),
                    'ignore_starting_point': ignore_starting_point
                }
            }
        ]

        algorithm_configs.extend(reg_newton_algorithms)

        allowed_labels = {'EG', 'FLEX-RegNewton', 'IFLEX-RegNewton'}
        algorithm_configs = [
            cfg for cfg in algorithm_configs if cfg['label'] in allowed_labels
        ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)
    
    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'n': n, 
            'alpha': alpha
        },
        'problem_instance': problem,  
        'count_operator_evals': False
    }

    return config


def setup_problem_r2_rotation():
    # -----------------------------------------------------------------------------------
    # Rotation in R^2
    # -----------------------------------------------------------------------------------
    problem_label = 'rotation in $\\mathbb{R}^2$' 
    
    problem = RotationR2Problem()
    z0 = np.ones(problem.dim_z)  # z0 in R^2
    
    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_rot(zk, Fzk, **kwargs):
        return np.linalg.norm(Fzk)

    def stopping_criterion_rot(perf_metric, tol, **kwargs):
        return perf_metric <= tol
    
    max_iter = 20
    tol = 1e-16
    ignore_starting_point = False
    FLEX_M = 2
    FLEX_sigma = 0.1
    FLEX_beta = 0.3
    FLEX_rho = 0.99
    fix_point_operator = 'FB'
    algorithm_configs = [
        {
            'label': 'FLEX-AAI',
            'class': FLEX,
            'force_run': True,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_rot,
                'stopping_criterion': stopping_criterion_rot,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M,         
                'beta': FLEX_beta,    
                'rho': FLEX_rho,     
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=2,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'FLEX-AAII',
            'class': FLEX,
            'force_run': True,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_rot,
                'stopping_criterion': stopping_criterion_rot,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M,
                'beta': FLEX_beta,
                'rho': FLEX_rho,
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=3,
                    fix_point_operator=fix_point_operator
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'FLEX-Jsym',
            'class': FLEX,
            'force_run': True,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_rot,
                'stopping_criterion': stopping_criterion_rot,
                'gamma': 0.9 / problem.L,
                'sigma': FLEX_sigma,
                'M': FLEX_M, 
                'beta': FLEX_beta,
                'rho': FLEX_rho,     
                'tol': tol,
                'direction': Jsymmetric(
                    dim_x=1, 
                    dim_z=problem.dim_z,
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_rot,
                'stopping_criterion': stopping_criterion_rot,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EAG-C',
            'class': EAGC,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_rot,
                'stopping_criterion': stopping_criterion_rot,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(),
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_rot,
                'stopping_criterion': stopping_criterion_rot,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'aGRAAL',
            'class': AGRAAL,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_rot,
                'stopping_criterion': stopping_criterion_rot,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG-AA',
            'class': EGAA,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_rot,
                'stopping_criterion': stopping_criterion_rot,
                'tol': tol,
                't': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]
    
    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)
    
    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            # Add any problem-specific parameters here if needed
        },
        'problem_instance': problem,
        'count_operator_evals': False
    }

    return config

def setup_problem_cnep_n10():
    # -----------------------------------------------------------------------------------
    # Cournot-Nash Equilibrium Problem: n = 10
    # -----------------------------------------------------------------------------------
    problem_label = 'Cournot-Nash equilibrium problem'

    # Initialize the problem
    n = 10  # You can adjust the number of players as needed
    seed = 42
    problem = CournotNashProblem(n=n, seed=seed)
    z0 = np.zeros(problem.dim_z)

    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_cournot(zk, Fzk, **kwargs):
        gamma = 1 / (2 * problem.L)
        r_k = (zk - problem.prox_g(zk - gamma * Fzk, gamma)) / gamma
        return np.linalg.norm(r_k)

    def stopping_criterion_cournot(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 1000
    tol = 1e-10
    ignore_starting_point = False
    ProxFLEX_M = 2
    ProxFLEX_sigma = 0.1
    ProxFLEX_beta = 0.3
    ProxFLEX_rho = 0.99
    fix_point_operator = 'FB'
    m_memory = 3
    force_run = True
    algorithm_configs = [
        {
            'label': 'ProxFLEX-AAI',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'ProxFLEX-AAII',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'aGRAAL',
            'class': AGRAAL,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG-AA',
            'class': EGAA,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'tol': tol,
                't': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)

    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'n': n
        },
        'problem_instance': problem,
        'count_operator_evals': True
    }

    return config

def setup_problem_cnep_n100():
    # -----------------------------------------------------------------------------------
    # Cournot-Nash Equilibrium Problem: n = 100
    # -----------------------------------------------------------------------------------
    problem_label = 'Cournot-Nash equilibrium problem'

    # Initialize the problem
    n = 100  # You can adjust the number of players as needed
    seed = 42
    problem = CournotNashProblem(n=n, seed=seed)
    z0 = np.zeros(problem.dim_z)

    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_cournot(zk, Fzk, **kwargs):
        gamma = 1 / (2 * problem.L)
        r_k = (zk - problem.prox_g(zk - gamma * Fzk, gamma)) / gamma
        return np.linalg.norm(r_k)

    def stopping_criterion_cournot(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 2000
    tol = 1e-10
    ignore_starting_point = False
    ProxFLEX_M = 2
    ProxFLEX_sigma = 0.1
    ProxFLEX_beta = 0.3
    ProxFLEX_rho = 0.99
    fix_point_operator = 'FB'
    m_memory = 3
    force_run = True
    force_run_other = False
    algorithm_configs = [
        {
            'label': 'ProxFLEX-AAI',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'ProxFLEX-AAII',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG-AA',
            'class': EGAA,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'tol': tol,
                't': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'aGRAAL',
            'class': AGRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_cournot,
                'stopping_criterion': stopping_criterion_cournot,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)

    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'n': n
        },
        'problem_instance': problem,
        'count_operator_evals': True
    }

    return config


def setup_problem_logistic_data_spambase_lambda_1_over_N():
    # -----------------------------------------------------------------------------------
    # Logistic Regression Problem: data = spambase, lambda = 1/N
    # -----------------------------------------------------------------------------------
    problem_label = 'logistic regression problem'

    # Initialize the problem
    seed = 42
    data_name = 'spambase'  # Dataset name
    N = get_num_datapoints(data_name)
    lambda_param = 1/N  # Regularization parameter
    f_star = 1007.7506833766104 # for max_iter_fista=2000000 and lambda_param = 1/N

    problem = LogisticRegressionProblem(lambda_param=lambda_param, 
                                        data_name=data_name, 
                                        seed=seed, 
                                        max_iter_fista=2000000,
                                        f_star=f_star,
                                        f_star_method='manual')
    z0 = np.zeros(problem.dim_z)

    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_logistic(zk, Fzk, **kwargs):
        return problem.f(zk) - problem.f_star

    def stopping_criterion_logistic(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 1000000
    tol = 1e-10
    ignore_starting_point = False
    ProxFLEX_M = 2
    ProxFLEX_sigma = 0.1
    ProxFLEX_beta = 0.3
    ProxFLEX_rho = 0.99
    fix_point_operator = 'FB'
    m_memory = 6
    force_run = False
    force_run_other = False
    algorithm_configs = [
        {
            'label': 'ProxFLEX-AAI',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,
                'M': ProxFLEX_M,   
                'beta': ProxFLEX_beta,
                'rho': ProxFLEX_rho,
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'ProxFLEX-AAII',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        # {
        #     'label': 'aGRAAL',
        #     'class': AGRAAL,
        #     'force_run': force_run_other,
        #     'params': {
        #         'problem': problem,
        #         'z0': z0,
        #         'max_iter': max_iter*2,
        #         'performance_evaluator': performance_evaluator_logistic,
        #         'stopping_criterion': stopping_criterion_logistic,
        #         'tol': tol,
        #         'ignore_starting_point': ignore_starting_point
        #     }
        # },
        {
            'label': 'FISTA',
            'class': FISTA,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'L_f': problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)

    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'lambda_param': lambda_param,
            'data_name': data_name
        },
        'problem_instance': problem,
        'count_operator_evals': True
    }

    return config

def setup_problem_logistic_data_spambase_lambda_10_over_N():
    # -----------------------------------------------------------------------------------
    # Logistic Regression Problem: data = spambase, lambda = 10/N
    # -----------------------------------------------------------------------------------
    problem_label = 'logistic regression problem'

    # Initialize the problem
    seed = 42
    data_name = 'spambase'  # Dataset name
    N = get_num_datapoints(data_name)
    lambda_param = 10/N  # Regularization parameter
    f_star = 1007.8763644963917 # for max_iter_fista=2000000 and lambda_param = 10/N

    problem = LogisticRegressionProblem(lambda_param=lambda_param, 
                                        data_name=data_name, 
                                        seed=seed, 
                                        max_iter_fista=2000000,
                                        f_star=f_star,
                                        f_star_method='manual')
    z0 = np.zeros(problem.dim_z)

    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_logistic(zk, Fzk, **kwargs):
        return problem.f(zk) - problem.f_star

    def stopping_criterion_logistic(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 1000000
    tol = 1e-10
    ignore_starting_point = False
    ProxFLEX_M = 2
    ProxFLEX_sigma = 0.1
    ProxFLEX_beta = 0.3
    ProxFLEX_rho = 0.99
    fix_point_operator = 'FB'
    m_memory = 6
    force_run = False
    force_run_other = False
    algorithm_configs = [
        {
            'label': 'ProxFLEX-AAI',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,
                'M': ProxFLEX_M,   
                'beta': ProxFLEX_beta,
                'rho': ProxFLEX_rho,
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'ProxFLEX-AAII',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        # {
        #     'label': 'aGRAAL',
        #     'class': AGRAAL,
        #     'force_run': force_run_other,
        #     'params': {
        #         'problem': problem,
        #         'z0': z0,
        #         'max_iter': max_iter*2,
        #         'performance_evaluator': performance_evaluator_logistic,
        #         'stopping_criterion': stopping_criterion_logistic,
        #         'tol': tol,
        #         'ignore_starting_point': ignore_starting_point
        #     }
        # },
        {
            'label': 'FISTA',
            'class': FISTA,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'L_f': problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)

    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'lambda_param': lambda_param,
            'data_name': data_name
        },
        'problem_instance': problem,
        'count_operator_evals': True
    }

    return config

def setup_problem_logistic_data_spambase_lambda_100_over_N():
    # -----------------------------------------------------------------------------------
    # Logistic Regression Problem: data = spambase, lambda = 100/N
    # -----------------------------------------------------------------------------------
    problem_label = 'logistic regression problem'

    # Initialize the problem
    seed = 42
    data_name = 'spambase'  # Dataset name
    N = get_num_datapoints(data_name)
    lambda_param = 100/N  # Regularization parameter
    f_star = 1009.1280211254257 # for max_iter_fista=2000000 and lambda_param = 100/N

    problem = LogisticRegressionProblem(lambda_param=lambda_param, 
                                        data_name=data_name, 
                                        seed=seed, 
                                        max_iter_fista=2000000,
                                        f_star=f_star,
                                        f_star_method='manual')
    z0 = np.zeros(problem.dim_z)

    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_logistic(zk, Fzk, **kwargs):
        return problem.f(zk) - problem.f_star

    def stopping_criterion_logistic(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 1000000
    tol = 1e-10
    ignore_starting_point = False
    ProxFLEX_M = 2
    ProxFLEX_sigma = 0.1
    ProxFLEX_beta = 0.3
    ProxFLEX_rho = 0.99
    fix_point_operator = 'FB'
    m_memory = 6
    force_run = False
    force_run_other = False
    algorithm_configs = [
        {
            'label': 'ProxFLEX-AAI',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,
                'M': ProxFLEX_M,   
                'beta': ProxFLEX_beta,
                'rho': ProxFLEX_rho,
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'ProxFLEX-AAII',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        # {
        #     'label': 'aGRAAL',
        #     'class': AGRAAL,
        #     'force_run': force_run_other,
        #     'params': {
        #         'problem': problem,
        #         'z0': z0,
        #         'max_iter': max_iter*2,
        #         'performance_evaluator': performance_evaluator_logistic,
        #         'stopping_criterion': stopping_criterion_logistic,
        #         'tol': tol,
        #         'ignore_starting_point': ignore_starting_point
        #     }
        # },
        {
            'label': 'FISTA',
            'class': FISTA,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'L_f': problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)

    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'lambda_param': lambda_param,
            'data_name': data_name
        },
        'problem_instance': problem,
        'count_operator_evals': True
    }

    return config

def setup_problem_logistic_data_a9a_lambda_1_over_N():

    # -----------------------------------------------------------------------------------
    # Logistic Regression Problem: data = a9a, lambda = 1/N
    # -----------------------------------------------------------------------------------
    problem_label = 'logistic regression problem'

    # Initialize the problem
    seed = 42
    #data_name = 'breast_cancer'  # Dataset name
    data_name = 'a9a'  # Dataset name
    N = get_num_datapoints(data_name)
    lambda_param = 1/N  # Regularization parameter
    f_star = 10504.8562236346  # using max_iter_fista = 2000000 for data_name = 'a9a' and lambda_param = 1/N

    problem = LogisticRegressionProblem(lambda_param=lambda_param, 
                                        data_name=data_name, 
                                        seed=seed, 
                                        max_iter_fista=2000000, 
                                        f_star = f_star,
                                        f_star_method='manual')
    z0 = np.zeros(problem.dim_z)

    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_logistic(zk, Fzk, **kwargs):
        return problem.f(zk) - problem.f_star

    def stopping_criterion_logistic(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 1000000
    tol = 1e-10
    ignore_starting_point = False
    ProxFLEX_M = 2
    ProxFLEX_sigma = 0.1
    ProxFLEX_beta = 0.3
    ProxFLEX_rho = 0.99
    fix_point_operator = 'FB'
    m_memory = 10
    force_run = False
    force_run_other = False
    algorithm_configs = [
        {
            'label': 'ProxFLEX-AAI',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,
                'M': ProxFLEX_M,   
                'beta': ProxFLEX_beta,
                'rho': ProxFLEX_rho,
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'ProxFLEX-AAII',
            'class': ProxFLEX,
            'force_run': True, # force_run
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        # {
        #     'label': 'aGRAAL',
        #     'class': AGRAAL,
        #     'force_run': force_run_other,
        #     'params': {
        #         'problem': problem,
        #         'z0': z0,
        #         'max_iter': max_iter*2,
        #         'performance_evaluator': performance_evaluator_logistic,
        #         'stopping_criterion': stopping_criterion_logistic,
        #         'tol': tol,
        #         'ignore_starting_point': ignore_starting_point
        #     }
        # },
        {
            'label': 'FISTA',
            'class': FISTA,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'L_f': problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)

    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'lambda_param': lambda_param,
            'data_name': data_name
        },
        'problem_instance': problem,
        'count_operator_evals': True
    }

    return config

def setup_problem_logistic_data_a9a_lambda_10_over_N():

    # -----------------------------------------------------------------------------------
    # Logistic Regression Problem: data = a9a, lambda = 10/N
    # -----------------------------------------------------------------------------------
    problem_label = 'logistic regression problem'

    # Initialize the problem
    seed = 42
    #data_name = 'breast_cancer'  # Dataset name
    data_name = 'a9a'  # Dataset name
    N = get_num_datapoints(data_name)
    lambda_param = 10/N  # Regularization parameter
    f_star = 10504.88286929581  # using max_iter_fista = 1000000 for data_name = 'a9a' and lambda_param = 10/N

    problem = LogisticRegressionProblem(lambda_param=lambda_param, 
                                        data_name=data_name, 
                                        seed=seed, 
                                        max_iter_fista=1000000, 
                                        f_star = f_star,
                                        f_star_method='manual')
    z0 = np.zeros(problem.dim_z)

    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_logistic(zk, Fzk, **kwargs):
        return problem.f(zk) - problem.f_star

    def stopping_criterion_logistic(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 1000000
    tol = 1e-10
    ignore_starting_point = False
    ProxFLEX_M = 2
    ProxFLEX_sigma = 0.1
    ProxFLEX_beta = 0.3
    ProxFLEX_rho = 0.99
    fix_point_operator = 'FB'
    m_memory = 10
    # force_run = False
    force_run_other = False
    algorithm_configs = [
        {
            'label': 'ProxFLEX-AAI',
            'class': ProxFLEX,
            'force_run': False,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,
                'M': ProxFLEX_M,   
                'beta': ProxFLEX_beta,
                'rho': ProxFLEX_rho,
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'ProxFLEX-AAII',
            'class': ProxFLEX,
            'force_run': True,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        # {
        #     'label': 'aGRAAL',
        #     'class': AGRAAL,
        #     'force_run': force_run_other,
        #     'params': {
        #         'problem': problem,
        #         'z0': z0,
        #         'max_iter': max_iter*2,
        #         'performance_evaluator': performance_evaluator_logistic,
        #         'stopping_criterion': stopping_criterion_logistic,
        #         'tol': tol,
        #         'ignore_starting_point': ignore_starting_point
        #     }
        # },
        {
            'label': 'FISTA',
            'class': FISTA,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'L_f': problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)

    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'lambda_param': lambda_param,
            'data_name': data_name
        },
        'problem_instance': problem,
        'count_operator_evals': True
    }

    return config


def setup_problem_logistic_data_a9a_lambda_100_over_N():

    # -----------------------------------------------------------------------------------
    # Logistic Regression Problem: data = a9a, lambda = 100/N
    # -----------------------------------------------------------------------------------
    problem_label = 'logistic regression problem'

    # Initialize the problem
    seed = 42
    data_name = 'a9a'  # Dataset name
    N = get_num_datapoints(data_name)
    lambda_param = 100/N  # Regularization parameter
    f_star = 10505.117394775218  # using max_iter_fista = 2000000 for data_name = 'a9a' and lambda_param = 100/N

    problem = LogisticRegressionProblem(lambda_param=lambda_param, 
                                        data_name=data_name, 
                                        seed=seed, 
                                        max_iter_fista=2000000, 
                                        f_star = f_star,
                                        f_star_method='manual')
    z0 = np.zeros(problem.dim_z)

    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_logistic(zk, Fzk, **kwargs):
        return problem.f(zk) - problem.f_star

    def stopping_criterion_logistic(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 1000000
    tol = 1e-10
    ignore_starting_point = False
    ProxFLEX_M = 2
    ProxFLEX_sigma = 0.1
    ProxFLEX_beta = 0.3
    ProxFLEX_rho = 0.99
    fix_point_operator = 'FB'
    m_memory = 10
    force_run = False
    force_run_other = False
    algorithm_configs = [
        {
            'label': 'ProxFLEX-AAI',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,
                'M': ProxFLEX_M,   
                'beta': ProxFLEX_beta,
                'rho': ProxFLEX_rho,
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'ProxFLEX-AAII',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        # {
        #     'label': 'aGRAAL',
        #     'class': AGRAAL,
        #     'force_run': force_run_other,
        #     'params': {
        #         'problem': problem,
        #         'z0': z0,
        #         'max_iter': max_iter*2,
        #         'performance_evaluator': performance_evaluator_logistic,
        #         'stopping_criterion': stopping_criterion_logistic,
        #         'tol': tol,
        #         'ignore_starting_point': ignore_starting_point
        #     }
        # },
        {
            'label': 'FISTA',
            'class': FISTA,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_logistic,
                'stopping_criterion': stopping_criterion_logistic,
                'tol': tol,
                'L_f': problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)

    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'lambda_param': lambda_param,
            'data_name': data_name
        },
        'problem_instance': problem,
        'count_operator_evals': True
    }

    return config


def setup_problem_bmg_n500():
    # -----------------------------------------------------------------------------------
    # Bilinear matrix game: n = 500
    # -----------------------------------------------------------------------------------
    problem_label = 'bilinear matrix game'

    # Initialize the problem
    n = 500
    seed = 42
    problem = BilinearZeroSumGameProblem(n=n, seed=seed)
    z0 = np.ones(problem.dim_z) / n

    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_bmg(zk, Fzk, **kwargs):
        gamma = 1 / (2 * problem.L)
        r_k = (zk - problem.prox_g(zk - gamma * Fzk, gamma)) / gamma
        return np.linalg.norm(r_k)

    def stopping_criterion_bmg(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 100000
    tol = 1e-10
    ignore_starting_point = False
    ProxFLEX_M = 2
    ProxFLEX_sigma = 0.1
    ProxFLEX_beta = 0.3
    ProxFLEX_rho = 0.99
    fix_point_operator = 'FB'
    AAI_m_memory = 20
    AAII_m_memory = 20
    force_run = False
    force_run_other = False
    algorithm_configs = [
        {
            'label': 'ProxFLEX-AAI',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=AAI_m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'ProxFLEX-AAII',
            'class': ProxFLEX,
            'force_run': True,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=AAII_m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'aGRAAL',
            'class': AGRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG-AA',
            'class': EGAA,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'tol': tol,
                't': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)

    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'n': n
        },
        'problem_instance': problem,
        'count_operator_evals': True
    }

    return config

def setup_problem_bmg_n250():
    # -----------------------------------------------------------------------------------
    # Bilinear matrix game: n = 250
    # -----------------------------------------------------------------------------------
    problem_label = 'bilinear matrix game'

    # Initialize the problem
    n = 250
    seed = 42
    problem = BilinearZeroSumGameProblem(n=n, seed=seed)
    z0 = np.ones(problem.dim_z) / n

    # Define problem-specific performance evaluator and stopping criterion
    def performance_evaluator_bmg(zk, Fzk, **kwargs):
        gamma = 1 / (2 * problem.L)
        r_k = (zk - problem.prox_g(zk - gamma * Fzk, gamma)) / gamma
        return np.linalg.norm(r_k)

    def stopping_criterion_bmg(perf_metric, tol, **kwargs):
        return perf_metric <= tol

    max_iter = 100000
    tol = 1e-10
    ignore_starting_point = False
    ProxFLEX_M = 2
    ProxFLEX_sigma = 0.1
    ProxFLEX_beta = 0.3
    ProxFLEX_rho = 0.99
    fix_point_operator = 'FB'
    AAI_m_memory = 10
    AAII_m_memory = 10
    force_run = True
    force_run_other = False
    algorithm_configs = [
        {
            'label': 'ProxFLEX-AAI',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAI(
                    dim_z=problem.dim_z,
                    m_memory=AAI_m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'ProxFLEX-AAII',
            'class': ProxFLEX,
            'force_run': force_run,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,                   
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'gamma': 0.9 / problem.L,
                'sigma': ProxFLEX_sigma,   
                'M': ProxFLEX_M,         
                'beta': ProxFLEX_beta,    
                'rho': ProxFLEX_rho,     
                'tol': tol,
                'direction': AAII(
                    dim_z=problem.dim_z,
                    m_memory=AAII_m_memory,               
                    fix_point_operator=fix_point_operator   
                ),
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG',
            'class': Extragradient,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'tol': tol,
                'gamma': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'GRAAL',
            'class': GRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'z_bar_m1': z0.copy(), 
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'aGRAAL',
            'class': AGRAAL,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter*2,
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'tol': tol,
                'ignore_starting_point': ignore_starting_point
            }
        },
        {
            'label': 'EG-AA',
            'class': EGAA,
            'force_run': force_run_other,
            'params': {
                'problem': problem,
                'z0': z0,
                'max_iter': max_iter,
                'performance_evaluator': performance_evaluator_bmg,
                'stopping_criterion': stopping_criterion_bmg,
                'tol': tol,
                't': 0.9 / problem.L,
                'ignore_starting_point': ignore_starting_point
            }
        }
    ]

    algorithms = configure_algorithms(problem=problem, algorithm_configs=algorithm_configs)

    config = {
        'label': problem_label,
        'algorithms': algorithms,
        'parameters': {
            'n': n
        },
        'problem_instance': problem,
        'count_operator_evals': True
    }

    return config
