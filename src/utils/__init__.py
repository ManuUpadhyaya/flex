from .filename_utils import sanitize_filename
from .latex_utils import LATEX_KEY_MAP
from .plotting import plot_performance, plot_tau_history
from .data_utils import get_data_filepath, save_results, load_results, get_num_datapoints
from .setup_problems import (
    setup_problem_qmp_n20_omega0,
    setup_problem_qmp_n20_omega_pos,
    setup_problem_qmp_n500_omega0,
    setup_problem_qmp_n500_omega_pos,
    setup_problem_r2_rotation,
    setup_problem_cnep_n10,
    setup_problem_cnep_n100,
    setup_problem_logistic_data_a9a_lambda_100_over_N,
    setup_problem_logistic_data_a9a_lambda_10_over_N,
    setup_problem_logistic_data_a9a_lambda_1_over_N,
    setup_problem_logistic_data_spambase_lambda_100_over_N,
    setup_problem_logistic_data_spambase_lambda_10_over_N,
    setup_problem_logistic_data_spambase_lambda_1_over_N,
    setup_problem_bmg_n500,
    setup_problem_bmg_n250,
)

__all__ = [
    'sanitize_filename',
    'LATEX_KEY_MAP',
    'plot_performance',
    'plot_tau_history',
    'get_data_filepath',
    'save_results',
    'load_results',
    'get_num_datapoints',
    'setup_problem_qmp_n20_omega0',
    'setup_problem_qmp_n20_omega_pos', 
    'setup_problem_qmp_n500_omega0',
    'setup_problem_qmp_n500_omega_pos', 
    'setup_problem_r2_rotation',
    'setup_problem_cnep_n10',
    'setup_problem_cnep_n100',
    'setup_problem_logistic_data_a9a_lambda_100_over_N',
    'setup_problem_logistic_data_a9a_lambda_10_over_N',
    'setup_problem_logistic_data_a9a_lambda_1_over_N',
    'setup_problem_logistic_data_spambase_lambda_100_over_N',
    'setup_problem_logistic_data_spambase_lambda_10_over_N',
    'setup_problem_logistic_data_spambase_lambda_1_over_N',
    'setup_problem_bmg_n500',
    'setup_problem_bmg_n250',
]
