# main.py

import sys
import os

# Add the 'src' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from utils import (
    plot_performance,
    LATEX_KEY_MAP,
    sanitize_filename,
    setup_problem_qmp_n20_alpha0,
    setup_problem_qmp_n20_alpha_pos,
    setup_problem_qmp_n500_alpha0,
    setup_problem_qmp_n500_alpha_pos,
    setup_problem_cnep_n10,
    setup_problem_cnep_n100,
    setup_problem_logistic_data_a9a_lambda_1_over_N,
    setup_problem_logistic_data_a9a_lambda_10_over_N,
    setup_problem_logistic_data_a9a_lambda_100_over_N,
    setup_problem_logistic_data_spambase_lambda_1_over_N,
    setup_problem_logistic_data_spambase_lambda_10_over_N,
    setup_problem_logistic_data_spambase_lambda_100_over_N,
    setup_problem_bmg_n500,
    setup_problem_bmg_n250
)


def setup_problem_configurations():
    """
    Initializes and configures all problem instances along with their applicable algorithms.

    Returns:
        list: A list of dictionaries, each containing problem configuration details, including:
            - 'label': (str) The problem label.
            - 'algorithms': (list) List of algorithm instances applicable to the problem.
            - 'parameters': (dict) Parameters specific to the problem.
            - 'problem_instance': The instantiated problem object.
            - 'count_operator_evals': (bool) Flag indicating whether to count operator evaluations.
    """
    problem_configurations = []

    # problem_configurations.append(setup_problem_qmp_n20_alpha0()) # Quadratic minimax problem: n=20, alpha=0

    # problem_configurations.append(setup_problem_qmp_n20_alpha_pos()) # Quadratic minimax problem: n=20, alpha=0.0001

    # problem_configurations.append(setup_problem_qmp_n500_alpha0()) # Quadratic minimax problem: n=500, alpha=0

    # problem_configurations.append(setup_problem_qmp_n500_alpha_pos()) # Quadratic minimax problem: n=500, alpha=0.0001

    # problem_configurations.append(setup_problem_cnep_n10()) # Cournot-Nash Equilibrium Problem: n=10

    # problem_configurations.append(setup_problem_cnep_n100()) # Cournot-Nash Equilibrium Problem: n=100

    # problem_configurations.append(setup_problem_logistic_data_a9a_lambda_100_over_N()) # Logistic Regression Problem: data_name = 'a9a', lambda = 100/N

    # problem_configurations.append(setup_problem_logistic_data_a9a_lambda_10_over_N()) # Logistic Regression Problem: data_name = 'a9a', lambda = 10/N

    # problem_configurations.append(setup_problem_logistic_data_a9a_lambda_1_over_N()) # Logistic Regression Problem: data_name = 'a9a', lambda = 1/N

    # problem_configurations.append(setup_problem_logistic_data_spambase_lambda_100_over_N()) # Logistic Regression Problem: data_name = 'spambase', lambda = 100/N

    # problem_configurations.append(setup_problem_logistic_data_spambase_lambda_10_over_N()) # Logistic Regression Problem: data_name = 'spambase', lambda = 10/N
    
    # problem_configurations.append(setup_problem_logistic_data_spambase_lambda_1_over_N()) # Logistic Regression Problem: data_name = 'spambase', lambda = 1/N

    # problem_configurations.append(setup_problem_bmg_n500()) # Bilinear matrix game: n = 500

    # problem_configurations.append(setup_problem_bmg_n250()) # Bilinear matrix game: n = 250
    
    return problem_configurations

def run_algorithms_on_problem(problem_label, algorithms, problem_instance, problem_parameters):
    """
    Executes each algorithm on the specified problem and collects the results.

    Parameters:
        problem_label (str): Label of the problem.
        algorithms (list): List of dictionaries containing algorithm label, instance, and force_run flag.
        problem_instance: The problem instance.
        problem_parameters (dict): Parameters of the problem.

    Returns:
        tuple: A tuple containing lists of results and corresponding labels.
    """
    from utils.data_utils import get_data_filepath, save_results, load_results

    print(f"\n=== Solving {problem_label} ===\n")
    
    results_list = []
    labels_list = []
    
    for algo_conf in algorithms:
        algorithm_label = algo_conf['label']
        algorithm = algo_conf['instance']
        force_run = algo_conf['force_run']
        
        # Determine the filepath for the algorithm's results
        filepath = get_data_filepath(problem_label, problem_parameters, algorithm_label)
        
        # Check if data exists and force_run is False
        if os.path.exists(filepath) and not force_run:
            print(f"--> Loading existing data for '{algorithm_label}' on '{problem_label}'...")
            try:
                results = load_results(filepath)
                results_list.append(results)
                labels_list.append(algorithm_label)
                print(f"    Loaded existing data for '{algorithm_label}'.\n")
                continue
            except Exception as e:
                print(f"    Error loading data for '{algorithm_label}'. Will attempt to rerun.")
                print(f"    Exception message: {e}\n")
        
        # Run the algorithm and save results
        print(f"--> Running {algorithm_label} on {problem_label}...")
        try:
            algorithm.run()
            results = algorithm.get_results()
            save_results(filepath, results)
            results_list.append(results)
            labels_list.append(algorithm_label)
            print(f"    {algorithm_label} completed successfully and results saved.\n")
        except Exception as e:
            print(f"    Error: {algorithm_label} failed on {problem_label}.")
            print(f"    Exception message: {e}\n")
    
    return results_list, labels_list

def generate_plots(problem_label, results_list, labels_list, problem_parameters, problem_instance, count_operator_evals=False):
    """
    Generates and saves plots for the specified problem based on collected results.

    Parameters:
        problem_label (str): Label of the problem.
        results_list (list): List of results dictionaries from algorithms.
        labels_list (list): List of labels for the algorithms.
        problem_parameters (dict): Parameters of the problem.
        problem_instance: The problem instance.
        count_operator_evals (bool): Flag to decide whether to count both F and prox_g evaluations.
    """
    # Check if any results were collected before plotting
    if not results_list:
        print(f"No successful algorithm runs for {problem_label}. Skipping plot generation.\n")
        return

    # Ensure 'plots' directory exists
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Prepare problem parameter string for directory name
    parameters_str = '_'.join([f'{key}{value}' for key, value in problem_parameters.items()])
    parameters_id = sanitize_filename(parameters_str)

    # Prepare plot directory for the current problem and parameters
    problem_id = sanitize_filename(problem_label)
    if parameters_id:
        problem_plot_dir = os.path.join(plots_dir, problem_id, parameters_id)
    else:
        problem_plot_dir = os.path.join(plots_dir, problem_id)

    if not os.path.exists(problem_plot_dir):
        os.makedirs(problem_plot_dir)

    # Prepare a string representation of problem parameters for filenames
    filename_parameters_str = ''.join([f'_{key}{value}' for key, value in problem_parameters.items()])

    # Retrieve plot settings from the problem instance
    try:
        plot_settings = problem_instance.get_plot_settings()
    except AttributeError as e:
        print(f"Error: The problem '{problem_label}' does not implement 'get_plot_settings()'. Skipping plotting.")
        print(f"Exception message: {e}")
        return  # Skip plotting for this problem

    # Extract plot settings
    y_label = plot_settings.get('y_label', 'Performance metric')
    x_label_iterations = plot_settings.get('x_label_iterations', 'iteration $k$')
    x_label_F_evals = plot_settings.get('x_label_F_evals', 'function evaluations')
    x_label_operator_evals = plot_settings.get('x_label_operator_evals', 'operator evaluations') 
    x_label_time = plot_settings.get('x_label_time', 'time (s)')
    title_prefix = plot_settings.get('title_prefix', 'Convergence of')

    # Adjust plot title
    plot_title = f"{title_prefix} {problem_label}"

    # If there are problem parameters, append them to the title without parentheses
    if problem_parameters:
        parameters_list = []
        for key, value in problem_parameters.items():
            # Map the key to its LaTeX representation if available
            latex_key = LATEX_KEY_MAP.get(key, key)
            
            # Escape underscores in keys
            latex_key = latex_key.replace('_', '\\_')
            
            # Escape underscores in string values
            if isinstance(value, str):
                value = value.replace('_', '\\_')
            
            # Format the parameter for the title
            parameters_list.append(f"${latex_key}={value}$")
        
        parameters_str_for_title = ', '.join(parameters_list)
        plot_title += f" {parameters_str_for_title}"

    print(f"Generating plots for {problem_label} with parameters {problem_parameters}...")

    # Define base filename for plots
    base_filename = f'{problem_id}_convergence'

    # Plot performance vs. iteration number
    plot_performance(
        results_list,
        labels_list,
        x_metric='iterations',
        y_metric='performance',
        x_label=x_label_iterations,
        y_label=y_label,
        title=plot_title,
        filename=os.path.join(problem_plot_dir, f'{base_filename}_iterations.png')
    )

    # Decide which x_metric to use based on the flag
    if count_operator_evals:
        x_metric = 'num_operator_evals'
        x_label_evals = x_label_operator_evals
        filename_suffix = 'operator_evals'
    else:
        x_metric = 'num_F_evals'
        x_label_evals = x_label_F_evals
        filename_suffix = 'Fevals'

    # Plot performance vs. number of evaluations
    plot_performance(
        results_list,
        labels_list,
        x_metric=x_metric,
        y_metric='performance',
        x_label=x_label_evals,
        y_label=y_label,
        title=plot_title,
        filename=os.path.join(problem_plot_dir, f'{base_filename}_{filename_suffix}.png')
    )

    # Plot performance vs. wall clock time
    plot_performance(
        results_list,
        labels_list,
        x_metric='times',
        y_metric='performance',
        x_label=x_label_time,
        y_label=y_label,
        title=plot_title,
        filename=os.path.join(problem_plot_dir, f'{base_filename}_time.png')
    )

    print(f"Done.\n")


def main():
    """
    Main function to orchestrate problem configurations, algorithm executions, and plot generations.
    """
    # Setup problem configurations
    problem_configurations = setup_problem_configurations()

    # Iterate over each problem configuration
    for config in problem_configurations:
        problem_label = config['label']
        algorithms = config['algorithms']
        problem_parameters = config.get('parameters', {})
        problem_instance = config.get('problem_instance')  
        count_operator_evals = config.get('count_operator_evals', False)

        # Run algorithms on the current problem
        results_list, labels_list = run_algorithms_on_problem(
            problem_label=problem_label,
            algorithms=algorithms,
            problem_instance=problem_instance,
            problem_parameters=problem_parameters
        )

        # Generate plots based on the results
        generate_plots(
            problem_label=problem_label,
            results_list=results_list,
            labels_list=labels_list,
            problem_parameters=problem_parameters,
            problem_instance=problem_instance,
            count_operator_evals=count_operator_evals 
        )

if __name__ == "__main__":
    main()
