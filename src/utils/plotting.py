import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import cycle

def setup_matplotlib():
    """
    Configures Matplotlib settings to use LaTeX rendering and consistent font styles.

    Sets up font family, sizes, and includes necessary LaTeX packages for rendering mathematical expressions.
    """
    # Enable LaTeX rendering
    matplotlib.rcParams['text.usetex'] = True

    # Set font family to match LaTeX default (Computer Modern)
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']

    # Set consistent font sizes
    font_size = 11  # You can adjust this value as needed
    matplotlib.rcParams['font.size'] = font_size  # Base font size
    matplotlib.rcParams['axes.titlesize'] = font_size
    matplotlib.rcParams['axes.labelsize'] = font_size
    matplotlib.rcParams['xtick.labelsize'] = font_size
    matplotlib.rcParams['ytick.labelsize'] = font_size
    matplotlib.rcParams['legend.fontsize'] = font_size
    matplotlib.rcParams['figure.titlesize'] = font_size

    # Use LaTeX default font for all text elements
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['mathtext.rm'] = 'serif'
    matplotlib.rcParams['mathtext.it'] = 'serif:italic'
    matplotlib.rcParams['mathtext.bf'] = 'serif:bold'

    # Include necessary LaTeX packages
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'

def plot_performance(results_list, labels_list, x_metric, y_metric, x_label, y_label, title, filename):
    """
    Plots the performance metrics of different algorithms with distinct line styles.

    Parameters:
        results_list (list): List of dictionaries containing performance data for each algorithm.
        labels_list (list): List of labels corresponding to each algorithm.
        x_metric (str): The key for the x-axis data in the results dictionaries.
        y_metric (str): The key for the y-axis data in the results dictionaries.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        filename (str): Path to save the generated plot image.

    Notes:
        - The function uses a logarithmic scale for the y-axis.
        - Legends are placed outside the plot area to avoid clutter.
        - Distinct line styles are used to help visually distinguish curves.
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a cycle of line styles. You can add or remove styles as needed.
    line_styles = cycle(['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))])

    for results, label in zip(results_list, labels_list):
        x = results.get(x_metric)
        y = results.get(y_metric)
        if x is None or y is None:
            print(f"Warning: Missing data for {label}. Skipping.")
            continue

        style = next(line_styles)  # Get the next line style in the cycle
        ax.semilogy(x, y, linestyle=style, label=label)

    ax.set_xlabel(x_label)

    # Rotate y-axis label horizontally
    if y_label == r'$f(z^k) - f(z^{\star})$':
        ax.set_ylabel(y_label, labelpad=40)
    else:
        ax.set_ylabel(y_label, rotation=0, labelpad=40)
    ax.yaxis.set_label_coords(-0.09, 0.5)  # Position label midway along y-axis

    ax.set_title(title)

    # Adjust the plot area to make room for the legend on the right
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # Place the legend outside to the right
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.015))

    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tau_history(results, label, x_label, title, filename):
    """
    Plots the step-size sequence tau_k for a single FLEX-family solver.

    Parameters:
        results (dict): Result dictionary for one algorithm run.
        label (str): Algorithm label for the legend/title context.
        x_label (str): Label for the x-axis.
        title (str): Plot title (already includes fractions summary).
        filename (str): Output path for the saved figure.
    """
    iterations = results.get('iterations')
    tau_values = results.get('tau')

    if iterations is None or tau_values is None:
        return False

    filtered = [(k, tau) for k, tau in zip(iterations, tau_values) if tau is not None]
    if not filtered:
        return False

    ks, taus = zip(*filtered)
    taus = np.asarray(taus, dtype=float)

    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ks, taus, linestyle='None', marker='o', markersize=4)

    ax.set_xlabel(x_label)
    ax.set_ylabel(r'$\tau_k$', rotation=0, labelpad=25)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(True, which='both', ls='--')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return True
