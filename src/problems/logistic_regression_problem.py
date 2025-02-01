# src/problems/logistic_regression_problem.py

import numpy as np
from .base_problem import Problem
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
import urllib.request
from scipy.io import arff
import zipfile

class LogisticRegressionProblem(Problem):
    def __init__(self, a_i=None, b_i=None, lambda_param=0.1, data_name='breast_cancer', 
                 data_path=None, max_iter_sklearn=10000, max_iter_fista=10000, f_star=None, seed=None, f_star_method='sklearn'):
        """
        Initializes the LogisticRegressionProblem.

        Parameters:
            a_i (np.ndarray or sparse matrix, optional): Data matrix of shape (m, n).
            b_i (np.ndarray, optional): Labels vector of shape (m,).
            lambda_param (float): Regularization parameter lambda.
            data_name (str): Name of dataset to load if a_i and b_i are not provided.
            data_path (str, optional): Path to the dataset file if using 'custom_svmlight'.
            max_iter_sklearn (int): Maximum iterations for scikit-learn solver.
            seed (int, optional): Random seed.
            f_star_method (str): Method to compute f^star  ('sklearn' or 'fista' or 'manual').
        """
        self.lambda_param = lambda_param

        if a_i is not None and b_i is not None:
            self.a_i = a_i
            self.b_i = b_i
        else:
            # Load dataset
            self._load_dataset(data_name, data_path)

        # Ensure labels are in {-1, 1}
        self.b_i = np.where(self.b_i <= 0, -1, 1)

        # Shuffle data if seed is provided
        if seed is not None:
            np.random.seed(seed)
            perm = np.random.permutation(len(self.b_i))
            self.a_i = self.a_i[perm]
            self.b_i = self.b_i[perm]

        # Standardize features if data is dense
        if not hasattr(self.a_i, 'toarray'):
            scaler = StandardScaler()
            self.a_i = scaler.fit_transform(self.a_i)

        self.m, self.n = self.a_i.shape
        print(f"(Number of data points) m = {self.m}")
        print(f"(Number of features) n = {self.n}")

        self.seed = seed

        # Initialize the base class and compute Lipschitz constant
        L = self._compute_lipschitz_constant()
        super().__init__(dim_z=self.n, L=L)

        # Compute f_star using the specified method
        self.max_iter_sklearn = max_iter_sklearn
        self.max_iter_fista = max_iter_fista
        if f_star_method == 'sklearn':
            self.f_star = self.compute_f_star_sklearn()
            print("f_star computed using scikit-learn:", self.f_star)
        elif f_star_method == 'fista':
            self.f_star = self.compute_f_star_fista()
            print("f_star computed using FISTA:", self.f_star)
        elif f_star_method == 'manual':
            self.f_star = f_star
        else:
            raise ValueError(f"Unsupported f_star_method: {f_star_method}")

    def get_num_data_points(self):
        """
        Returns the number of data points (samples) in the problem.

        Returns:
            int: The number of data points, m.
        """
        return self.m

    def f(self, x):
        """
        Evaluates the objective function f at a given point x.

        f(x) = sum_{i=1}^m log(1 + exp(-b_i * a_i^T x)) + lambda * ||x||_1

        Parameters:
            x (np.ndarray): The point at which to evaluate f, of shape (n,).

        Returns:
            float: The value of the objective function at x.
        """
        if x.shape[0] != self.dim_z:
            raise ValueError(f"Input x must be of dimension {self.dim_z}.")

        # Compute a_i^T x
        if hasattr(self.a_i, 'dot'):
            a_i_x = self.a_i.dot(x)
        else:
            a_i_x = self.a_i @ x

        # Compute -b_i * a_i^T x
        neg_b_a_i_x = -self.b_i * a_i_x

        # Compute log(1 + exp(-b_i * a_i^T x)) in a numerically stable way
        loss = np.where(
            neg_b_a_i_x > 0,
            neg_b_a_i_x + np.log1p(np.exp(-neg_b_a_i_x)),
            np.log1p(np.exp(neg_b_a_i_x))
        ).sum()

        # Compute L1 regularization term
        reg = self.lambda_param * np.linalg.norm(x, 1)

        return loss + reg
    
    def F(self, x):
        """
        Evaluates the operator F at a given point x.

        Parameters:
            x (np.ndarray): The point at which to evaluate F, of shape (n,).

        Returns:
            np.ndarray: The value of F at x, of shape (n,).
        """
        if x.shape[0] != self.dim_z:
            raise ValueError(f"Input x must be of dimension {self.dim_z}.")

        # Compute Kx = -b_i * (a_i @ x)
        if hasattr(self.a_i, 'dot'):
            a_i_x = self.a_i.dot(x)
        else:
            a_i_x = self.a_i @ x
        Kx = -self.b_i * a_i_x  # Shape (m,)

        # Compute sigma(Kx) using scipy's expit function
        sigma_Kx = expit(Kx)

        # Compute F(x) = K^T sigma(Kx)
        if hasattr(self.a_i, 'transpose'):
            F_x = -self.a_i.transpose().dot(self.b_i * sigma_Kx)  # Shape (n,)
        else:
            F_x = -self.a_i.T @ (self.b_i * sigma_Kx)

        return F_x

    def prox_g(self, x, gamma):
        """
        Computes the proximal operator of gamma * g at point x.

        Since g(x) = lambda * ||x||_1, the proximal operator is soft-thresholding.

        Parameters:
            x (np.ndarray): The point at which to compute the proximal operator, of shape (n,).
            gamma (float): Step size parameter.

        Returns:
            np.ndarray: The result of the proximal operator at x, of shape (n,).
        """
        if x.shape[0] != self.dim_z:
            raise ValueError(f"Input x must be of dimension {self.dim_z}.")

        # Soft-thresholding operator
        threshold = gamma * self.lambda_param
        prox_x = np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

        return prox_x

    def _load_dataset(self, data_name, data_path):
        """
        Loads a standard dataset.

        Parameters:
            data_name (str): Name of the dataset to load.
            data_path (str, optional): Path to the dataset file (used for some datasets).
        """
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        if data_name == 'breast_cancer':
            data = load_breast_cancer()
            self.a_i = data.data
            self.b_i = data.target  # 0 or 1

        elif data_name == 'a9a':
            if data_path is None:
                data_dir = os.path.join('data', 'a9a')
                data_path = os.path.join(data_dir, 'a9a')
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                if not os.path.exists(data_path):
                    print("Downloading 'a9a' dataset...")
                    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a'
                    urllib.request.urlretrieve(url, data_path)
                    print("Download complete.")
            else:
                if not os.path.exists(data_path):
                    print(f"Dataset file not found at {data_path}. Downloading...")
                    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a'
                    urllib.request.urlretrieve(url, data_path)
                    print("Download complete.")
            self.a_i, self.b_i = load_svmlight_file(data_path)
            self.b_i = np.where(self.b_i <= 0, -1, 1)

        elif data_name == 'rcv1':
            from sklearn.datasets import fetch_rcv1
            data = fetch_rcv1()
            self.a_i = data.data  # Sparse matrix
            # binary classification (for example: the first category as +1, otherwise -1)
            self.b_i = data.target[:, 0].toarray().ravel()

        elif data_name == 'news20':
            if data_path is None:
                raise ValueError("Please provide 'data_path' for the news20 dataset.")
            self.a_i, self.b_i = load_svmlight_file(data_path)

        elif data_name == 'webspam':
            if data_path is None:
                raise ValueError("Please provide 'data_path' for the webspam dataset.")
            self.a_i, self.b_i = load_svmlight_file(data_path)

        elif data_name == 'custom_svmlight':
            if data_path is None or not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset file not found at {data_path}")
            self.a_i, self.b_i = load_svmlight_file(data_path)

        elif data_name == 'mushroom':
            # Location of the mushroom dataset on UCI:
            # https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
            # This CSV has no header row; the first column is the label: 'p' (poisonous) or 'e' (edible).
            data_dir = os.path.join('data', 'mushroom')
            data_file = os.path.join(data_dir, 'mushroom.csv')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            if not os.path.exists(data_file):
                print("Downloading 'mushroom' dataset...")
                url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
                urllib.request.urlretrieve(url, data_file)
                print("Download complete.")

            # Read the CSV (no headers; 23 columns: col 0 => label, col 1..22 => features)
            df = pd.read_csv(data_file, header=None)

            # Convert labels 'p' -> -1, 'e' -> +1
            y_raw = df.iloc[:, 0]
            self.b_i = np.where(y_raw == 'p', -1, 1)

            X_raw = df.iloc[:, 1:]  # all feature columns

            # All features in mushroom are categorical; one-hot encode them
            # One way is to convert each column with LabelEncoder then apply OneHotEncoder, or directly:
            X_encoded = pd.get_dummies(X_raw, columns=X_raw.columns, prefix=X_raw.columns)

            self.a_i = X_encoded.values  # convert to numpy array

        elif data_name == 'website+phishing':
            # Website Phishing dataset from UCI #379
            # https://archive.ics.uci.edu/static/public/379/website+phishing.zip

            data_dir = os.path.join('data', 'website+phishing')
            os.makedirs(data_dir, exist_ok=True)

            # 1. Download the ZIP if it's not already present
            zip_url = 'https://archive.ics.uci.edu/static/public/379/website+phishing.zip'
            zip_file = os.path.join(data_dir, 'website_phishing.zip')
            arff_filename = 'PhishingData.arff'
            arff_path = os.path.join(data_dir, arff_filename)

            if not os.path.exists(zip_file):
                print("Downloading 'website+phishing' dataset (website+phishing.zip)...")
                urllib.request.urlretrieve(zip_url, zip_file)
                print("Download complete.")

            # 2. Extract the ARFF file if we haven't already
            if not os.path.exists(arff_path):
                print("Extracting 'website+phishing.zip'...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print("Extraction complete.")

            # 3. Load the ARFF file into a pandas DataFrame
            data, meta = arff.loadarff(arff_path)
            df = pd.DataFrame(data)

            # 4. Assume the label is the last column in the ARFF
            #    (If your label is a different column, adjust accordingly)
            # Assume the ARFF label column is the last column in df.
            # df might have label values like b'1', b'-1', b'0', etc.
            self.b_i = df.iloc[:, -1].values  # This is still bytes if the ARFF used nominal attributes

            # Convert from bytes to string, then to float
            self.b_i = self.b_i.astype(str).astype(float)

            # Now self.b_i is numeric, so you can do <= comparisons
            self.b_i = np.where(self.b_i <= 0, -1, 1)
            
            X_raw = df.iloc[:, :-1]

            # 5. Convert features and labels to numeric if needed
            # ARFF data often comes in string/object dtype. If the features are numeric,
            # pandas will usually interpret them automatically. Otherwise, convert as needed:
            self.a_i = X_raw.values.astype(float)

        elif data_name.lower() == 'phishing+websites':

            # Directory where we store this dataset
            data_dir = os.path.join('data', 'phishing+websites')
            os.makedirs(data_dir, exist_ok=True)

            # Direct link to the ZIP file
            zip_url = 'https://archive.ics.uci.edu/static/public/327/phishing+websites.zip'
            zip_file = os.path.join(data_dir, 'phishing_websites.zip')

            # Name of the ARFF file inside the ZIP
            arff_filename = 'Training Dataset.arff'  # Pay attention to the space in the filename
            arff_path = os.path.join(data_dir, arff_filename)

            # 1. Download the ZIP if not already present
            if not os.path.exists(zip_file):
                print("Downloading 'phishing+websites' dataset (phishing+websites.zip)...")
                urllib.request.urlretrieve(zip_url, zip_file)
                print("Download complete.")

            # 2. Extract the ARFF file if we haven't already
            if not os.path.exists(arff_path):
                print("Extracting 'phishing+websites.zip'...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print("Extraction complete.")

            # 3. Load the ARFF file into a pandas DataFrame
            data, meta = arff.loadarff(arff_path)
            df = pd.DataFrame(data)

            # 4. The dataset has 31 columns (30 features + 1 label).
            #    The label appears to be the last column (named 'Result' in ARFF).
            #    ARFF nominal values often come in as bytes, e.g., b'-1' and b'1'.
            self.b_i = df.iloc[:, -1].values  # label column

            # Convert from bytes (e.g. b'-1') -> str -> float
            self.b_i = self.b_i.astype(str).astype(float)

            # Ensure labels in {-1, +1}
            self.b_i = np.where(self.b_i <= 0, -1, 1)

            # 5. Features are everything else (first 30 columns).
            X_raw = df.iloc[:, :-1]
            # Convert to float in case they’re read as objects/bytes
            self.a_i = X_raw.values.astype(float)

        elif data_name == 'spambase':
            # Directory where we store the spambase dataset
            data_dir = os.path.join('data', 'spambase')
            data_file = os.path.join(data_dir, 'spambase.data')
            
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            # Download the dataset if not present
            if not os.path.exists(data_file):
                print("Downloading 'spambase' dataset...")
                url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
                urllib.request.urlretrieve(url, data_file)
                print("Download complete.")

            # Read the dataset (no header, 58 columns total: 57 features + 1 label)
            df = pd.read_csv(data_file, header=None)

            # The last column is the label: 1 => spam, 0 => not spam
            # Convert 1 => +1, 0 => -1
            self.b_i = np.where(df.iloc[:, -1] == 1, 1, -1)
            
            # Features are the first 57 columns
            self.a_i = df.iloc[:, :-1].values
    
        elif data_name == 'bank_marketing':
            # Directory where we store the bank marketing dataset
            data_dir = os.path.join('data', 'bank_marketing')
            os.makedirs(data_dir, exist_ok=True)

            # We'll use "bank-additional-full.csv" (larger version of the dataset)
            data_file = os.path.join(data_dir, 'bank-additional-full.csv')

            # Download if the file is not already present
            if not os.path.exists(data_file):
                print("Downloading 'bank_marketing' dataset (bank-additional-full.csv)...")
                url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv'
                urllib.request.urlretrieve(url, data_file)
                print("Download complete.")

            # Read the CSV with semicolon delimiter
            df = pd.read_csv(data_file, sep=';')

            # The label column is 'y', which has values 'yes' or 'no'
            # Convert 'yes' -> +1, 'no' -> -1
            self.b_i = np.where(df['y'] == 'yes', 1, -1)

            # Drop the label column so we only have features left
            df.drop(columns='y', inplace=True)

            # Many columns (e.g., job, marital, education) are categorical strings
            # We'll one-hot encode them so they're numeric
            X_encoded = pd.get_dummies(df, drop_first=False)

            self.a_i = X_encoded.values

        else:
            raise ValueError(f"Unknown dataset name '{data_name}'.")

    def _compute_lipschitz_constant(self):
        """
        Computes an estimate of the Lipschitz constant L_F.

        Returns:
            float: Estimated Lipschitz constant L_F.
        """
        K_norm_estimate = self._estimate_K_norm(num_iterations=500)
        L = (1/4) * K_norm_estimate ** 2
        return L

    def _estimate_K_norm(self, num_iterations=500):
        """
        Estimates the largest singular value (spectral norm) of K using the power iteration method.
        """
        rng = np.random.default_rng(self.seed)
        v = rng.standard_normal(self.n)
        v /= np.linalg.norm(v)
        for _ in range(num_iterations):
            if hasattr(self.a_i, 'dot'):
                a_i_v = self.a_i.dot(v)
            else:
                a_i_v = self.a_i @ v
            Kv = -self.b_i * a_i_v

            if hasattr(self.a_i, 'transpose'):
                KtKv = -self.a_i.transpose().dot(self.b_i * Kv)
            else:
                KtKv = -self.a_i.T @ (self.b_i * Kv)

            v_new = KtKv
            v_new_norm = np.linalg.norm(v_new)
            if v_new_norm == 0:
                break
            v = v_new / v_new_norm
        K_norm_estimate = np.sqrt(v_new_norm)
        return K_norm_estimate

    def compute_f_star_sklearn(self):
        """
        Computes the optimal objective function value (f_star) using scikit-learn's LogisticRegression.

        This method fits a LogisticRegression model with L1 regularization and a very low tolerance
        to ensure high precision in the solution.

        Returns:
            float: The optimal objective function value f_star.
        """
        model = LogisticRegression(
            penalty='l1',
            C=1.0 / self.lambda_param,
            solver='liblinear',
            tol=1e-8,
            fit_intercept=False,
            max_iter=self.max_iter_sklearn,
            random_state=self.seed
        )
        model.fit(self.a_i, self.b_i)
        x_opt = model.coef_.flatten()
        f_star = self.f(x_opt)
        return f_star
    
    def compute_f_star_fista(self, tol=1e-10):
        """
        Computes the optimal value  f^star  using the FISTA algorithm.

        Parameters:
            max_iter (int): Maximum number of iterations for FISTA.
            tol (float): Convergence tolerance.

        Returns:
            float: Optimal objective function value  f^star .
        """
        # Initialize
        x = np.zeros(self.n)
        y = x.copy()
        t = 1
        L = self.L

        for k in range(self.max_iter_fista):
            grad = self.F(y)  # Gradient of the smooth part
            prox_input = y - grad / L
            x_next = self.prox_g(prox_input, 1 / L)

            # Check convergence
            if np.linalg.norm(x_next - x) < tol:
                print(f"FISTA converged in {k+1} iterations.")
                break

            # Update variables
            t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x_next + ((t - 1) / t_next) * (x_next - x)
            x, t = x_next, t_next
        else:
            print(f"FISTA did not converge within {self.max_iter_fista} iterations.")

        return self.f(x)

    def get_plot_settings(self):
        """
        Returns the plotting settings specific to this problem.

        Returns:
            dict: A dictionary containing plot labels and titles.
        """
        return {
            'y_label': r'$f(z^k) - f(z^{\star})$',
            'x_label_iterations': 'Iteration $k$',
            'x_label_F_evals': 'Number of $F$ evaluations',
            'x_label_operator_evals': 'Number of operator evaluations',
            'x_label_time': 'Time (s)',
            'title_prefix': 'Convergence on'
        }
