# src/utils/data_utils.py

import os
import json
import numpy as np
from typing import Any, Dict

from sklearn.datasets import load_svmlight_file
import urllib.request
import pandas as pd
from scipy.io import arff
import zipfile

from .filename_utils import sanitize_filename 

def get_data_filepath(problem_label: str, problem_params: Dict[str, Any], algorithm_label: str) -> str:
    """
    Constructs a unique filepath for storing the algorithm's results based on the problem and algorithm labels.
    """
    if problem_params:
        params_str = '_'.join([f"{key}{value}" for key, value in problem_params.items()])
        params_id = sanitize_filename(params_str)
        problem_id = f"{sanitize_filename(problem_label)}_{params_id}"
    else:
        problem_id = sanitize_filename(problem_label)
    
    algorithm_id = sanitize_filename(algorithm_label)
    
    data_dir = os.path.join('generated_data', problem_id)
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, f"{algorithm_id}.json")
    return filepath

def save_results(filepath: str, results: Dict[str, Any]) -> None:
    """
    Saves the algorithm's results to a JSON file.
    """
    def convert(obj):
        if isinstance(obj, (np.ndarray, list)):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(filepath, 'w') as f:
        json.dump(results, f, default=convert, indent=4)

def load_results(filepath: str) -> Dict[str, Any]:
    """
    Loads the algorithm's results from a JSON file.
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def get_num_datapoints(data_name: str, data_path: str = None) -> int:
    """
    Loads a given dataset (e.g., 'a9a', 'mushroom', 'website+phishing') and returns
    the number of data points (rows).

    Parameters:
        data_name (str): Name of the dataset. 
                         Supported: "a9a", "mushroom", "phishing".
        data_path (str, optional): Path to the dataset file.

    Returns:
        int: The number of data points (rows) in the dataset.

    Raises:
        ValueError: If the dataset name is not recognized.
        FileNotFoundError: If data_path does not exist for custom dataset.
    """

    # --------------------------------------------------------------------------
    # 1. a9a
    # --------------------------------------------------------------------------
    if data_name.lower() == 'a9a':
        if data_path is None:
            # Default location for 'a9a' data
            data_dir = os.path.join('data', 'a9a')
            data_path = os.path.join(data_dir, 'a9a')
            os.makedirs(data_dir, exist_ok=True)

            # Download if file not found
            if not os.path.exists(data_path):
                print("Downloading 'a9a' dataset...")
                url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a'
                urllib.request.urlretrieve(url, data_path)
                print("Download complete.")
        else:
            # If user-provided data_path doesn't exist, attempt download
            if not os.path.exists(data_path):
                print(f"Dataset file not found at {data_path}. Downloading...")
                url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a'
                urllib.request.urlretrieve(url, data_path)
                print("Download complete.")

        # Load the dataset using load_svmlight_file
        X, y = load_svmlight_file(data_path)
        # Return the number of rows
        return X.shape[0]

    # --------------------------------------------------------------------------
    # 2. Mushroom
    # --------------------------------------------------------------------------
    elif data_name.lower() == 'mushroom':
        # https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/
        # The file is agaricus-lepiota.data (no header). The first column is label, 
        # and the remaining 22 columns are features. 
        # We'll just return the number of rows, no label conversion necessary here.
        data_dir = os.path.join('data', 'mushroom')
        data_file = os.path.join(data_dir, 'mushroom.csv')
        os.makedirs(data_dir, exist_ok=True)

        if not os.path.exists(data_file):
            print("Downloading 'mushroom' dataset...")
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
            urllib.request.urlretrieve(url, data_file)
            print("Download complete.")

        # Read the CSV file. No headers; 23 columns in total.
        df = pd.read_csv(data_file, header=None)
        # Return number of rows
        return len(df)

    
    # --------------------------------------------------------------------------
    # 3. website+phishing (UCI #379)
    # --------------------------------------------------------------------------
    elif data_name.lower() == 'website+phishing':
        data_dir = os.path.join('data', 'website+phishing')
        os.makedirs(data_dir, exist_ok=True)

        zip_url = 'https://archive.ics.uci.edu/static/public/379/website+phishing.zip'
        zip_file = os.path.join(data_dir, 'website_phishing.zip')
        arff_filename = 'PhishingData.arff'
        arff_path = os.path.join(data_dir, arff_filename)

        if not os.path.exists(zip_file):
            print("Downloading 'website+phishing' dataset (website+phishing.zip)...")
            urllib.request.urlretrieve(zip_url, zip_file)
            print("Download complete.")

        if not os.path.exists(arff_path):
            print("Extracting 'website+phishing.zip'...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extraction complete.")

        data, meta = arff.loadarff(arff_path)
        df = pd.DataFrame(data)
        return len(df)

    # --------------------------------------------------------------------------
    # 4. phishing+websites (UCI #327)
    # --------------------------------------------------------------------------
    elif data_name.lower() == 'phishing+websites':
        data_dir = os.path.join('data', 'phishing+websites')
        os.makedirs(data_dir, exist_ok=True)

        zip_url = 'https://archive.ics.uci.edu/static/public/327/phishing+websites.zip'
        zip_file = os.path.join(data_dir, 'phishing_websites.zip')
        # The ARFF file inside the ZIP is 'Training Dataset.arff'
        arff_filename = 'Training Dataset.arff'
        arff_path = os.path.join(data_dir, arff_filename)

        if not os.path.exists(zip_file):
            print("Downloading 'phishing+websites' dataset (phishing+websites.zip)...")
            urllib.request.urlretrieve(zip_url, zip_file)
            print("Download complete.")

        if not os.path.exists(arff_path):
            print("Extracting 'phishing+websites.zip'...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extraction complete.")

        data, meta = arff.loadarff(arff_path)
        df = pd.DataFrame(data)
        return len(df)

    elif data_name.lower() == 'spambase':
        data_dir = os.path.join('data', 'spambase')
        data_file = os.path.join(data_dir, 'spambase.data')
        os.makedirs(data_dir, exist_ok=True)

        # Download if file not found
        if not os.path.exists(data_file):
            print("Downloading 'spambase' dataset...")
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
            urllib.request.urlretrieve(url, data_file)
            print("Download complete.")

        # Read the CSV file (no header)
        df = pd.read_csv(data_file, header=None)
        # Return the number of rows (data points)
        return len(df)

    elif data_name.lower() == 'bank_marketing':
        data_dir = os.path.join('data', 'bank_marketing')
        os.makedirs(data_dir, exist_ok=True)

        data_file = os.path.join(data_dir, 'bank-additional-full.csv')

        if not os.path.exists(data_file):
            print("Downloading 'bank_marketing' dataset (bank-additional-full.csv)...")
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv'
            urllib.request.urlretrieve(url, data_file)
            print("Download complete.")

        # Read CSV with semicolon delimiter
        df = pd.read_csv(data_file, sep=';')

        # Just return the number of rows
        return len(df)

    # --------------------------------------------------------------------------
    # Unsupported dataset
    # --------------------------------------------------------------------------
    else:
        raise ValueError(
            f"Unsupported dataset name '{data_name}'. "
        )