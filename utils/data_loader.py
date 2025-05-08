import pandas as pd

def load_csv(dataset_path):
    if dataset_path.endswith('.csv'):
        return pd.read_csv(dataset_path)
    else:
        raise ValueError("Only CSV Files allowed in this version")