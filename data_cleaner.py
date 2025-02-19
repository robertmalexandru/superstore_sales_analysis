import pandas as pd
import numpy as np
import time
import os
import random
from pathlib import Path

def random_delay(min_sec=1, max_sec=1):
    """Generates a random delay between min_sec and max_sec."""
    delay = random.randint(min_sec, max_sec)
    print(f'Please wait {delay} second...')
    time.sleep(delay)

def load_dataset(data_path):
    """Loads a dataset based on its file type."""
    file_extension = Path(data_path).suffix.lower()
    if file_extension == '.csv':
        print('Dataset is CSV.')
        return pd.read_csv(data_path, encoding_errors='ignore')
    elif file_extension in ('.xlsx', '.xls'):
        print('Dataset is Excel.')
        return pd.read_excel(data_path)
    elif file_extension == '.json':
        print('Dataset is JSON.')
        return pd.read_json(data_path)
    elif file_extension == '.parquet':
        print('Dataset is Parquet.')
        return pd.read_parquet(data_path)
    else:
        raise ValueError('Unsupported file type. Please provide a CSV, Excel, JSON, or Parquet file.')

def handle_duplicates(data, data_name):
    """Handles duplicate records in the dataset."""
    total_duplicates = data.duplicated().sum()
    print(f'Dataset has {total_duplicates} duplicate records.')

    if total_duplicates > 0:
        print('Saving duplicates...')
        duplicate_records = data[data.duplicated()]
        duplicate_records.to_csv(f'{data_name}_duplicates.csv', index=False)
        print('Removing duplicates...')
        data = data.drop_duplicates()
    return data

def handle_missing_values(data):
    """Handles missing values in the dataset."""
    total_missing_values = data.isnull().sum().sum()
    missing_value_columns = data.isnull().sum()

    print(f'Dataset has {total_missing_values} missing values.')
    print('Missing values per column:')
    print(missing_value_columns)

    for col in data.columns:
        if data[col].dtype in (float, int):
            data[col] = data[col].fillna(data[col].mean())  # Fill numeric columns with mean
        else:
            data.dropna(subset=[col], inplace=True)  # Drop rows with missing non-numeric values
    return data

def drop_unnecessary_columns(data):
    """Drops columns with more than 50% missing values."""
    columns_to_drop = [col for col in data.columns if data[col].isnull().mean() > 0.5]
    if columns_to_drop:
        print(f'Dropping columns with >50% missing values: {columns_to_drop}')
        data.drop(columns=columns_to_drop, inplace=True)
    return data

def standardize_columns(data):
    """Standardizes column names by removing spaces and special characters."""
    data.columns = [col.strip().replace(' ', '_').lower() for col in data.columns]
    return data

def handle_outliers(data):
    """Handles outliers in numeric columns by capping them to IQR bounds."""
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = np.where((data[col] < lower_bound) | (data[col] > upper_bound), np.nan, data[col])
    return data

def data_cleaner(data_path, data_name):
    """Main function to clean the dataset."""
    print('Thank you for the dataset.')

    # Validate file path
    if not os.path.exists(data_path):
        print('Incorrect path. Please enter a valid path.')
        return

    # Load dataset
    random_delay()
    try:
        data = load_dataset(data_path)
    except Exception as e:
        print(f'Error loading the dataset: {e}')
        return

    print(f'Dataset contains {data.shape[0]} rows and {data.shape[1]} columns.')

    # Handle duplicates
    data = handle_duplicates(data, data_name)

    # Handle missing values
    data = handle_missing_values(data)

    # Drop unnecessary columns
    data = drop_unnecessary_columns(data)

    # Standardize column names
    data = standardize_columns(data)

    # Handle outliers
    data = handle_outliers(data)

    # Save cleaned dataset
    random_delay()
    cleaned_file_path = f'{data_name}_clean_data.csv'
    data.to_csv(cleaned_file_path, index=False)
    print(f'Clean dataset saved to {cleaned_file_path}.')

if __name__ == '__main__':
    print('Welcome to the Data Cleaner!')

    data_path = input('Please enter the dataset path: ')
    data_name = input('Please enter a name for the dataset (without extension): ')
    data_cleaner(data_path, data_name)