"""
Data Ingestion Module
=====================
Handles loading raw census data and initial train/test splits.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


class DataIngestion:
    """Load and split census data for machine learning."""

    def __init__(self, raw_data_path: str, test_size: float = 0.2, random_state: int = 0):
        """
        Initialize DataIngestion.
        
        Args:
            raw_data_path (str): Path to the raw CSV file (e.g., 'data/raw/census.csv')
            test_size (float): Proportion of data to use for testing (default: 0.2)
            random_state (int): Random seed for reproducibility (default: 0)
        """
        self.raw_data_path = raw_data_path
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self) -> pd.DataFrame:
        """
        Load raw census data from CSV.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Data file not found: {self.raw_data_path}")
        
        self.data = pd.read_csv(self.raw_data_path)
        print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data

    def extract_target(self, target_col: str = 'income') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and target variable from the dataset.
        
        Args:
            target_col (str): Name of the target column (default: 'income')
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target series
            
        Raises:
            ValueError: If target column doesn't exist or data not loaded
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")
        
        features = self.data.drop(target_col, axis=1)
        target = self.data[target_col]
        
        print(f"Target extracted: {target.name}")
        return features, target

    def split_data(self, features: pd.DataFrame, target: pd.Series) -> None:
        """
        Split features and target into train and test sets.
        
        Args:
            features (pd.DataFrame): Feature matrix
            target (pd.Series): Target variable
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, 
            target,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        print(f"Data split:")
        print(f"  - Training set: {self.X_train.shape[0]} samples")
        print(f"  - Testing set: {self.X_test.shape[0]} samples")

    def get_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get the split train/test data.
        
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
            
        Raises:
            ValueError: If split hasn't been performed yet
        """
        if any(x is None for x in [self.X_train, self.X_test, self.y_train, self.y_test]):
            raise ValueError("Data not split yet. Call split_data() first.")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_data_info(self) -> dict:
        """
        Get summary information about the loaded data.
        
        Returns:
            dict: Contains shape, columns, dtypes, missing values, etc.
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 ** 2
        }
