"""
Preprocessing Module
====================
Handles feature engineering, scaling, and encoding using sklearn pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple
import joblib


class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to apply log transformation to skewed features."""
    
    def __init__(self, features: List[str] = None):
        """
        Initialize LogTransformer.
        
        Args:
            features (List[str]): Column names to apply log transformation to.
                                  Default: ['capital-gain', 'capital-loss']
        """
        self.features = features or ['capital-gain', 'capital-loss']
    
    def fit(self, X, y=None):
        """Fit transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """
        Apply log(x+1) transformation to specified features.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with log-transformed columns
        """
        X_copy = X.copy()
        for feature in self.features:
            if feature in X_copy.columns:
                X_copy[feature] = np.log(X_copy[feature] + 1)
        return X_copy


class FeaturePreprocessor:
    """Orchestrates feature preprocessing including log transformation, scaling, and encoding."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = MinMaxScaler()
        self.encoder = None
        self.numerical_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.categorical_features = None
        self.log_features = ['capital-gain', 'capital-loss']
        self.feature_names_encoded = None
        self.pipeline = None
    
    def fit(self, X_train: pd.DataFrame) -> 'FeaturePreprocessor':
        """
        Fit the preprocessing pipeline on training data.
        
        Args:
            X_train (pd.DataFrame): Training features
            
        Returns:
            self: Fitted preprocessor instance
        """
        # Step 1: Log-transform skewed features
        X_log = X_train.copy()
        for feature in self.log_features:
            if feature in X_log.columns:
                X_log[feature] = np.log(X_log[feature] + 1)
        
        # Step 2: Scale numerical features
        X_scaled = X_log.copy()
        X_scaled[self.numerical_features] = self.scaler.fit_transform(X_log[self.numerical_features])
        
        # Step 3: One-hot encode categorical features
        # Identify categorical columns (non-numerical)
        self.categorical_features = X_scaled.select_dtypes(include='object').columns.tolist()
        
        # One-hot encode and store the encoder
        X_encoded = pd.get_dummies(X_scaled, columns=self.categorical_features, drop_first=False)
        self.feature_names_encoded = X_encoded.columns.tolist()
        
        print(f"Preprocessor fitted:")
        print(f"  - Log-transformed {len(self.log_features)} skewed features")
        print(f"  - Scaled {len(self.numerical_features)} numerical features")
        print(f"  - One-hot encoded {len(self.categorical_features)} categorical features")
        print(f"  - Total features after encoding: {len(self.feature_names_encoded)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted preprocessing pipeline.
        
        Args:
            X (pd.DataFrame): Features to transform
            
        Returns:
            pd.DataFrame: Preprocessed features
            
        Raises:
            ValueError: If preprocessor hasn't been fitted yet
        """
        if self.scaler is None or self.feature_names_encoded is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Step 1: Log-transform skewed features
        X_log = X.copy()
        for feature in self.log_features:
            if feature in X_log.columns:
                X_log[feature] = np.log(X_log[feature] + 1)
        
        # Step 2: Scale numerical features
        X_scaled = X_log.copy()
        X_scaled[self.numerical_features] = self.scaler.transform(X_log[self.numerical_features])
        
        # Step 3: One-hot encode using the categorical features identified during fit
        X_encoded = pd.get_dummies(X_scaled, columns=self.categorical_features, drop_first=False, dtype=int)
        
        # Align columns with training data (in case test set is missing some categories)
        for col in self.feature_names_encoded:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        
        # Remove extra columns that weren't in training data
        X_encoded = X_encoded[[col for col in X_encoded.columns if col in self.feature_names_encoded]]
        
        # Ensure column order matches training data
        X_encoded = X_encoded[self.feature_names_encoded]
        
        return X_encoded
    
    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X_train (pd.DataFrame): Training features
            
        Returns:
            pd.DataFrame: Preprocessed training features
        """
        self.fit(X_train)
        return self.transform(X_train)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of encoded features.
        
        Returns:
            List[str]: List of feature names after encoding
        """
        if self.feature_names_encoded is None:
            raise ValueError("No encoded features available. Call fit() first.")
        return self.feature_names_encoded
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk using joblib.
        
        Args:
            filepath (str): Path where to save the preprocessor
        """
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FeaturePreprocessor':
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath (str): Path to load the preprocessor from
            
        Returns:
            FeaturePreprocessor: Loaded preprocessor instance
        """
        preprocessor = joblib.load(filepath)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor
