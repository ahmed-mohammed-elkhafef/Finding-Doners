"""
Model Trainer Module
====================
Handles model training, evaluation, and serialization.
"""

import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix, classification_report
import joblib
from typing import Dict, Any, Tuple


class ModelTrainer:
    """Train and evaluate classification models."""
    
    def __init__(self, model=None, model_name: str = "RandomForestClassifier"):
        """
        Initialize ModelTrainer.
        
        Args:
            model: Sklearn model instance. If None, uses RandomForestClassifier.
            model_name (str): Name of the model for logging purposes.
        """
        self.model = model or RandomForestClassifier(random_state=42, n_jobs=-1)
        self.model_name = model_name
        self.is_trained = False
        self.training_time = None
        self.prediction_time = None
        self.metrics = {}
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model on training data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
        """
        print(f"Training {self.model_name}...")
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        self.is_trained = True
        print(f"Training completed in {self.training_time:.4f} seconds")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Predicted labels (0 or 1)
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = time.time() - start_time
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Predicted probabilities for each class
            
        Raises:
            ValueError: If model hasn't been trained yet or doesn't support probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_name} doesn't support predict_proba().")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, beta: float = 0.5) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            beta (float): Beta parameter for F-score (default: 0.5)
            
        Returns:
            Dict[str, float]: Dictionary of metrics (accuracy, f-score, etc.)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f_score = fbeta_score(y_test, y_pred, beta=beta)
        
        # Additional metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.metrics = {
            'accuracy': accuracy,
            'f_score': f_score,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'prediction_time': self.prediction_time
        }
        
        return self.metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the last computed metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics
        """
        if not self.metrics:
            raise ValueError("No metrics available. Call evaluate() first.")
        return self.metrics
    
    def get_feature_importances(self) -> pd.Series:
        """
        Get feature importances (if the model supports it).
        
        Returns:
            pd.Series: Feature importances indexed by feature name
            
        Raises:
            ValueError: If model doesn't support feature_importances_
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"{self.model_name} doesn't have feature_importances_ attribute.")
        
        return pd.Series(self.model.feature_importances_)
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'ModelTrainer':
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to load the model from
            
        Returns:
            ModelTrainer: ModelTrainer instance with loaded model
        """
        model = joblib.load(filepath)
        trainer = ModelTrainer(model=model)
        trainer.is_trained = True
        print(f"Model loaded from {filepath}")
        return trainer
    
    def print_evaluation_report(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Print a detailed evaluation report.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        """
        y_pred = self.predict(X_test)
        
        print("\n" + "="*60)
        print(f"EVALUATION REPORT: {self.model_name}")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
        
        metrics = self.evaluate(X_test, y_test)
        print("\nKey Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F-Score (β=0.5): {metrics['f_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Prediction Time: {metrics['prediction_time']:.6f} seconds")
        print("="*60 + "\n")
