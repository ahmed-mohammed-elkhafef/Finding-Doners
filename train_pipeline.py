"""
Main Training Pipeline
======================
Orchestrates the entire ML pipeline: data ingestion, preprocessing, model training, and artifact saving.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_ingestion import DataIngestion
from preprocessing import FeaturePreprocessor
from model_trainer import ModelTrainer


def create_directories(artifact_dir: str) -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(artifact_dir, exist_ok=True)
    print(f"Artifact directory ready: {artifact_dir}")


def main(raw_data_path: str = "data/raw/census.csv", artifact_dir: str = "data/artifacts/", model_dir: str = "models/") -> None:
    """
    Execute the complete ML pipeline.
    
    Args:
        raw_data_path (str): Path to raw census data
        artifact_dir (str): Directory to save preprocessor and feature names
        model_dir (str): Directory to save trained model
    """
    print("\n" + "="*70)
    print("INCOME PREDICTION ML PIPELINE")
    print("="*70 + "\n")
    
    # Create artifact and model directories
    create_directories(artifact_dir)
    create_directories(model_dir)
    
    # =========== STEP 1: DATA INGESTION ===========
    print("\n[STEP 1] DATA INGESTION")
    print("-" * 70)
    
    ingestion = DataIngestion(
        raw_data_path=raw_data_path,
        test_size=0.2,
        random_state=0
    )
    
    try:
        ingestion.load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"   Expected file at: {os.path.abspath(raw_data_path)}")
        return
    
    # Extract features and target
    X, y = ingestion.extract_target(target_col='income')
    
    # Print data info
    data_info = ingestion.get_data_info()
    print(f"\nDataset Info:")
    print(f"  - Shape: {data_info['shape']}")
    print(f"  - Memory Usage: {data_info['memory_usage_mb']:.2f} MB")
    print(f"  - Columns: {len(data_info['columns'])}")
    
    # Split data
    ingestion.split_data(X, y)
    X_train, X_test, y_train, y_test = ingestion.get_split_data()
    
    # =========== STEP 2: PREPROCESSING ===========
    print("\n[STEP 2] FEATURE PREPROCESSING")
    print("-" * 70)
    
    preprocessor = FeaturePreprocessor()
    
    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    print(f"\n  Processed training shape: {X_train_processed.shape}")
    
    # Transform test data using fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)
    print(f"  Processed test shape: {X_test_processed.shape}")
    
    # Encode target variable
    y_train_encoded = y_train.apply(lambda x: 1 if x == ">50K" else 0)
    y_test_encoded = y_test.apply(lambda x: 1 if x == ">50K" else 0)
    print(f"\n  Target classes: 0 (<=50K), 1 (>50K)")
    print(f"  Training target distribution: {y_train_encoded.value_counts().to_dict()}")
    
    # =========== STEP 3: MODEL TRAINING ===========
    print("\n[STEP 3] MODEL TRAINING")
    print("-" * 70)
    
    trainer = ModelTrainer(model_name="RandomForestClassifier")
    trainer.train(X_train_processed, y_train_encoded)
    
    # =========== STEP 4: MODEL EVALUATION ===========
    print("\n[STEP 4] MODEL EVALUATION")
    print("-" * 70)
    
    trainer.print_evaluation_report(X_test_processed, y_test_encoded)
    
    # =========== STEP 5: SAVE ARTIFACTS ===========
    print("\n[STEP 5] SAVING ARTIFACTS")
    print("-" * 70)
    
    model_path = os.path.join(model_dir, "model.pkl")
    preprocessor_path = os.path.join(artifact_dir, "preprocessor.pkl")
    feature_names_path = os.path.join(artifact_dir, "feature_names.txt")
    
    trainer.save(model_path)
    preprocessor.save(preprocessor_path)
    
    # Save feature names
    feature_names = preprocessor.get_feature_names()
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f"Feature names saved to {feature_names_path}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nArtifacts saved to: {os.path.abspath(artifact_dir)}")
    print("  - model.pkl (trained RandomForestClassifier)")
    print("  - preprocessor.pkl (fitted FeaturePreprocessor)")
    print("  - feature_names.txt (encoded feature names)")
    print("\n")


if __name__ == "__main__":
    # Adjust paths if running from different directories
    raw_data_path = "data/raw/census.csv"
    artifact_dir = "data/artifacts"
    model_dir = "models"
    
    # Check if running from project root; if not, adjust paths
    if not os.path.exists(raw_data_path):
        # Try running from script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_path = os.path.join(script_dir, "data", "raw", "census.csv")
        artifact_dir = os.path.join(script_dir, "data", "artifacts")
        model_dir = os.path.join(script_dir, "models")
    
    main(raw_data_path=raw_data_path, artifact_dir=artifact_dir, model_dir=model_dir)
