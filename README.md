# Income Prediction ML Dashboard - Group 2

**Project Name:** Income Prediction Machine Learning System  
**Group:** Group 2  
**Supervisor:** George Samuel

---

## 📋 Project Overview

This project implements a **production-ready machine learning pipeline** for income classification using a Random Forest classifier. The system predicts whether a person's income exceeds $50K based on demographic and employment features.

The project includes:
- **Modular ML Pipeline** - Data ingestion, preprocessing, model training
- **Interactive Plotly Dash Dashboard** - Real-time income predictions
- **Serialized Artifacts** - Trained model and preprocessor for deployment
- **Professional Documentation** - Complete setup and usage guides

---

## 🎯 Problem Statement

Given demographic information about individuals (age, education, occupation, etc.), predict whether their income exceeds $50,000 per year.

**Dataset:** Census Income Dataset (14 features, 32,561 samples)  
**Target Variable:** Income (<=50K or >50K)  
**Model:** Random Forest Classifier  
**Evaluation Metric:** F-Score (β=0.5)

---

## 📁 Project Structure

```
finding_donors/
├── data/
│   ├── raw/
│   │   └── census.csv                 # Original dataset
│   └── artifacts/
│       ├── preprocessor.pkl           # Fitted scaler & encoder
│       └── feature_names.txt           # One-hot encoded feature names
│
├── models/
│   └── model.pkl                      # Trained RandomForestClassifier
│
├── src/                               # Modular source code
│   ├── __init__.py
│   ├── data_ingestion.py             # DataIngestion class
│   ├── preprocessing.py              # FeaturePreprocessor class
│   └── model_trainer.py              # ModelTrainer class
│
├── app.py                            # Plotly Dash dashboard
├── train_pipeline.py                 # ML pipeline orchestration
├── requirements.txt                  # Dependencies
├── README.md                         # This file
└── ... (documentation files)
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

### Step 1: Clone/Setup Project
```bash
cd finding_donors
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option A: Train from Scratch

**Step 1:** Ensure `data/raw/census.csv` exists

**Step 2:** Run training pipeline
```bash
python train_pipeline.py
```

Expected output:
```
[STEP 1] DATA INGESTION
[STEP 2] FEATURE PREPROCESSING
[STEP 3] MODEL TRAINING
[STEP 4] MODEL EVALUATION
[STEP 5] SAVING ARTIFACTS
PIPELINE COMPLETED SUCCESSFULLY
```

Artifacts generated:
- `models/model.pkl` (~50 MB)
- `data/artifacts/preprocessor.pkl`
- `data/artifacts/feature_names.txt`

### Option B: Use Pre-trained Model

If model artifacts already exist, skip training and go directly to Step 3.

**Step 3:** Launch Dashboard
```bash
python app.py
```

Dashboard URL: **http://127.0.0.1:8050/**

---

## 📊 Dashboard Features

### Input Section
- **Numerical Inputs (Sliders):**
  - Age: 17-90 years
  - Education Years: 1-16 years
  - Capital Gain: 0-99,999
  - Capital Loss: 0-4,356
  - Hours/Week: 1-100 hours

- **Categorical Inputs (Dropdowns):**
  - Workclass: Private, Self-employed, Government, etc.
  - Education Level: Preschool to Doctorate
  - Marital Status: Married, Single, Divorced, etc.
  - Occupation: 12+ categories
  - Relationship: Husband, Wife, Own-child, etc.
  - Race: White, Black, Asian, etc.
  - Sex: Male, Female
  - Native Country: 40+ countries

### Output Section
- **Predicted Income Class:** ">50K" or "<=50K"
- **Confidence Score:** 0-100%
- **Probability Distribution:** Visual bars for both classes
- **Input Summary:** Echo of all entered values

---

## 🔧 Technical Details

### Feature Engineering Pipeline

1. **Log Transformation** (Skewed Features)
   - Applied to: `capital-gain`, `capital-loss`
   - Formula: `log(x + 1)`
   - Reason: Handle extreme values and right skewness

2. **MinMax Scaling** (Numerical Features)
   - Applied to: `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
   - Range: [0, 1]
   - Reason: Fair weight distribution in tree splits

3. **One-Hot Encoding** (Categorical Features)
   - Applied to: 8 categorical features
   - Result: ~103 binary features after encoding
   - Reason: RandomForest requirement for categorical data

4. **Target Encoding**
   - `<=50K` → 0
   - `>50K` → 1

### Model Architecture

**Algorithm:** Random Forest Classifier
- **n_estimators:** 100 trees
- **random_state:** 0 (for reproducibility)
- **max_depth:** No limit
- **min_samples_split:** 2
- **Training/Test Split:** 80/20

### Model Performance

**Top 5 Most Important Features:**
1. Age - 23.0%
2. Hours-per-week - 11.4%
3. Capital-gain - 10.4%
4. Marital Status (Married) - 8.0%
5. Education-num - 6.6%

**Evaluation Metrics:**
- Accuracy: [See train_pipeline.py output]
- F-Score (β=0.5): [See train_pipeline.py output]
- Precision/Recall: [See train_pipeline.py output]

---

## 📦 Module Documentation

### `src/data_ingestion.py`

**Class:** `DataIngestion`

**Methods:**
- `load_data()` - Load CSV from `data/raw/census.csv`
- `extract_target(target_col)` - Separate features and target
- `split_data(features, target)` - 80/20 train/test split
- `get_split_data()` - Retrieve split datasets
- `get_data_info()` - Return summary statistics

**Example:**
```python
from src.data_ingestion import DataIngestion

ingestion = DataIngestion("data/raw/census.csv")
ingestion.load_data()
X, y = ingestion.extract_target('income')
ingestion.split_data(X, y)
X_train, X_test, y_train, y_test = ingestion.get_split_data()
```

---

### `src/preprocessing.py`

**Classes:** 
- `LogTransformer` - Custom sklearn transformer
- `FeaturePreprocessor` - Main orchestrator

**Methods:**
- `fit(X_train)` - Learn from training data
- `transform(X)` - Apply transformations
- `fit_transform(X_train)` - Fit and transform in one step
- `get_feature_names()` - Get encoded feature names
- `save(filepath)` - Serialize with joblib
- `load(filepath)` - Deserialize from disk

**Example:**
```python
from src.preprocessing import FeaturePreprocessor

preprocessor = FeaturePreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
preprocessor.save("data/artifacts/preprocessor.pkl")
```

---

### `src/model_trainer.py`

**Class:** `ModelTrainer`

**Methods:**
- `train(X_train, y_train)` - Fit the model
- `predict(X)` - Make predictions
- `predict_proba(X)` - Get class probabilities
- `evaluate(X_test, y_test, beta)` - Compute metrics
- `get_metrics()` - Retrieve evaluation results
- `get_feature_importances()` - Extract importance scores
- `save(filepath)` - Serialize model
- `load(filepath)` - Deserialize model
- `print_evaluation_report(X_test, y_test)` - Print detailed report

**Example:**
```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train(X_train_processed, y_train_encoded)
metrics = trainer.evaluate(X_test_processed, y_test_encoded)
trainer.save("models/model.pkl")
```

---

### `train_pipeline.py`

**Purpose:** Orchestrates complete workflow

**Usage:**
```bash
python train_pipeline.py
```

**Flow:**
1. Load data from `data/raw/census.csv`
2. Extract features and target
3. Preprocess training data
4. Train RandomForestClassifier
5. Evaluate on test set
6. Save artifacts to `models/` and `data/artifacts/`

---

### `app.py`

**Purpose:** Interactive Plotly Dash dashboard

**Key Functions:**
- `load_artifacts()` - Load model, preprocessor, features
- `make_prediction()` - Callback for real-time predictions

**Usage:**
```bash
python app.py
```

**Access:** http://127.0.0.1:8050/

---

## ⚙️ Configuration

### Default Paths
```python
# Training Pipeline
raw_data_path = "data/raw/census.csv"
artifact_dir = "data/artifacts"
model_dir = "models"

# Dashboard
MODEL_DIR = Path(__file__).parent / "models"
ARTIFACT_DIR = Path(__file__).parent / "data" / "artifacts"
```

### Model Hyperparameters
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=0,
    n_jobs=-1  # Use all CPUs
)
```

---


---

## 📈 Expected Results

### Training Output Example
```
Dataset Info:
  - Shape: (32561, 13)
  - Memory Usage: 3.35 MB
  - Columns: 13

Data split:
  - Training set: 26048 samples
  - Testing set: 6513 samples

Preprocessor fitted:
  - Log-transformed 2 skewed features
  - Scaled 5 numerical features
  - One-hot encoded 8 categorical features
  - Total features after encoding: 103

Training RandomForestClassifier...
Training completed in 12.3456 seconds

Model Evaluation:
  - Accuracy: 0.8523
  - Precision: 0.7234
  - Recall: 0.6145
  - F-Score (β=0.5): 0.6890

Artifacts saved to: [paths shown]
```

---

## 🔄 Workflow Summary

```
┌─────────────────┐
│  census.csv     │
└────────┬────────┘
         │
    ┌────▼─────────────┐
    │ DataIngestion    │
    │ - Load data      │
    │ - Extract target │
    │ - Split 80/20    │
    └────┬─────────────┘
         │
    ┌────▼───────────────────┐
    │ FeaturePreprocessor    │
    │ - Log transform        │
    │ - MinMax scale         │
    │ - One-hot encode       │
    └────┬───────────────────┘
         │
    ┌────▼─────────────────┐
    │ ModelTrainer         │
    │ - Train RF           │
    │ - Evaluate metrics   │
    │ - Save artifacts     │
    └────┬─────────────────┘
         │
    ┌────▼──────────────────┐
    │ models/model.pkl      │
    │ data/artifacts/*.pkl  │
    └───────────────────────┘
         │
    ┌────▼──────────────┐
    │ Dash App          │
    │ - Load model      │
    │ - Real-time pred  │
    │ - Interactive UI  │
    └───────────────────┘
```

---

## 📚 Additional Resources

- **Original Notebook:** `finding_donors.ipynb` (reference only)
- **Feature Importance:** See `train_pipeline.py` output
- **API Docs:** See docstrings in each module

---

## 👥 Team Information

**Project Name:** Income Prediction ML System  
**Group:** Group 2  
**Supervisor:** George Samuel  
**Date:** April 2026

---


