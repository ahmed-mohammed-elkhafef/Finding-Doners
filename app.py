"""
Plotly Dash Application
=======================
Interactive ML prediction dashboard for income classification.
"""

import os
import sys
import json
import pandas as pd
import joblib
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
from preprocessing import FeaturePreprocessor


# ==================== LOAD ARTIFACTS ====================
def load_artifacts():
    """Load trained model, preprocessor, and feature names."""
    try:
        model_path = MODEL_DIR / "model.pkl"
        preprocessor_path = ARTIFACT_DIR / "preprocessor.pkl"
        feature_names_path = ARTIFACT_DIR / "feature_names.txt"
        
        if not all([model_path.exists(), preprocessor_path.exists(), feature_names_path.exists()]):
            raise FileNotFoundError(
                f"Missing artifacts. Please run train_pipeline.py first.\n"
                f"Expected model location: {MODEL_DIR}\n"
                f"Expected artifacts location: {ARTIFACT_DIR}"
            )
        
        model = joblib.load(str(model_path))
        preprocessor = joblib.load(str(preprocessor_path))
        
        with open(str(feature_names_path), 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        return model, preprocessor, feature_names
    
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        return None, None, None


# Load artifacts
MODEL, PREPROCESSOR, FEATURE_NAMES = load_artifacts()

if MODEL is None:
    print("Warning: Artifacts not loaded. Dashboard may not function properly.")


# ==================== SAMPLE DATA DISTRIBUTIONS ====================
# These should be loaded from training data in production
CATEGORICAL_FEATURES = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                  'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    'education': ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', 
                  '12th', 'HS-grad', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 
                  'Bachelors', 'Masters', 'Prof-school', 'Doctorate'],
    'marital-status': ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse', 
                       'Never-married', 'Divorced', 'Separated', 'Widowed'],
    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 
                   'Prof-specialty', 'Protective-serv', 'Machine-op-inspct', 'Transport-moving', 
                   'Handlers-cleaners', 'Farming-fishing', 'Armed-Forces'],
    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    'sex': ['Female', 'Male'],
    'native-country': ['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'Canada', 'Philippines', 
                       'Vietnam', 'Germany', 'England', 'Puerto-Rico', 'Hong', 'Ireland', 'France', 
                       'Scotland', 'China', 'Taiwan', 'Japan', 'Hungary', 'Iran', 'Greece', 'Italy', 
                       'Poland', 'Portugal', 'Holand-Netherlands', 'Cambodia', 'Thailand', 'Guatemala', 
                       'Nicaragua', 'El-Salvador', 'Dominican-Republic', 'Trinadad&Tobago', 'Guatemala', 
                       'South', 'Ecuador', 'Haiti', 'Columbia', 'Outlying-US(Guam-USVI-etc)']
}


# ==================== DASH APPLICATION ====================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Income Prediction Dashboard"


# App Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Income Prediction Dashboard", className="text-center mb-4 mt-4", 
                   style={'color': '#1f77b4', 'fontWeight': 'bold'})
        ])
    ]),
    
    html.Hr(),
    
    dbc.Row([
        # Left Column: Input Form
        dbc.Col([
            html.H4("Enter Individual Information", className="mb-3", style={'color': '#333'}),
            
            html.Div(id='form-errors', className="alert alert-danger d-none", role="alert"),
            
            # Row 1: Age and Education Years
            dbc.Row([
                dbc.Col([
                    html.Label("Age", className="fw-bold"),
                    dcc.Slider(
                        id='age-slider',
                        min=17,
                        max=90,
                        step=1,
                        value=40,
                        marks={i: str(i) if i % 10 == 0 else '' for i in range(17, 91, 1)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], width=12, lg=6, className="mb-4"),
                
                dbc.Col([
                    html.Label("Years of Education", className="fw-bold"),
                    dcc.Slider(
                        id='education-num-slider',
                        min=1,
                        max=16,
                        step=1,
                        value=10,
                        marks={i: str(i) if i % 2 == 0 else '' for i in range(1, 17)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], width=12, lg=6, className="mb-4"),
            ]),
            
            # Row 2: Capital Gain and Capital Loss
            dbc.Row([
                dbc.Col([
                    html.Label("Capital Gain ($)", className="fw-bold"),
                    dcc.Slider(
                        id='capital-gain-slider',
                        min=0,
                        max=99999,
                        step=1000,
                        value=0,
                        marks={i: f'${i//1000}k' if i % 15000 == 0 else '' for i in range(0, 100000, 1000)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], width=12, lg=6, className="mb-4"),
                
                dbc.Col([
                    html.Label("Capital Loss ($)", className="fw-bold"),
                    dcc.Slider(
                        id='capital-loss-slider',
                        min=0,
                        max=4356,
                        step=100,
                        value=0,
                        marks={i: f'${i//1000}k' if i % 1000 == 0 else '' for i in range(0, 4500, 100)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], width=12, lg=6, className="mb-4"),
            ]),
            
            # Row 3: Hours per Week
            dbc.Row([
                dbc.Col([
                    html.Label("Hours per Week", className="fw-bold"),
                    dcc.Slider(
                        id='hours-slider',
                        min=1,
                        max=100,
                        step=1,
                        value=40,
                        marks={i: str(i) if i % 10 == 0 else '' for i in range(1, 101, 1)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], width=12, className="mb-4"),
            ]),
            
            # Row 4: Categorical Features
            dbc.Row([
                dbc.Col([
                    html.Label("Work Class", className="fw-bold"),
                    dcc.Dropdown(
                        id='workclass-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in CATEGORICAL_FEATURES['workclass']],
                        value='Private',
                        clearable=False
                    ),
                ], width=12, lg=6, className="mb-3"),
                
                dbc.Col([
                    html.Label("Education Level", className="fw-bold"),
                    dcc.Dropdown(
                        id='education-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in CATEGORICAL_FEATURES['education']],
                        value='HS-grad',
                        clearable=False
                    ),
                ], width=12, lg=6, className="mb-3"),
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Marital Status", className="fw-bold"),
                    dcc.Dropdown(
                        id='marital-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in CATEGORICAL_FEATURES['marital-status']],
                        value='Never-married',
                        clearable=False
                    ),
                ], width=12, lg=6, className="mb-3"),
                
                dbc.Col([
                    html.Label("Occupation", className="fw-bold"),
                    dcc.Dropdown(
                        id='occupation-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in CATEGORICAL_FEATURES['occupation']],
                        value='Tech-support',
                        clearable=False
                    ),
                ], width=12, lg=6, className="mb-3"),
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Relationship", className="fw-bold"),
                    dcc.Dropdown(
                        id='relationship-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in CATEGORICAL_FEATURES['relationship']],
                        value='Not-in-family',
                        clearable=False
                    ),
                ], width=12, lg=6, className="mb-3"),
                
                dbc.Col([
                    html.Label("Race", className="fw-bold"),
                    dcc.Dropdown(
                        id='race-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in CATEGORICAL_FEATURES['race']],
                        value='White',
                        clearable=False
                    ),
                ], width=12, lg=6, className="mb-3"),
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Sex", className="fw-bold"),
                    dcc.Dropdown(
                        id='sex-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in CATEGORICAL_FEATURES['sex']],
                        value='Male',
                        clearable=False
                    ),
                ], width=12, lg=6, className="mb-3"),
                
                dbc.Col([
                    html.Label("Native Country", className="fw-bold"),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in CATEGORICAL_FEATURES['native-country']],
                        value='United-States',
                        clearable=False
                    ),
                ], width=12, lg=6, className="mb-3"),
            ]),
            
            # Predict Button
            dbc.Row([
                dbc.Col([
                    html.Button(
                        "Predict Income",
                        id='predict-button',
                        n_clicks=0,
                        className="btn btn-primary btn-lg w-100 mt-3",
                        style={'backgroundColor': '#1f77b4', 'borderColor': '#1f77b4'}
                    ),
                ], width=12),
            ]),
            
        ], width=12, lg=5, className="px-4"),
        
        # Right Column: Prediction Output
        dbc.Col([
            html.H4("Prediction Result", className="mb-4", style={'color': '#333'}),
            
            html.Div(id='prediction-output', children=[
                dbc.Card([
                    dbc.CardBody([
                        html.P("Click 'Predict Income' to see results", className="text-muted text-center")
                    ])
                ])
            ], className="mb-4"),
            
            html.Div(id='prediction-details', className="mt-4"),
            
        ], width=12, lg=7, className="px-4"),
    ], className="my-5"),
    
    html.Hr(),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P("Income Prediction ML Model | Powered by Scikit-learn + Plotly Dash",
                      className="text-muted text-center small mb-0"),
            ], className="text-center py-3")
        ])
    ]),
    
], fluid=True, style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh'})


# ==================== CALLBACKS ====================
@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-details', 'children')],
    [
        Input('age-slider', 'value'),
        Input('education-num-slider', 'value'),
        Input('capital-gain-slider', 'value'),
        Input('capital-loss-slider', 'value'),
        Input('hours-slider', 'value'),
        Input('workclass-dropdown', 'value'),
        Input('education-dropdown', 'value'),
        Input('marital-dropdown', 'value'),
        Input('occupation-dropdown', 'value'),
        Input('relationship-dropdown', 'value'),
        Input('race-dropdown', 'value'),
        Input('sex-dropdown', 'value'),
        Input('country-dropdown', 'value'),
    ],
    prevent_initial_call=True
)
def make_prediction(age, education_num, capital_gain, capital_loss, hours,
                   workclass, education, marital, occupation, relationship, race, sex, country):
    """Make income prediction based on user inputs."""
    
    if MODEL is None or PREPROCESSOR is None:
        return (
            dbc.Alert("Model artifacts not loaded. Please run train_pipeline.py first.", color="danger"),
            html.Div()
        )
    
    try:
        # Create input dataframe matching training data structure
        input_data = pd.DataFrame({
            'age': [age],
            'workclass': [workclass],
            'education_level': [education],
            'education-num': [education_num],
            'marital-status': [marital],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'sex': [sex],
            'capital-gain': [capital_gain],
            'capital-loss': [capital_loss],
            'hours-per-week': [hours],
            'native-country': [country],
        })
        
        # Validate input columns
        expected_cols = ['age', 'workclass', 'education_level', 'education-num', 'marital-status', 
                        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 
                        'capital-loss', 'hours-per-week', 'native-country']
        
        for col in expected_cols:
            if col not in input_data.columns:
                raise ValueError(f"Missing column: {col}")
        
        # Preprocess input
        input_processed = PREPROCESSOR.transform(input_data)
        
        # Validate processed shape
        if input_processed.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Feature mismatch: expected {len(FEATURE_NAMES)} features, got {input_processed.shape[1]}")
        
        # Make prediction
        prediction = MODEL.predict(input_processed)[0]
        probabilities = MODEL.predict_proba(input_processed)[0]
        
        income_class = ">50K" if prediction == 1 else "<=50K"
        confidence = max(probabilities) * 100
        
        # Create output card
        output_card = dbc.Card([
            dbc.CardBody([
                html.H3(f"Predicted Income: {income_class}", 
                       className="text-center mb-3",
                       style={'color': '#27ae60' if prediction == 1 else '#e74c3c'}),
                
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.P("Confidence Score", className="fw-bold text-muted"),
                            html.H4(f"{confidence:.2f}%", style={'color': '#3498db'}),
                        ], width=6),
                        dbc.Col([
                            html.P("Prediction Class", className="fw-bold text-muted"),
                            html.H4(f"{prediction} (1='>50K', 0='<=50K')", className="text-muted"),
                        ], width=6),
                    ])
                ], className="my-4"),
                
                # Probability bars
                html.P("Probability Distribution:", className="fw-bold mt-4 mb-2"),
                dbc.Row([
                    dbc.Col([
                        html.P("<=50K", className="text-muted small"),
                        dbc.Progress(value=probabilities[0] * 100, style={"height": "25px"})
                    ], width=6),
                    dbc.Col([
                        html.P(">50K", className="text-muted small"),
                        dbc.Progress(value=probabilities[1] * 100, style={"height": "25px"})
                    ], width=6),
                ]),
            ])
        ], className="mt-3 shadow-sm")
        
        # Input summary
        details = dbc.Card([
            dbc.CardBody([
                html.H5("Input Summary", className="mb-3"),
                html.Div([
                    dbc.Row([
                        dbc.Col(html.P(f"Age: {age}", className="small"), width=6),
                        dbc.Col(html.P(f"Sex: {sex}", className="small"), width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(html.P(f"Education: {education} ({education_num} yrs)", className="small"), width=6),
                        dbc.Col(html.P(f"Hours/Week: {hours}", className="small"), width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(html.P(f"Work Class: {workclass}", className="small"), width=6),
                        dbc.Col(html.P(f"Occupation: {occupation}", className="small"), width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(html.P(f"Capital Gain: ${capital_gain:,}", className="small"), width=6),
                        dbc.Col(html.P(f"Capital Loss: ${capital_loss:,}", className="small"), width=6),
                    ]),
                ])
            ])
        ], className="mt-3")
        
        return output_card, details
    
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        return (
            dbc.Alert(f"{error_msg}", color="danger"),
            html.Div()
        )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING DASH APPLICATION")
    print("="*70)
    print("\nDashboard URL: http://127.0.0.1:8050/")
    print("   Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
