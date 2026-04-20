from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def train_predictor(fake_df):
    X = fake_df[[
        'age', 'wbc_count', 
        'platelet_count', 'hemoglobin',
        'blast_cells_pct'
    ]]
    y = fake_df['has_leukemia']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

def predict_risk(model, child_data):
    risk = model.predict_proba(child_data)[0][1]
    return risk