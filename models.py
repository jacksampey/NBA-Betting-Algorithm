from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import xgboost as xgb
import numpy as np

def train_xgboost(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # XGBoost classifier
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    return y_test, y_pred

def evaluate_model(y_test, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return metrics
def get_model(name):
    if name == "logistic_regression":
        return LogisticRegression()
    elif name == "random_forest":
        return RandomForestClassifier()
    else:
        print("Model not recognized. Returning a Logistic Regression model.")
        return LogisticRegression()
def cross_validate_model(X, y, model, cv=5):
    #Cross validation
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores
def calculate_ev(y_true, y_pred, bet_amount=1):
    correct_predictions = y_true == y_pred
    gains_losses = np.where(correct_predictions, bet_amount, -bet_amount)
    expected_value = np.mean(gains_losses)
    return expected_value
