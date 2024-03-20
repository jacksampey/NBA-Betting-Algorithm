from data import DataHandler
from models import train_xgboost, evaluate_model, cross_validate_model, get_model, calculate_ev
from visualizations import plot_ev_over_time, plot_ev_linear
import pandas as pd
import numpy as np

def main():
    filepath = 'NBA-Betting-Algorithm\game_data.csv'
    
    data_handler = DataHandler(filepath)
    
    # Load and preprocess data
    data = data_handler.load_data()
    if data is not None:
        data_clean = data_handler.preprocess_data(data)
        if data_clean is not None:
            # Display initial data insights
            columns_to_print = ['game_id', 'date', 'team_favorite', 'spread_five', 'final_score_difference', 'total_score', 'outcome']
            print(data_clean[columns_to_print].head())
            
            # Prepare data for modeling
            X = data_clean[['spread_five', 'total_score']]
            y = data_clean['outcome']

            # Model training and evaluation
            y_test, y_pred = train_xgboost(X, y)
            metrics = evaluate_model(y_test, y_pred)
            print("XGBoost metrics:", metrics)

            # Expected Value (EV) calculation
            model_ev = calculate_ev(y_test, y_pred)
            print(f"Model Expected Value: {model_ev}")

            # Random strategy comparison
            random_pred = np.random.randint(2, size=len(y_test))
            random_ev = calculate_ev(y_test, random_pred)
            print(f"Random Strategy Expected Value: {random_ev}")

            # EV over time visualization
            plot_ev_linear(y_test, y_pred, random_pred)
            plot_ev_over_time(y_test, y_pred, random_pred, num_bets=len(y_test))
        else:
            print("Error cleaning data.")
    else:
        print("Data could not be loaded.")

if __name__ == "__main__":
    main()

