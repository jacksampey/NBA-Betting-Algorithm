import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

def plot_ev_linear(y_true, model_pred, random_pred, bet_amount=1):
    plt.style.use('dark_background')  
    
    # Calculate cumulative gains/losses
    model_gains_losses = np.where(y_true == model_pred, bet_amount, -bet_amount).cumsum()
    random_gains_losses = np.where(y_true == random_pred, bet_amount, -bet_amount).cumsum()

    # Generate x-axis values (number of bets)
    x = np.arange(1, len(y_true) + 1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, model_gains_losses, label='Model Strategy', color='lime')
    plt.plot(x, random_gains_losses, label='Random Strategy', linestyle='--', color='cyan')
    plt.xlabel('Number of Bets')
    plt.ylabel('Cumulative Expected Value')
    plt.title('EV Over Time: Model Strategy vs. Random Strategy')
    plt.legend()
    plt.grid(True, color='gray')
    plt.show()


def plot_ev_over_time(y_test, model_preds, random_preds, num_bets=1000, bet_amount=1):
    plt.style.use('dark_background') 
    
    model_gains = np.cumsum(np.where(y_test[:num_bets] == model_preds[:num_bets], bet_amount, -bet_amount))
    random_gains = np.cumsum(np.where(y_test[:num_bets] == random_preds[:num_bets], bet_amount, -bet_amount))
    
    model_gains_pct = (model_gains / (np.arange(num_bets) + 1)) * 100
    random_gains_pct = (random_gains / (np.arange(num_bets) + 1)) * 100
    
    start_bet = 25
    if num_bets >= start_bet:
        x_values = np.arange(start_bet, num_bets + 1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, model_gains_pct[start_bet-1:], label='Model EV %', color='lime')  # Adjust the index to match x_values
        plt.plot(x_values, random_gains_pct[start_bet-1:], label='Random Choice EV %', color='cyan')  # Adjust the index
        plt.xlabel('Number of Bets')
        plt.ylabel('Cumulative EV (%)')
        plt.title('Cumulative Expected Value Percentage Over Time')
        plt.legend()
        plt.grid(True, color='gray')
        plt.xlim(left=start_bet)  # Ensure x-axis starts at 25
        plt.show()
    else:
        print(f"Not enough bets to start from {start_bet}. The total number of bets is {num_bets}.")
