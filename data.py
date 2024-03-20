import pandas as pd

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        try:
            data = pd.read_csv(self.filepath)
            print("Data loaded successfully.")
            required_columns = ['game_id', 'date', 'team_favorite', 'team_underdog', 'spread_five', 'final_score_difference','total_score', 'outcome']
            if not all(column in data.columns for column in required_columns):
                print("Missing required columns.")
                return None
            return data
        except FileNotFoundError:
            print(f"File not found: {self.filepath}")
            return None
    def preprocess_data(self, data):
        data_clean = data.dropna(subset=['spread_five', 'final_score_difference', 'total_score', 'outcome'])
        return data_clean