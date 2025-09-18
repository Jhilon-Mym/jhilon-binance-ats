import pandas as pd
import os

def load_data():
	csv_path = os.path.join(os.path.dirname(__file__), "..", "klines_BTCUSDT_5m.csv")
	df = pd.read_csv(csv_path)
	return df

def show_label_stats():
	df = load_data()
	print("Label distribution:")
	print(df["label"].value_counts())
	print("Feature summary:")
	print(df.describe())

if __name__ == "__main__":
	show_label_stats()
