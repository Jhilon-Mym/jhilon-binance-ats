import os
import pickle
import numpy as np
import pandas as pd

def load_model_scaler():
	model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
	scaler_path = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
	with open(model_path, "rb") as f:
		model = pickle.load(f)
	with open(scaler_path, "rb") as f:
		scaler = pickle.load(f)
	return model, scaler

def test_predict():
	model, scaler = load_model_scaler()
	# ডেমো ডাটা (random, বাস্তবে df থেকে নিন)
	X = np.random.rand(1, 6)
	X_scaled = scaler.transform(X)
	pred = model.predict(X_scaled)
	print(f"Prediction: {pred}")

if __name__ == "__main__":
	test_predict()
