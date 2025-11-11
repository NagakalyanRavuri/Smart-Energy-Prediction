import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout


def run_training(csv_path: str, seq_len: int = 48, epochs: int = 10, batch_size: int = 32):
	# Load
	df = pd.read_csv(csv_path)
	df.columns = df.columns.str.strip()
	df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
	df = df.sort_values("Datetime")
	df = df.drop(columns=["Date", "Time"])

	# Encode
	le_appliance = LabelEncoder()
	le_season = LabelEncoder()
	df["Appliance Type"] = le_appliance.fit_transform(df["Appliance Type"])
	df["Season"] = le_season.fit_transform(df["Season"])

	# Features
	features = ["Appliance Type", "Outdoor Temperature (°C)", "Season", "Household Size", "Energy Consumption (kWh)"]
	data = df[features].values

	# Scale
	scaler = MinMaxScaler()
	scaled = scaler.fit_transform(data)

	# Sequences
	def create_sequences(arr, lookback):
		X, y = [], []
		for i in range(len(arr) - lookback):
			X.append(arr[i:i + lookback])
			y.append(arr[i + lookback, -1])
		return np.array(X), np.array(y)

	X, y = create_sequences(scaled, seq_len)
	if len(X) == 0:
		raise ValueError("Not enough rows to create sequences. Increase data or reduce seq_len.")

	split = int(0.8 * len(X))
	X_train, X_test = X[:split], X[split:]
	y_train, y_test = y[:split], y[split:]

	# RNN
	rnn = Sequential([
		SimpleRNN(64, return_sequences=True, input_shape=(seq_len, X.shape[2])),
		Dropout(0.2),
		SimpleRNN(32),
		Dense(1)
	])
	rnn.compile(optimizer="adam", loss="mse")
	rnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test))

	# LSTM
	lstm = Sequential([
		LSTM(64, return_sequences=True, input_shape=(seq_len, X.shape[2])),
		Dropout(0.2),
		LSTM(32),
		Dense(1)
	])
	lstm.compile(optimizer="adam", loss="mse")
	lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test))

	# Predict + metrics
	rnn_pred = rnn.predict(X_test, verbose=0).reshape(-1)
	lstm_pred = lstm.predict(X_test, verbose=0).reshape(-1)

	def evaluate(y_true, y_pred):
		mae = float(mean_absolute_error(y_true, y_pred))
		rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
		return mae, rmse

	rnn_mae, rnn_rmse = evaluate(y_test, rnn_pred)
	lstm_mae, lstm_rmse = evaluate(y_test, lstm_pred)

	n = min(200, len(y_test))
	return {
		"metrics": {
			"rnn": {"mae": rnn_mae, "rmse": rnn_rmse},
			"lstm": {"mae": lstm_mae, "rmse": lstm_rmse}
		},
		"series": {
			"actual": y_test[:n].tolist(),
			"rnn": rnn_pred[:n].tolist(),
			"lstm": lstm_pred[:n].tolist()
		}
	}


if __name__ == "__main__":
	project_root = os.path.dirname(os.path.abspath(__file__))
	csv_path = os.path.join(project_root, "smart_home_energy_consumption.csv")
	if not os.path.exists(csv_path):
		raise SystemExit("smart_home_energy_consumption.csv not found at project root.")
	result = run_training(csv_path)
	out_dir = os.path.join(project_root, "frontend")
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, "data.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(result, f)
	print(f"Saved precomputed results to {out_path}")


