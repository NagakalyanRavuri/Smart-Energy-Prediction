from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)


def run_training(csv_path: str, seq_len: int = 48, epochs: int = 10, batch_size: int = 32):
	# 1. Load Dataset
	df = pd.read_csv(csv_path)
	df.columns = df.columns.str.strip()

	# Combine Date + Time into one column
	df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
	df = df.sort_values("Datetime")
	df = df.drop(columns=["Date", "Time"])

	# 2. Encode Categorical Features
	le_appliance = LabelEncoder()
	le_season = LabelEncoder()
	df["Appliance Type"] = le_appliance.fit_transform(df["Appliance Type"])
	df["Season"] = le_season.fit_transform(df["Season"])

	# 3. Select Features
	features = ["Appliance Type", "Outdoor Temperature (°C)", "Season", "Household Size", "Energy Consumption (kWh)"]
	data = df[features].values

	# 4. Feature Scaling
	scaler = MinMaxScaler()
	scaled_data = scaler.fit_transform(data)

	# 5. Create Sequences
	def create_sequences(arr, lookback):
		X, y = [], []
		for i in range(len(arr) - lookback):
			X.append(arr[i:i + lookback])
			y.append(arr[i + lookback, -1])
		return np.array(X), np.array(y)

	X, y = create_sequences(scaled_data, seq_len)

	# train/test split
	split = int(0.8 * len(X))
	X_train, X_test = X[:split], X[split:]
	y_train, y_test = y[:split], y[split:]

	# 6. RNN
	rnn_model = Sequential([
		SimpleRNN(64, return_sequences=True, input_shape=(seq_len, X.shape[2])),
		Dropout(0.2),
		SimpleRNN(32),
		Dense(1)
	])
	rnn_model.compile(optimizer="adam", loss="mse")
	rnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

	# 7. LSTM
	lstm_model = Sequential([
		LSTM(64, return_sequences=True, input_shape=(seq_len, X.shape[2])),
		Dropout(0.2),
		LSTM(32),
		Dense(1)
	])
	lstm_model.compile(optimizer="adam", loss="mse")
	lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

	# 8. Predictions & Evaluation
	rnn_pred = rnn_model.predict(X_test, verbose=0).reshape(-1)
	lstm_pred = lstm_model.predict(X_test, verbose=0).reshape(-1)

	def evaluate(y_true, y_pred):
		mae = float(mean_absolute_error(y_true, y_pred))
		rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
		return mae, rmse

	rnn_mae, rnn_rmse = evaluate(y_test, rnn_pred)
	lstm_mae, lstm_rmse = evaluate(y_test, lstm_pred)

	# sample for plotting (first 200)
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
		},
		"seq_len": seq_len,
		"num_features": X.shape[2]
	}


@app.route("/")
def index():
	return send_from_directory(app.static_folder, "index.html")


@app.route("/api/train", methods=["GET"])
def api_train():
	csv_path = os.path.join(os.getcwd(), "smart_home_energy_consumption.csv")
	if not os.path.exists(csv_path):
		return jsonify({"error": "Dataset smart_home_energy_consumption.csv not found in project root."}), 404
	try:
		result = run_training(csv_path)
		return jsonify(result)
	except Exception as e:
		return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
	# Windows friendly
	app.run(host="127.0.0.1", port=5000, debug=True)


