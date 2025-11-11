## Smart Home Energy Consumption Prediction (LSTM vs RNN)

This project trains Simple RNN and LSTM neural networks to predict smart home energy consumption and provides a web frontend to visualize results.

### Key Features
- End-to-end pipeline: preprocessing, sequence creation, training, evaluation.
- Flask backend exposing an API to trigger training.
- Modern HTML/CSS/JS frontend using Chart.js to plot actual vs predicted values.
- Metrics reported: MAE and RMSE for both RNN and LSTM.

---

## Dataset
Place your dataset file at the project root with the exact name: `smart_home_energy_consumption.csv`.

Required columns (case-sensitive after trimming spaces):
- `Date` (e.g., `2023-01-01`)
- `Time` (e.g., `08:30:00`)
- `Appliance Type` (categorical, will be label-encoded)
- `Outdoor Temperature (°C)` (numeric)
- `Season` (categorical, will be label-encoded)
- `Household Size` (numeric/integer)
- `Energy Consumption (kWh)` (numeric, target)

The app will:
1. Combine `Date` + `Time` into a `Datetime` column.
2. Sort by `Datetime` and drop `Date`/`Time` afterwards.
3. Label-encode `Appliance Type` and `Season`.
4. Scale all features using MinMaxScaler.
5. Create sequences of length 48 and predict the next step of `Energy Consumption (kWh)`.

---

## Project Structure
```
DL PROJECT/
├─ app.py                  # Flask backend + training endpoint
├─ requirements.txt        # Python dependencies
├─ smart_home_energy_consumption.csv  # Your dataset (not included)
├─ frontend/
│  ├─ index.html
│  ├─ styles.css
│  └─ app.js
└─ README.md
```

---

## Setup (Windows PowerShell)
1. Create and activate a virtual environment (recommended):
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Ensure `smart_home_energy_consumption.csv` is in the project root.

### Option A: Run full app with Flask API
4A. Run the app:
```powershell
python app.py
```

5A. Open the frontend in your browser:
- Navigate to `http://127.0.0.1:5000/`

6A. Click “Run Training” to start. Training may take from seconds to a couple of minutes depending on your CPU/GPU and dataset size.

### Option B: Pure static mode (no Flask API)
4B. Precompute results JSON:
```powershell
python generate_results.py
```
This creates `frontend/data.json` with metrics and first 200 points for plotting.

5B. Serve the static frontend (no Flask):
```powershell
cd frontend
python -m http.server 5500
```
Open `http://127.0.0.1:5500/` and click “Load Results” to display the precomputed metrics and chart.

---

## Notes
- The frontend shows the first 200 test points to keep charts readable.
- Metrics are computed on the test split (`last 20%` of the sequence windows).
- If you change the CSV filename or columns, update `app.py` accordingly.
- For faster experiments, reduce epochs in `app.py` or use a smaller dataset subset.

---

## Original Notebook/Script
If you prefer running everything in pure Python without the web UI, you can use your provided script. This repository integrates that logic into a Flask server and static web UI for easy visualization.


