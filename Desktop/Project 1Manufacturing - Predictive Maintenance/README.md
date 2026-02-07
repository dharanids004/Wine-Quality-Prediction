# IoT Predictive Maintenance (FactoryGuard AI)

This project implements a production-ready pipeline for predictive maintenance using time-series sensor data. It includes:

- Data cleaning and interpolation
- Rolling window, lag, and EMA feature engineering
- Baseline Logistic Regression
- XGBoost model with imbalance handling (SMOTE and/or scale_pos_weight)
- Time-aware train/test split to avoid leakage
- SHAP explainability reports
- Flask API for deployment

## Project Structure

- `src/train.py` - training pipeline
- `src/features.py` - feature engineering utilities
- `src/api.py` - inference API
- `src/config.yaml` - configuration
- `artifacts/` - saved model and metadata
- `reports/` - metrics, correlation matrix, SHAP outputs

## Setup

```bash
pip install -r requirements.txt
```

## Configure

Edit `src/config.yaml` and set:

- `data.path` to your CSV (e.g., `data/sensor_logs.csv`)
- `data.time_col`, `data.id_col`, `data.target_col`
- If the dataset does not have `target_col`, set `failure_event_col` and `lead_time_hours` so the script builds the target.

## Train

```bash
python src/train.py
```

Outputs:

- `reports/metrics.json`
- `reports/corr_matrix.csv`
- `reports/shap_summary.png` (if SHAP succeeds)
- `reports/shap_force.html`
- `artifacts/model.joblib`

## Run API

```bash
python src/api.py
```

### Request Example

```json
{
  "history": [
    {"timestamp": "2025-01-01 00:00:00", "machine_id": "M01", "vibration": 0.12, "temperature": 63.2, "pressure": 2.1},
    {"timestamp": "2025-01-01 00:10:00", "machine_id": "M01", "vibration": 0.15, "temperature": 64.0, "pressure": 2.0}
  ],
  "current": {"timestamp": "2025-01-01 00:20:00", "machine_id": "M01", "vibration": 0.22, "temperature": 66.1, "pressure": 2.3}
}
```

Response:

```json
{
  "failure_probability": 0.42,
  "prediction": 0
}
```

## Notes

- The feature pipeline avoids leakage by shifting rolling windows.
- If your data is huge, consider using Spark for ETL and export to CSV for modeling.
