# AI Portfolio

My portfolio of applied AI and machine learning projects demonstrating various techniques across time series forecasting, natural language processing, and traditional machine learning tasks.

## Features

- **Modular Architecture**: Clean separation between different ML domains with reusable pipeline components
- **Production-Ready Code**: Well-structured, documented, and maintainable implementations
- **Comprehensive Evaluation**: Multiple metrics, visualisations, and comparative analysis
- **Configuration-Driven**: Easy experimentation through YAML configuration files
- **Extensible Design**: Framework supports adding new models and project types

## Current Implementation: Time Series Forecasting

The time series module provides a complete pipeline for forecasting with multiple models and evaluation horizons.

### Models in Use

- **XGBoost**: Gradient boosting implementation for time series prediction
- **Prophet**: Facebook's forecasting tool for time series with seasonality

### Capabilities

- Multi-horizon forecasting (e.g., 1-day, 7-day, 30-day ahead)
- Comprehensive metrics: MAE, RMSE, MAPE, SMAPE, R², MASE
- Automated train/validation/test splitting with temporal ordering
- Feature engineering (lag features, date components)
- Model comparison with detailed visualisations
- Persistent model storage and results tracking

## Project Structure

```text
AI-Portfolio/
├── TimeSeries/              # Time series forecasting
│   ├── configs/
│   │   └── config.yaml      # Model and pipeline configuration
│   ├── data/
│   │   ├── raw/             # Original time series data
│   │   └── processed/       # Processed datasets
│   ├── models/              # Model implementations
│   │   ├── statistical/     # Prophet, ARIMA, ETS, Theta
│   │   ├── ml/              # XGBoost, LightGBM, Random Forest
│   │   └── neural/          # LSTM, GRU, N-BEATS, Transformer
│   ├── src/                 # Core pipeline code
│   │   ├── data_loader.py   # Data loading and preprocessing
│   │   ├── pipeline.py      # Main pipeline orchestration
│   │   ├── evaluate.py      # Evaluation and metrics
│   │   └── utils.py         # Utility functions
│   ├── results/             # Output storage
│   │   ├── models/          # Saved models
│   │   ├── figures/         # Visualisations
│   │   └── metrics/         # Performance metrics
│   ├── main.py              # Entry point
│   └── requirements.txt     # Dependencies
├── NLP/                     # Natural Language Processing (PLANNED)
├── MachineLearning/         # Traditional ML tasks (PLANNED)
└── DataVisualisation/       # Data visualisation projects (PLANNED)
```

## Setup

### Time Series Forecasting

1. **Navigate to the TimeSeries directory:**

```bash
cd TimeSeries
```

1. **Install dependencies:**

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate   # optional

uv pip install -r requirements.txt
```

## Usage

### Running Time Series Forecasting

Run the pipeline with default settings (XGBoost and Prophet):

```bash
python main.py
```

### Configuration

Edit `TimeSeries/configs/config.yaml` to customise:

- **Data Settings**: File paths, timestamp column, target variables
- **Model Parameters**: Hyperparameters for XGBoost and Prophet
- **Forecast Horizons**: Which time steps ahead to predict (e.g., [1, 7, 30])
- **Evaluation Metrics**: Which metrics to compute

Example configuration for XGBoost:

```yaml
models:
  ml:
    xgboost:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
```

### Experiment Configuration

Define experiments in `TimeSeries/main.py`:

```python
EXPERIMENTS = [
    {
        "model": "xgboost",
    },
    {
        "model": "prophet",
    },
]
```

### Results

After running the pipeline, results are saved to:

- **Models**: `TimeSeries/results/models/`
- **Metrics**: `TimeSeries/results/metrics/results_summary.json`
- **Figures**: `TimeSeries/results/figures/`

## Latest Results

Below are the latest model comparison results from the time series forecasting pipeline comparing XGBoost and Prophet across multiple forecast horizons.

### Model Comparison by Horizon

<p align="center">
  <img src="TimeSeries/results/figures/model_comparison_by_horizon.png" alt="Model Comparison by Horizon" width="800"/>
</p>

Comprehensive comparison of model performance across different forecast horizons (1-day, 7-day, 30-day ahead).

### Single Horizon Performance (H=1)

<p align="center">
  <img src="TimeSeries/results/figures/model_comparison_h1.png" alt="Model Comparison H1" width="800"/>
</p>

Detailed performance metrics for 1-step ahead forecasting.

### Test Set Predictions

<p align="center">
  <img src="TimeSeries/results/figures/test_comparison_all_horizons.png" alt="Test Comparison All Horizons" width="800"/>
</p>

Visual comparison of actual vs predicted values on the test set across all forecast horizons.

## Future Work

### Time Series

- Additional statistical models: ARIMA, ETS, Theta
- Machine learning models: LightGBM, Random Forest, Gradient Boosting
- Deep learning models: LSTM, GRU, N-BEATS, Transformer, TCN
- Advanced ensemble methods
- Hyperparameter optimisation
