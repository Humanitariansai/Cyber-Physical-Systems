# Simple Moving Average Time-Series Forecasting

## Overview
This module implements simple moving average-based time-series forecasting methods as an alternative to linear regression approaches. It provides three different moving average strategies:

- **Simple Moving Average (SMA)**: Equal weight to all values in the window
- **Exponential Moving Average (EMA)**: More weight to recent values  
- **Weighted Moving Average (WMA)**: Linear increasing weights to recent values

## Files

### Core Implementation
- **`simple_arima_forecaster.py`**: Main moving average forecasting implementation
- **`test_moving_average_forecaster.py`**: Comprehensive test suite

### Generated Results
- **`results/ma_predictions_*.csv`**: Future predictions  
- **`results/ma_metrics_*.json`**: Performance metrics and model metadata
- **`results/ma_forecast_plot_*.png`**: Visualization plots

## Quick Start

```python
from simple_arima_forecaster import SimpleMovingAverageForecaster, create_sample_data

# Generate sample data
data = create_sample_data(n_points=100)

# Method 1: Simple Moving Average
sma_forecaster = SimpleMovingAverageForecaster(window=7, method='sma')
sma_forecaster.fit(data, 'temperature')
sma_predictions = sma_forecaster.predict(n_steps=5)

# Method 2: Exponential Moving Average  
ema_forecaster = SimpleMovingAverageForecaster(method='ema', alpha=0.3)
ema_forecaster.fit(data, 'temperature')
ema_predictions = ema_forecaster.predict(n_steps=5)

# Method 3: Weighted Moving Average
wma_forecaster = SimpleMovingAverageForecaster(window=5, method='wma')
wma_forecaster.fit(data, 'temperature')
wma_predictions = wma_forecaster.predict(n_steps=5)
```

## Features

### Multiple Moving Average Methods
- **SMA**: `window` parameter controls averaging window
- **EMA**: `alpha` parameter controls decay rate (0 < alpha < 1)
- **WMA**: `window` parameter with linearly increasing weights

### Automatic Model Comparison
The demo automatically tests all three methods and selects the best performing one based on RMSE.

### Comprehensive Results Storage
- Timestamped output files
- Performance metrics (RMSE, MAE, R²)
- Model configuration metadata
- High-resolution plots

### Evaluation Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of determination
- **Method-specific metadata**: Window size, alpha values

## Running the Demo

```bash
python simple_arima_forecaster.py
```

Expected output:
```
============================================================
SIMPLE MOVING AVERAGE TIME-SERIES FORECASTING DEMO
============================================================
1. Generating sample temperature data...
   Generated 168 data points
2. Split data: 134 train, 34 test samples

3. Training SMA forecaster...
   SMA Performance Metrics:
   - RMSE: 2.656
   - MAE:  2.278
   - R²:   0.477

3. Training EMA forecaster...
   EMA Performance Metrics:
   - RMSE: 1.858
   - MAE:  1.599
   - R²:   0.744

3. Training WMA forecaster...
   WMA Performance Metrics:
   - RMSE: 1.399
   - MAE:  1.161
   - R²:   0.855

🏆 Best model: WMA (RMSE: 1.399)
```

## Running Tests

```bash
python test_moving_average_forecaster.py
```

## Method Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **SMA** | Simple, stable | Lags trends | Noisy data |
| **EMA** | Responsive to recent changes | Can be unstable | Trending data |
| **WMA** | Balance of stability/responsiveness | Moderate complexity | General purpose |

## Sample Results

### Performance Comparison (Typical Results)
- **SMA (window=7)**: RMSE ≈ 2.66, R² ≈ 0.48
- **EMA (α=0.3)**: RMSE ≈ 1.86, R² ≈ 0.74  
- **WMA (window=5)**: RMSE ≈ 1.40, R² ≈ 0.86

*WMA typically performs best for synthetic temperature data with daily patterns.*

## Output Files Structure

```
results/
├── ma_predictions_YYYYMMDD_HHMMSS.csv    # Future predictions
├── ma_metrics_YYYYMMDD_HHMMSS.json       # Performance metrics  
└── ma_forecast_plot_YYYYMMDD_HHMMSS.png  # Visualization
```

## Configuration Options

### Simple Moving Average (SMA)
```python
SimpleMovingAverageForecaster(
    method='sma',
    window=7,           # Number of periods to average
    results_dir='results'
)
```

### Exponential Moving Average (EMA)  
```python
SimpleMovingAverageForecaster(
    method='ema', 
    alpha=0.3,          # Smoothing parameter (0 < alpha < 1)
    results_dir='results'
)
```

### Weighted Moving Average (WMA)
```python
SimpleMovingAverageForecaster(
    method='wma',
    window=5,           # Number of periods with linear weights
    results_dir='results'  
)
```

## Integration with Other Models

This moving average forecaster complements the basic linear regression forecaster:

- **Linear Regression**: Uses lag features, good for complex patterns
- **Moving Averages**: Uses recent averages, good for trend following
- **Combined Approach**: Use ensemble of both for robust predictions

## Advanced Usage

### Custom Evaluation
```python
# Fit model
forecaster.fit(train_data, 'temperature')

# Evaluate performance
metrics = forecaster.evaluate(test_data)
print(f"RMSE: {metrics['rmse']:.3f}")

# Save results
forecaster.save_results(predictions, metrics)
```

### Plotting and Visualization
```python
# Generate plot with predictions
forecaster.plot_predictions(n_steps=10, save_plot=True)
```

This implementation provides a solid foundation for moving average-based forecasting in cyber-physical systems applications.
