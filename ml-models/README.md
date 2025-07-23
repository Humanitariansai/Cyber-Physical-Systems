# Basic Time-Series Forecasting

**Author:** Udisha Dutta Chowdhury  
**Supervisor:** Prof. Rolando Herrero

## Overview

This module implements a basic time-series forecasting model using linear regression with lag features. It's designed to provide simple, interpretable forecasting for sensor data in cyber-physical systems.

## Features

- **Simple Implementation**: Uses scikit-learn's LinearRegression with lag features
- **Lag Feature Engineering**: Automatically creates lag features from time series data
- **Model Evaluation**: Includes RMSE, MAE, and R² metrics
- **Visualization**: Basic plotting functionality for predictions
- **Error Handling**: Robust error checking and validation

## Files

- `basic_forecaster.py` - Main forecasting implementation
- `test_basic_forecaster.py` - Test suite
- `requirements.txt` - Dependencies
- `README.md` - This documentation

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from basic_forecaster import BasicTimeSeriesForecaster, create_sample_data

# Generate sample data
data = create_sample_data(n_points=100)

# Create and train forecaster
forecaster = BasicTimeSeriesForecaster(n_lags=6)
forecaster.fit(data, target_col='temperature')

# Make predictions
predictions = forecaster.predict(data, n_steps=5)
print(f"Next 5 predictions: {predictions}")

# Evaluate model
train_data = data[:80]
test_data = data[80:]
forecaster.fit(train_data, 'temperature')
metrics = forecaster.evaluate(test_data)
print(f"RMSE: {metrics['rmse']:.3f}")
```

### Running the Demo

```bash
python basic_forecaster.py
```

### Running Tests

```bash
python test_basic_forecaster.py
```

## How It Works

### Lag Features

The forecaster creates lag features by shifting the time series:
- `value_lag_1`: Previous time step value
- `value_lag_2`: Value from 2 time steps ago
- `value_lag_3`: Value from 3 time steps ago
- etc.

### Linear Regression

Uses these lag features as input to predict the next value:
```
y(t) = β₀ + β₁·y(t-1) + β₂·y(t-2) + ... + βₙ·y(t-n)
```

### Multi-step Prediction

For multi-step ahead predictions, uses recursive forecasting where each prediction becomes input for the next.

## Model Parameters

- **n_lags**: Number of lag features to use (default: 5)
  - Higher values capture longer dependencies
  - Lower values reduce overfitting risk

## Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of determination

## Example Output

```
BASIC TIME-SERIES FORECASTING DEMO
========================================
1. Generating sample temperature data...
   Generated 168 data points
2. Split data: 134 train, 34 test samples
3. Training forecaster...
   Model fitted with 129 samples and 6 lag features
4. Evaluating on test data...
   Model Performance Metrics:
   - RMSE: 1.234
   - MAE:  0.987
   - R²:   0.845
5. Making future predictions...
   Next 5 hour predictions:
   - Hour 1: 21.45°C
   - Hour 2: 22.13°C
   - Hour 3: 22.78°C
   - Hour 4: 23.21°C
   - Hour 5: 23.45°C
```

## Limitations

- **Linear Model**: Only captures linear relationships
- **No Seasonality**: Doesn't explicitly model seasonal patterns
- **Simple Features**: Only uses lag features, no external variables
- **Recursive Prediction**: Error accumulates in multi-step forecasts

## Future Enhancements

- XGBoost for non-linear patterns
- Seasonal decomposition
- External feature integration
- Advanced evaluation frameworks
- Model selection and tuning

## Dependencies

- scikit-learn >= 1.0.0
- numpy >= 1.21.0
- pandas >= 1.5.0
- matplotlib >= 3.5.0
- joblib >= 1.2.0
