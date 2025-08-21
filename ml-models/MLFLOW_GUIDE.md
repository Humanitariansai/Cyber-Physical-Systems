# MLflow Integration Guide for Cyber-Physical Systems Forecasting

## Overview

This guide explains how MLflow experiment tracking is integrated into your forecasting models and how to add new models effectively.

## 🏗️ Architecture

```
Your Forecasting System + MLflow
├── MLflow Tracking Server (localhost:5000)
├── Experiments (logical groupings)
│   ├── linear-regression-forecasting
│   ├── xgboost-forecasting  
│   ├── comprehensive-forecasting-comparison
│   └── advanced-forecasting-models
├── Runs (individual model training sessions)
│   ├── Parameters (hyperparameters, configs)
│   ├── Metrics (RMSE, MAE, R², etc.)
│   ├── Artifacts (models, plots, data files)
│   └── Tags (metadata, categories)
└── Model Registry (versioned model storage)
```

## 🔄 How It Works in Your Scenario

### 1. **Experiment Organization**
- **Experiments** group related runs (e.g., all Linear Regression variations)
- **Runs** track individual training sessions with specific parameters
- **Tags** help categorize and filter runs

### 2. **Automatic Tracking**
```python
# Your models automatically log to MLflow
forecaster = BasicTimeSeriesForecaster(n_lags=6, enable_mlflow=True)
forecaster.fit(train_data, target_col='temperature', run_name='my_experiment')
# ✅ Automatically logs: parameters, training data info, model artifact

metrics = forecaster.evaluate(test_data, log_to_mlflow=True) 
# ✅ Automatically logs: metrics, prediction plots, performance charts
```

### 3. **What Gets Tracked**

#### Parameters Logged:
- `n_lags`: Number of lag features
- `model_type`: Type of algorithm used
- `target_column`: Column being predicted
- `data_points`: Size of training data
- Model-specific hyperparameters

#### Metrics Logged:
- `rmse`: Root Mean Square Error
- `mae`: Mean Absolute Error
- `r2`: R-squared score
- `mape`: Mean Absolute Percentage Error (for XGBoost)
- `n_samples`: Number of test samples

#### Artifacts Logged:
- **Model files**: Trained model objects
- **Prediction plots**: Actual vs predicted visualizations
- **Residual plots**: Error analysis charts
- **Feature importance**: For tree-based models
- **Data summaries**: Training/test data statistics

## 📊 Current Model Integration

### 1. **Linear Regression** (basic_forecaster.py)
```python
# ✅ Fully integrated with MLflow
forecaster = BasicTimeSeriesForecaster(enable_mlflow=True)
# Logs: lag features, linear regression parameters, sklearn model
```

### 2. **XGBoost** (xgboost_forecaster.py)  
```python
# ✅ Fully integrated with MLflow
forecaster = XGBoostTimeSeriesForecaster(enable_mlflow=True)
# Logs: 35+ engineered features, XGBoost parameters, feature importance
```

### 3. **Moving Averages** (simple_arima_forecaster.py)
```python
# ⚠️ No MLflow integration (statistical method)
forecaster = SimpleMovingAverageForecaster()
# Manual logging possible through MLflowModelManager
```

## 🚀 Adding New Models - Step by Step

### Method 1: Direct Integration (Recommended)

```python
class YourNewForecaster:
    def __init__(self, enable_mlflow=True):
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        if self.enable_mlflow:
            self.mlflow_tracker = ExperimentTracker("your-model-name")
    
    def fit(self, data, target_col='temperature', run_name=None):
        # Start MLflow run
        if self.enable_mlflow:
            self.current_run_id = self.mlflow_tracker.start_run(
                run_name=run_name or f"your_model_{datetime.now():%Y%m%d_%H%M%S}",
                model_type="your_model_type"
            )
            
            # Log parameters
            self.mlflow_tracker.log_parameters({
                "param1": self.param1,
                "param2": self.param2,
                # ... your parameters
            })
            
            # Log dataset info
            self.mlflow_tracker.log_dataset_info(data, "training_data")
        
        # Your training logic here
        # ...
        
        # Log model artifact
        if self.enable_mlflow:
            self.mlflow_tracker.log_model(self.model, "sklearn")  # or "xgboost", "pytorch"
    
    def evaluate(self, test_data, log_to_mlflow=True):
        # Your evaluation logic
        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        
        # Log to MLflow
        if log_to_mlflow and self.enable_mlflow:
            self.mlflow_tracker.log_metrics(metrics)
            self.mlflow_tracker.log_prediction_results(y_true, y_pred, "test_")
        
        return metrics
    
    def finish_mlflow_run(self):
        if self.enable_mlflow and self.current_run_id:
            self.mlflow_tracker.end_run()
```

### Method 2: Using MLflowModelManager

```python
from mlflow_model_guide import MLflowModelManager

# Set up manager
manager = MLflowModelManager("my-experiment")

# Register your model
manager.register_model(
    model_name="My New Model",
    model_class=YourModelClass,
    model_params={"param1": "value1", "param2": "value2"},
    mlflow_type="sklearn"  # or "xgboost", "pytorch", etc.
)

# Run experiment
result = manager.run_experiment(
    model_name="My New Model",
    train_data=train_data,
    test_data=test_data,
    custom_params={"param1": "custom_value"}
)
```

## 🎯 Model Types Supported

### 1. **Scikit-learn Models**
```python
mlflow_type="sklearn"
# Automatically handles: LinearRegression, RandomForest, SVM, etc.
# Logs with: mlflow.sklearn.log_model()
```

### 2. **XGBoost Models**
```python
mlflow_type="xgboost" 
# Handles: XGBRegressor, XGBClassifier
# Logs with: mlflow.xgboost.log_model()
# Includes: Feature importance, hyperparameters
```

### 3. **PyTorch Models**
```python
mlflow_type="pytorch"
# Handles: Neural networks, LSTM, CNN
# Logs with: mlflow.pytorch.log_model()
# Includes: Model architecture, state dict
```

### 4. **TensorFlow/Keras Models**
```python
mlflow_type="tensorflow"
# Handles: Sequential, Functional API models
# Logs with: mlflow.tensorflow.log_model()
# Includes: Model architecture, weights
```

### 5. **Custom Models**
```python
mlflow_type="sklearn"  # Use as fallback
# For any custom implementation
# Manual artifact logging required
```

## 🔍 Viewing Results

### 1. **Start MLflow UI**
```bash
# In your terminal
cd "C:\Users\udish\OneDrive\Documents\Udisha\Full time\Cyber-Physical Systems\Cyber-Physical-Systems\ml-models"
mlflow ui --host 127.0.0.1 --port 5000
```

### 2. **Access Dashboard**
- Open: http://localhost:5000
- View experiments, runs, metrics, artifacts
- Compare models side-by-side
- Download model artifacts

### 3. **Programmatic Access**
```python
from mlflow_tracking import ExperimentTracker

tracker = ExperimentTracker("your-experiment")
runs_df = tracker.get_experiment_results()
print(runs_df[['run_id', 'metrics.rmse', 'metrics.mae', 'params.n_lags']])
```

## 📈 Best Practices

### 1. **Experiment Naming**
- Use descriptive names: `"temperature-forecasting-comparison"`
- Group related experiments: `"lstm-variants"`, `"ensemble-methods"`

### 2. **Run Naming**
- Include timestamp: `f"xgboost_{datetime.now():%Y%m%d_%H%M%S}"`
- Include key parameters: `f"rf_n_estimators_{n_estimators}_depth_{max_depth}"`

### 3. **Parameter Logging**
- Log all hyperparameters
- Include data preprocessing parameters
- Log feature engineering choices

### 4. **Metric Selection**
- Always log: RMSE, MAE, R²
- Domain-specific: MAPE for percentage errors
- Custom metrics: Directional accuracy, trend prediction

### 5. **Artifact Management**
- Log model files for reproduction
- Save prediction plots for visual analysis
- Store feature importance for interpretability

## 🛠️ Practical Examples

### Example 1: Adding LSTM Model
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMForecaster:
    def __init__(self, sequence_length=24, lstm_units=50, enable_mlflow=True):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.enable_mlflow = enable_mlflow
        
        if enable_mlflow:
            self.mlflow_tracker = ExperimentTracker("lstm-forecasting")
    
    def fit(self, data, target_col='temperature', epochs=50):
        if self.enable_mlflow:
            run_id = self.mlflow_tracker.start_run(
                run_name=f"lstm_{self.lstm_units}units_{epochs}epochs",
                model_type="lstm"
            )
            
            self.mlflow_tracker.log_parameters({
                "sequence_length": self.sequence_length,
                "lstm_units": self.lstm_units,
                "epochs": epochs,
                "optimizer": "adam"
            })
        
        # Create sequences
        X, y = self.create_sequences(data[target_col].values)
        
        # Build model
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=False, input_shape=(self.sequence_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train with MLflow callback
        if self.enable_mlflow:
            import mlflow.tensorflow
            mlflow.tensorflow.autolog()
        
        model.fit(X, y, epochs=epochs, validation_split=0.2, verbose=0)
        self.model = model
        
        if self.enable_mlflow:
            self.mlflow_tracker.end_run()
```

### Example 2: Adding Ensemble Model
```python
class EnsembleForecaster:
    def __init__(self, base_models, weights=None):
        self.base_models = base_models
        self.weights = weights or [1/len(base_models)] * len(base_models)
        
    def fit_with_mlflow(self, train_data, test_data):
        tracker = ExperimentTracker("ensemble-forecasting")
        
        run_id = tracker.start_run(
            run_name=f"ensemble_{len(self.base_models)}models",
            model_type="ensemble"
        )
        
        # Log ensemble configuration
        tracker.log_parameters({
            "num_base_models": len(self.base_models),
            "model_types": [type(m).__name__ for m in self.base_models],
            "weights": self.weights
        })
        
        # Train each base model and log sub-metrics
        for i, model in enumerate(self.base_models):
            model.fit(train_data)
            sub_metrics = model.evaluate(test_data)
            
            # Log with prefix
            prefixed_metrics = {f"base_model_{i}_{k}": v for k, v in sub_metrics.items()}
            tracker.log_metrics(prefixed_metrics)
        
        # Evaluate ensemble
        ensemble_metrics = self.evaluate(test_data)
        tracker.log_metrics(ensemble_metrics)
        
        tracker.end_run()
```

## 🔧 Troubleshooting

### Common Issues:

1. **MLflow not found**: Install with `pip install mlflow`
2. **Permission errors**: Check file/directory permissions
3. **Port conflicts**: Use different port: `mlflow ui --port 5001`
4. **Model logging fails**: Check model type compatibility
5. **Large artifacts**: Use artifact compression or external storage

### Debug Mode:
```python
# Enable debug logging
import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)
```

## 📚 Next Steps

1. **Run the examples**: Test `advanced_mlflow_models.py`
2. **Add your models**: Follow the patterns shown
3. **Experiment systematically**: Compare different architectures
4. **Analyze results**: Use MLflow UI for insights
5. **Deploy best models**: Use MLflow Model Registry

## 🎉 Summary

MLflow in your forecasting system provides:
- ✅ **Automatic experiment tracking** for all model types
- ✅ **Systematic comparison** of different approaches  
- ✅ **Reproducible results** with parameter logging
- ✅ **Visual analysis** with artifact storage
- ✅ **Model versioning** and deployment support

Your forecasting models now have enterprise-grade experiment tracking!
