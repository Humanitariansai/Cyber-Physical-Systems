# Hyperparameter Optimization Guide

## Overview

This guide explains how to use the automated hyperparameter optimization system integrated with MLflow for your cyber-physical systems temperature forecasting project.

## 🎯 What This System Does

### Automated Optimization
- **Bayesian Optimization** using Optuna for intelligent parameter search
- **Grid Search** for exhaustive parameter exploration
- **Time Series Cross-Validation** for robust performance estimation
- **MLflow Integration** for comprehensive experiment tracking

### Supported Models
- **BasicTimeSeriesForecaster**: Linear regression with lag features
- **XGBoostForecaster**: Gradient boosting with advanced features
- **Multi-Model Comparison**: Automatic comparison across model types

## 🚀 Quick Start

### 1. Basic Usage

```python
from hyperparameter_optimizer import BasicForecasterOptimizer
import pandas as pd

# Load your data
data = pd.read_csv('your_temperature_data.csv')

# Create optimizer
optimizer = BasicForecasterOptimizer(
    experiment_name="my_optimization",
    n_trials=50,
    cv_folds=5
)

# Run optimization
study = optimizer.optimize(data, target_col='temperature')

# Get best parameters
best_params = study.best_params
best_score = study.best_value
```

### 2. Multi-Model Optimization

```python
from hyperparameter_optimizer import MultiModelOptimizer

# Create multi-model optimizer
multi_optimizer = MultiModelOptimizer(n_trials=30)

# Optimize all models and compare
results = multi_optimizer.optimize_all_models(data, target_col='temperature')

# Results contain best parameters for each model
for model_name, result in results.items():
    print(f"{model_name}: {result['best_params']} (RMSE: {result['best_score']:.4f})")
```

## 🔧 Configuration Options

### Optimizer Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `n_trials` | Number of optimization trials | 50 | 30-200 |
| `cv_folds` | Cross-validation folds | 5 | 3-10 |
| `experiment_name` | MLflow experiment name | auto | descriptive name |
| `tracking_uri` | MLflow tracking URI | "./mlruns" | local or remote |

### BasicTimeSeriesForecaster Parameters

| Parameter | Type | Search Range | Description |
|-----------|------|--------------|-------------|
| `n_lags` | int | 1-24 | Number of lag features |

### XGBoostForecaster Parameters

| Parameter | Type | Search Range | Description |
|-----------|------|--------------|-------------|
| `n_lags` | int | 3-24 | Number of lag features |
| `n_estimators` | int | 50-300 | Number of trees |
| `max_depth` | int | 3-10 | Maximum tree depth |
| `learning_rate` | float | 0.01-0.3 | Learning rate |
| `subsample` | float | 0.6-1.0 | Subsample ratio |
| `colsample_bytree` | float | 0.6-1.0 | Feature subsample ratio |
| `reg_alpha` | float | 0-10 | L1 regularization |
| `reg_lambda` | float | 0-10 | L2 regularization |

## 📊 MLflow Integration

### Experiment Tracking
Every optimization run is automatically tracked in MLflow with:

- **Parameters**: All hyperparameters being optimized
- **Metrics**: Cross-validation RMSE, MAE, R² scores
- **Artifacts**: Best models and optimization studies
- **Tags**: Model type, optimization metadata

### Dashboard Views
Access your optimization results at `http://127.0.0.1:5000`:

1. **Experiments Tab**: See all optimization experiments
2. **Runs Comparison**: Compare parameter combinations
3. **Metrics Visualization**: Performance trends and distributions
4. **Model Registry**: Best performing models

## 🎮 Advanced Usage

### Custom Objective Functions

```python
class CustomOptimizer(HyperparameterOptimizer):
    def objective(self, trial, data, target_col):
        # Define custom parameter suggestions
        param1 = trial.suggest_float('param1', 0.1, 1.0)
        param2 = trial.suggest_int('param2', 1, 10)
        
        # Create and evaluate model
        model = YourCustomModel(param1=param1, param2=param2)
        cv_results = self.evaluate_model_cv(model, data, target_col)
        
        return cv_results['mean_rmse']
```

### Grid Search Alternative

```python
from hyperparameter_optimization_demo import demo_grid_search_alternative

# Run exhaustive grid search
results = demo_grid_search_alternative()
```

### Parallel Optimization

```python
# Optuna supports parallel optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, n_jobs=4)  # 4 parallel workers
```

## 📈 Performance Tips

### Data Preparation
- **Ensure sufficient data**: At least 200-500 data points for reliable CV
- **Handle missing values**: Clean data before optimization
- **Feature scaling**: Some models benefit from normalized features

### Optimization Strategy
- **Start small**: Begin with 20-30 trials to get initial insights
- **Expand gradually**: Increase trials for fine-tuning
- **Use pruning**: Enable Optuna's pruning for faster optimization

### Model-Specific Tips

#### BasicTimeSeriesForecaster
- Focus on `n_lags` parameter
- Consider data frequency when setting lag range
- Cross-validation is crucial for time series

#### XGBoostForecaster
- `learning_rate` and `n_estimators` are most important
- Balance `max_depth` to avoid overfitting
- Regularization helps with noisy data

## 🔍 Interpreting Results

### Key Metrics
- **CV RMSE**: Primary optimization metric (lower is better)
- **CV Standard Deviation**: Measure of model stability
- **R² Score**: Explained variance (higher is better)

### Parameter Analysis
```python
# Analyze parameter importance
import optuna.visualization as vis

# Plot parameter importance
vis.plot_param_importances(study)

# Plot optimization history
vis.plot_optimization_history(study)

# Plot parameter relationships
vis.plot_parallel_coordinate(study)
```

## 🚨 Troubleshooting

### Common Issues

1. **ImportError for models**
   ```bash
   # Solution: Ensure ml-models directory is in Python path
   sys.path.append('./ml-models')
   ```

2. **MLflow tracking errors**
   ```python
   # Solution: Check MLflow server status
   mlflow.set_tracking_uri("./mlruns")  # Use local tracking
   ```

3. **Memory issues with large datasets**
   ```python
   # Solution: Reduce CV folds or data size
   optimizer = Optimizer(cv_folds=3)  # Reduce folds
   data_sample = data.sample(n=1000)  # Sample data
   ```

4. **Slow optimization**
   ```python
   # Solution: Reduce trials or enable pruning
   study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
   ```

### Debugging Tips
- Check MLflow UI for detailed error logs
- Use smaller datasets for initial testing
- Enable verbose logging for detailed output
- Validate model imports before optimization

## 📚 Next Steps

After hyperparameter optimization:

1. **Use Best Parameters**: Apply optimized parameters to your models
2. **Production Deployment**: Deploy best-performing models
3. **Continuous Optimization**: Set up scheduled re-optimization
4. **A/B Testing**: Compare optimized vs baseline models
5. **Feature Engineering**: Optimize feature selection parameters

## 🎉 Success Metrics

Your optimization is successful when you achieve:
- **Improved Performance**: 10-30% reduction in RMSE
- **Stable Results**: Low cross-validation standard deviation
- **Reproducible Models**: Consistent performance across runs
- **Business Value**: Better forecasting accuracy for decision-making

## 📞 Support

For questions or issues:
1. Check MLflow dashboard for detailed logs
2. Review optimization study objects for parameter insights
3. Use demo scripts to verify setup
4. Monitor memory and computation resources

---

*Happy Optimizing! 🚀*
