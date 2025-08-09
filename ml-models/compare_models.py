"""
Model Comparison Script
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Compare performance of all implemented forecasting models:
1. Linear Regression (Basic Forecaster)
2. Moving Averages (Simple ARIMA Forecaster)
3. XGBoost Forecaster
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Import all our forecasters
from basic_forecaster import BasicTimeSeriesForecaster
from simple_arima_forecaster import SimpleMovingAverageForecaster
from xgboost_forecaster import XGBoostTimeSeriesForecaster, create_sample_data


def create_comprehensive_test_data(n_points=200):
    """
    Create more complex test data for comprehensive model comparison.
    """
    return create_sample_data(
        n_points=n_points, 
        noise_level=0.8,  # More noise for realistic challenge
        trend=0.02,       # Stronger trend
        seasonal_period=24
    )


def evaluate_all_models(data, test_ratio=0.2):
    """
    Evaluate all implemented forecasting models on the same dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The complete dataset
    test_ratio : float
        Fraction of data to use for testing
        
    Returns:
    --------
    dict : Results for all models
    """
    print("=" * 70)
    print("COMPREHENSIVE FORECASTING MODEL COMPARISON")
    print("=" * 70)
    
    # Split data
    train_size = int((1 - test_ratio) * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"Dataset: {len(data)} points total")
    print(f"Training: {len(train_data)} points")
    print(f"Testing: {len(test_data)} points")
    print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
    
    results = {}
    
    # 1. Basic Linear Regression Forecaster
    print("\n" + "=" * 50)
    print("1. BASIC LINEAR REGRESSION FORECASTER")
    print("=" * 50)
    
    try:
        basic_forecaster = BasicTimeSeriesForecaster(n_lags=6)
        basic_forecaster.fit(train_data, 'temperature')
        
        # Get predictions
        basic_pred = []
        current_data = train_data.copy()
        
        for i in range(len(test_data)):
            pred = basic_forecaster.predict(current_data, n_steps=1)[0]
            basic_pred.append(pred)
            
            # Add actual value to current_data for next prediction
            actual_value = test_data.iloc[i]['temperature']
            next_row = pd.DataFrame({
                'temperature': [actual_value]
            }, index=[test_data.index[i]])
            current_data = pd.concat([current_data, next_row])
            
            # Keep only recent data
            if len(current_data) > 50:
                current_data = current_data.tail(50)
        
        basic_pred = np.array(basic_pred)
        basic_metrics = basic_forecaster.evaluate_predictions(
            test_data['temperature'].values, basic_pred
        )
        
        results['Basic Linear Regression'] = {
            'predictions': basic_pred,
            'metrics': basic_metrics,
            'model_type': 'Linear Regression with Lag Features'
        }
        
        print(f"✓ Basic Linear Regression RMSE: {basic_metrics['rmse']:.3f}°C")
        print(f"✓ Basic Linear Regression R²: {basic_metrics['r2']:.3f}")
        
    except Exception as e:
        print(f"❌ Basic Linear Regression failed: {e}")
        results['Basic Linear Regression'] = None
    
    # 2. Moving Average Methods
    print("\n" + "=" * 50)
    print("2. MOVING AVERAGE FORECASTERS")
    print("=" * 50)
    
    ma_methods = [
        ('SMA', 'sma', {'window': 6}),
        ('EMA', 'ema', {'alpha': 0.3}),
        ('WMA', 'wma', {'window': 6})
    ]
    
    for method_name, method_code, params in ma_methods:
        try:
            if method_code == 'ema':
                ma_forecaster = SimpleMovingAverageForecaster(
                    method=method_code, alpha=params['alpha']
                )
            else:
                ma_forecaster = SimpleMovingAverageForecaster(
                    method=method_code, window=params['window']
                )
            
            ma_forecaster.fit(train_data, 'temperature')
            
            # Get predictions step by step
            ma_pred = []
            current_data = train_data.copy()
            
            for i in range(len(test_data)):
                pred = ma_forecaster.predict(n_steps=1)[0]
                ma_pred.append(pred)
                
                # Add actual value for next prediction
                actual_value = test_data.iloc[i]['temperature']
                next_row = pd.DataFrame({
                    'temperature': [actual_value]
                }, index=[test_data.index[i]])
                current_data = pd.concat([current_data, next_row])
                
                # Refit with new data point
                if method_code == 'ema':
                    ma_forecaster = SimpleMovingAverageForecaster(
                        method=method_code, alpha=params.get('alpha', 0.3)
                    )
                else:
                    ma_forecaster = SimpleMovingAverageForecaster(
                        method=method_code, window=params.get('window', 6)
                    )
                ma_forecaster.fit(current_data, 'temperature')
                
                # Keep only recent data
                if len(current_data) > 50:
                    current_data = current_data.tail(50)
            
            ma_pred = np.array(ma_pred)
            ma_metrics = ma_forecaster.evaluate(test_data.tail(len(ma_pred)))
            
            results[f'{method_name} Moving Average'] = {
                'predictions': ma_pred,
                'metrics': ma_metrics,
                'model_type': f'{method_name} Moving Average'
            }
            
            print(f"✓ {method_name} RMSE: {ma_metrics['rmse']:.3f}°C, R²: {ma_metrics['r2']:.3f}")
            
        except Exception as e:
            print(f"❌ {method_name} Moving Average failed: {e}")
            results[f'{method_name} Moving Average'] = None
    
    # 3. XGBoost Forecaster
    print("\n" + "=" * 50)
    print("3. XGBOOST FORECASTER")
    print("=" * 50)
    
    try:
        xgb_forecaster = XGBoostTimeSeriesForecaster(
            n_lags=8,
            rolling_windows=[3, 6, 12],
            xgb_params={
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8
            }
        )
        xgb_forecaster.fit(train_data, 'temperature')
        
        # Make predictions
        xgb_pred = xgb_forecaster.predict(
            n_steps=len(test_data), 
            last_known_data=train_data.tail(50)
        )
        
        xgb_metrics = xgb_forecaster.evaluate(test_data, xgb_pred)
        
        results['XGBoost'] = {
            'predictions': xgb_pred,
            'metrics': xgb_metrics,
            'model_type': 'XGBoost with Feature Engineering',
            'n_features': xgb_metrics.get('n_features', 'Unknown')
        }
        
        print(f"✓ XGBoost RMSE: {xgb_metrics['rmse']:.3f}°C")
        print(f"✓ XGBoost R²: {xgb_metrics['r2']:.3f}")
        print(f"✓ Features used: {xgb_metrics.get('n_features', 'Unknown')}")
        
        # Show top features
        try:
            importance_df = xgb_forecaster.get_feature_importance(top_n=5)
            print("✓ Top 5 Features:")
            for idx, row in importance_df.iterrows():
                print(f"   {row['feature']:<20} {row['importance']:.4f}")
        except:
            pass
            
    except Exception as e:
        print(f"❌ XGBoost failed: {e}")
        results['XGBoost'] = None
    
    return results, train_data, test_data


def create_comparison_plots(results, train_data, test_data, save_plots=True):
    """
    Create comprehensive comparison plots.
    """
    print(f"\n{'='*50}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*50}")
    
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("❌ No valid results to plot")
        return
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Forecasting Models Comparison', fontsize=16, fontweight='bold')
    
    # Colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot 1: Time Series with Predictions
    ax1 = axes[0, 0]
    
    # Plot training data
    ax1.plot(train_data.index, train_data['temperature'], 
             label='Training Data', color='gray', alpha=0.7, linewidth=0.8)
    
    # Plot actual test data
    ax1.plot(test_data.index, test_data['temperature'], 
             label='Actual', color='black', linewidth=2)
    
    # Plot predictions for each model
    for i, (model_name, result) in enumerate(valid_results.items()):
        if result and 'predictions' in result:
            pred_data = result['predictions']
            # Use only the data points we have predictions for
            pred_index = test_data.index[:len(pred_data)]
            ax1.plot(pred_index, pred_data, 
                    label=f'{model_name}', 
                    color=colors[i % len(colors)], 
                    linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax1.set_title('Time Series Forecasting Comparison')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RMSE Comparison
    ax2 = axes[0, 1]
    
    model_names = []
    rmse_values = []
    
    for model_name, result in valid_results.items():
        if result and 'metrics' in result and 'rmse' in result['metrics']:
            model_names.append(model_name.replace(' Moving Average', '\nMoving Avg'))
            rmse_values.append(result['metrics']['rmse'])
    
    if rmse_values:
        bars = ax2.bar(model_names, rmse_values, color=colors[:len(rmse_values)], alpha=0.7)
        ax2.set_title('RMSE Comparison (Lower is Better)')
        ax2.set_ylabel('RMSE (°C)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, rmse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: R² Comparison
    ax3 = axes[1, 0]
    
    r2_values = []
    r2_names = []
    
    for model_name, result in valid_results.items():
        if result and 'metrics' in result and 'r2' in result['metrics']:
            r2_names.append(model_name.replace(' Moving Average', '\nMoving Avg'))
            r2_values.append(result['metrics']['r2'])
    
    if r2_values:
        bars = ax3.bar(r2_names, r2_values, color=colors[:len(r2_values)], alpha=0.7)
        ax3.set_title('R² Score Comparison (Higher is Better)')
        ax3.set_ylabel('R² Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, r2_values):
            ax3.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.02 if value >= 0 else -0.05),
                    f'{value:.3f}', ha='center', 
                    va='bottom' if value >= 0 else 'top', fontweight='bold')
    
    # Plot 4: Error Distribution
    ax4 = axes[1, 1]
    
    for i, (model_name, result) in enumerate(valid_results.items()):
        if result and 'predictions' in result:
            pred_data = result['predictions']
            actual_data = test_data['temperature'].values[:len(pred_data)]
            errors = actual_data - pred_data
            
            ax4.hist(errors, bins=15, alpha=0.6, 
                    label=model_name.replace(' Moving Average', ' MA'),
                    color=colors[i % len(colors)])
    
    ax4.set_title('Error Distribution')
    ax4.set_xlabel('Prediction Error (°C)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"results/model_comparison_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {plot_path}")
    
    plt.show()


def create_summary_report(results, train_data, test_data):
    """
    Create a comprehensive summary report.
    """
    print(f"\n{'='*70}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    # Filter valid results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("❌ No valid results to summarize")
        return
    
    # Create summary DataFrame
    summary_data = []
    
    for model_name, result in valid_results.items():
        if result and 'metrics' in result:
            metrics = result['metrics']
            summary_data.append({
                'Model': model_name,
                'RMSE (°C)': f"{metrics.get('rmse', 0):.3f}",
                'MAE (°C)': f"{metrics.get('mae', 0):.3f}",
                'R²': f"{metrics.get('r2', 0):.3f}",
                'MAPE (%)': f"{metrics.get('mape', 0):.2f}",
                'Type': result.get('model_type', 'Unknown')
            })
    
    # Convert to DataFrame and sort by RMSE
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df['RMSE_numeric'] = summary_df['RMSE (°C)'].astype(float)
        summary_df = summary_df.sort_values('RMSE_numeric')
        summary_df = summary_df.drop('RMSE_numeric', axis=1)
        
        print("\nRanked by RMSE (Best to Worst):")
        print("-" * 90)
        print(summary_df.to_string(index=False))
        
        # Best model analysis
        best_model_name = summary_df.iloc[0]['Model']
        best_rmse = summary_df.iloc[0]['RMSE (°C)']
        best_r2 = summary_df.iloc[0]['R²']
        
        print(f"\n🏆 BEST PERFORMING MODEL:")
        print(f"   Model: {best_model_name}")
        print(f"   RMSE: {best_rmse}°C")
        print(f"   R²: {best_r2}")
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        
        summary_file = f"results/model_comparison_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"✓ Summary saved to: {summary_file}")
        
        # Save detailed results
        detailed_results = {
            'dataset_info': {
                'total_points': len(train_data) + len(test_data),
                'train_points': len(train_data),
                'test_points': len(test_data),
                'temperature_range': {
                    'min': float(train_data['temperature'].min()),
                    'max': float(train_data['temperature'].max()),
                    'mean': float(train_data['temperature'].mean()),
                    'std': float(train_data['temperature'].std())
                }
            },
            'model_results': {}
        }
        
        for model_name, result in valid_results.items():
            if result and 'metrics' in result:
                detailed_results['model_results'][model_name] = {
                    'metrics': result['metrics'],
                    'model_type': result.get('model_type', 'Unknown')
                }
        
        results_file = f"results/detailed_comparison_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        print(f"✓ Detailed results saved to: {results_file}")


def main():
    """
    Main comparison function.
    """
    print("Starting comprehensive model comparison...")
    
    # Create test data
    print("Creating comprehensive test dataset...")
    data = create_comprehensive_test_data(n_points=150)  # About 6 days of hourly data
    
    print(f"Dataset created: {len(data)} points")
    print(f"Temperature range: {data['temperature'].min():.2f}°C to {data['temperature'].max():.2f}°C")
    
    # Evaluate all models
    results, train_data, test_data = evaluate_all_models(data, test_ratio=0.25)
    
    # Create plots
    create_comparison_plots(results, train_data, test_data)
    
    # Create summary
    create_summary_report(results, train_data, test_data)
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    results = main()
