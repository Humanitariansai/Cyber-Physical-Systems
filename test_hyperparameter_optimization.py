"""
Simple Hyperparameter Optimization Test
======================================

Test the hyperparameter optimization with just the BasicTimeSeriesForecaster
to demonstrate the functionality.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "ml-models"))

from hyperparameter_optimizer import BasicForecasterOptimizer

def generate_test_data(n_points=300):
    """Generate simple test data"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
    
    # Simple temperature pattern
    time_hours = np.arange(n_points)
    daily_pattern = 5 * np.sin(2 * np.pi * time_hours / 24)  # Daily cycle
    trend = 0.02 * time_hours  # Slight upward trend
    noise = np.random.normal(0, 1.5, n_points)
    
    temperature = 22 + daily_pattern + trend + noise
    
    return pd.DataFrame({
        'timestamp': dates,
        'temperature': temperature
    })

def test_basic_optimization():
    """Test BasicTimeSeriesForecaster optimization"""
    print("🧪 Testing Hyperparameter Optimization")
    print("=" * 50)
    
    # Generate test data
    data = generate_test_data(300)
    print(f" Generated {len(data)} test data points")
    print(f" Temperature range: {data['temperature'].min():.2f} to {data['temperature'].max():.2f}")
    
    # Create optimizer with small number of trials for testing
    optimizer = BasicForecasterOptimizer(
        experiment_name="Test_BasicForecaster_HPO",
        n_trials=10,  # Small number for quick test
        cv_folds=3
    )
    
    try:
        # Run optimization
        print("\n Starting optimization...")
        study = optimizer.optimize(data, target_col='temperature')
        
        print(f"\n Optimization completed successfully!")
        print(f"🏆 Best parameters: {study.best_params}")
        print(f" Best CV RMSE: {study.best_value:.4f}")
        print(f"🔢 Total trials completed: {len(study.trials)}")
        
        # Show parameter importance if available
        if len(study.trials) > 1:
            try:
                importances = optuna.importance.get_param_importances(study)
                print(f" Parameter importances: {importances}")
            except:
                pass
        
        print(f"\n View detailed results at: http://127.0.0.1:5000")
        print(" Look for the 'Test_BasicForecaster_HPO' experiment in MLflow!")
        
        return study
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Import optuna here to avoid import at module level
    import optuna
    
    study = test_basic_optimization()
    
    if study:
        print("\n Hyperparameter optimization test completed successfully!")
        print(" You're ready to use the full optimization system!")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
