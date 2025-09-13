"""
MLflow Dashboard Setup and Configuration
=======================================

This script provides utilities for setting up and managing MLflow dashboard
for comprehensive experiment tracking and model comparison.
"""

import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import mlflow
from mlflow_tracking import ExperimentTracker


def setup_mlflow_server(host="127.0.0.1", port=5000, backend_store_uri=None, default_artifact_root=None):
    """
    Setup and start MLflow tracking server.
    
    Args:
        host (str): Host address for MLflow server
        port (int): Port number for MLflow server
        backend_store_uri (str): Backend store URI (default: local file store)
        default_artifact_root (str): Default artifact root (default: local folder)
    
    Returns:
        dict: Server configuration
    """
    # Set default paths if not provided
    if backend_store_uri is None:
        mlruns_path = Path(__file__).parent / "mlruns"
        mlruns_path.mkdir(exist_ok=True)
        backend_store_uri = f"file://{mlruns_path.absolute()}"
    
    if default_artifact_root is None:
        artifacts_path = Path(__file__).parent / "mlflow-artifacts"
        artifacts_path.mkdir(exist_ok=True)
        default_artifact_root = str(artifacts_path.absolute())
    
    config = {
        "host": host,
        "port": port,
        "backend_store_uri": backend_store_uri,
        "default_artifact_root": default_artifact_root
    }
    
    print("MLflow Server Configuration:")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Backend Store: {backend_store_uri}")
    print(f"  Artifact Root: {default_artifact_root}")
    print(f"  URL: http://{host}:{port}")
    
    return config


def start_mlflow_ui(config=None, background=True):
    """
    Start MLflow UI server.
    
    Args:
        config (dict): MLflow server configuration
        background (bool): Whether to run in background
    
    Returns:
        subprocess.Popen or None: Process object if background=True
    """
    if config is None:
        config = setup_mlflow_server()
    
    # Build MLflow UI command
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--host", config["host"],
        "--port", str(config["port"]),
        "--backend-store-uri", config["backend_store_uri"],
        "--default-artifact-root", config["default_artifact_root"]
    ]
    
    print(f"Starting MLflow UI...")
    print(f"Command: {' '.join(cmd)}")
    
    if background:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f" MLflow UI started in background (PID: {process.pid})")
        print(f"🔗 Access dashboard at: http://{config['host']}:{config['port']}")
        return process
    else:
        subprocess.run(cmd)
        return None


def get_experiment_summary(experiment_name="forecasting-models"):
    """
    Get summary of all runs in an experiment.
    
    Args:
        experiment_name (str): Name of the experiment
    
    Returns:
        pd.DataFrame: Summary of all runs
    """
    try:
        # Set up MLflow tracking
        tracker = ExperimentTracker(experiment_name)
        
        # Get experiment results
        runs_df = tracker.get_experiment_results()
        
        if runs_df.empty:
            print(f"No runs found in experiment '{experiment_name}'")
            return pd.DataFrame()
        
        # Select relevant columns
        summary_cols = [
            'run_id', 'experiment_id', 'status', 'start_time', 'end_time',
            'tags.model_type', 'metrics.rmse', 'metrics.mae', 'metrics.r2',
            'params.n_lags', 'params.model_type'
        ]
        
        # Filter existing columns
        available_cols = [col for col in summary_cols if col in runs_df.columns]
        summary_df = runs_df[available_cols].copy()
        
        # Sort by RMSE if available
        if 'metrics.rmse' in summary_df.columns:
            summary_df = summary_df.sort_values('metrics.rmse')
        
        return summary_df
        
    except Exception as e:
        print(f"Error getting experiment summary: {e}")
        return pd.DataFrame()


def generate_experiment_report(experiment_name="forecasting-models", output_file=None):
    """
    Generate a detailed experiment report.
    
    Args:
        experiment_name (str): Name of the experiment
        output_file (str): Output file path (optional)
    
    Returns:
        str: Report content
    """
    summary_df = get_experiment_summary(experiment_name)
    
    if summary_df.empty:
        return "No experiment data available."
    
    # Generate report
    report_lines = []
    report_lines.append("MLflow Experiment Report")
    report_lines.append("=" * 50)
    report_lines.append(f"Experiment: {experiment_name}")
    report_lines.append(f"Generated: {pd.Timestamp.now()}")
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("Summary Statistics:")
    report_lines.append(f"  Total Runs: {len(summary_df)}")
    
    if 'metrics.rmse' in summary_df.columns:
        rmse_values = summary_df['metrics.rmse'].dropna()
        if not rmse_values.empty:
            report_lines.append(f"  Best RMSE: {rmse_values.min():.4f}")
            report_lines.append(f"  Worst RMSE: {rmse_values.max():.4f}")
            report_lines.append(f"  Average RMSE: {rmse_values.mean():.4f}")
    
    report_lines.append("")
    
    # Model comparison
    if 'tags.model_type' in summary_df.columns and 'metrics.rmse' in summary_df.columns:
        model_comparison = summary_df.groupby('tags.model_type')['metrics.rmse'].agg(['count', 'min', 'mean']).round(4)
        report_lines.append("Model Comparison (by RMSE):")
        report_lines.append(model_comparison.to_string())
        report_lines.append("")
    
    # Detailed runs
    report_lines.append("Detailed Runs:")
    report_lines.append(summary_df.to_string(index=False))
    
    report_content = "\n".join(report_lines)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_content)
        print(f"Report saved to: {output_file}")
    
    return report_content


def cleanup_old_runs(experiment_name="forecasting-models", keep_best_n=10):
    """
    Clean up old experiment runs, keeping only the best N runs.
    
    Args:
        experiment_name (str): Name of the experiment
        keep_best_n (int): Number of best runs to keep
    
    Returns:
        list: List of deleted run IDs
    """
    try:
        summary_df = get_experiment_summary(experiment_name)
        
        if summary_df.empty or len(summary_df) <= keep_best_n:
            print("No runs to cleanup.")
            return []
        
        # Sort by RMSE and keep best N
        if 'metrics.rmse' in summary_df.columns:
            sorted_df = summary_df.sort_values('metrics.rmse')
            runs_to_delete = sorted_df.iloc[keep_best_n:]
            
            deleted_runs = []
            for _, run in runs_to_delete.iterrows():
                run_id = run['run_id']
                try:
                    mlflow.delete_run(run_id)
                    deleted_runs.append(run_id)
                    print(f"Deleted run: {run_id}")
                except Exception as e:
                    print(f"Failed to delete run {run_id}: {e}")
            
            print(f"Cleanup completed. Deleted {len(deleted_runs)} runs.")
            return deleted_runs
        else:
            print("Cannot cleanup - no RMSE metrics found.")
            return []
            
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return []


def create_mlflow_commands_script():
    """
    Create a script with useful MLflow commands.
    
    Returns:
        str: Script file path
    """
    script_content = """#!/bin/bash
# MLflow Management Commands
# =========================

# Start MLflow UI (local)
echo "Starting MLflow UI on http://localhost:5000"
mlflow ui --host 0.0.0.0 --port 5000

# Alternative: Start with specific backend store
# mlflow ui --backend-store-uri file:./mlruns --default-artifact-root ./mlflow-artifacts

# List all experiments
echo "Listing all experiments:"
mlflow experiments list

# Search runs with specific criteria
echo "Searching runs with RMSE < 2.0:"
mlflow runs list --experiment-name "forecasting-models" --filter "metrics.rmse < 2.0"

# Export experiment data
echo "Exporting experiment data:"
mlflow experiments csv --experiment-name "forecasting-models" --filename experiment_export.csv

# Compare runs
echo "Comparing top 5 runs:"
mlflow runs compare --experiment-name "forecasting-models" --order-by "metrics.rmse ASC" --max-results 5

# Delete old runs (be careful!)
# mlflow gc --backend-store-uri file:./mlruns

echo "MLflow commands completed!"
"""
    
    script_file = "mlflow_commands.sh"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    try:
        os.chmod(script_file, 0o755)
    except:
        pass
    
    print(f"MLflow commands script created: {script_file}")
    return script_file


def main():
    """
    Main function for MLflow dashboard setup.
    """
    print(" MLflow Dashboard Setup")
    print("=" * 40)
    
    # Setup server configuration
    config = setup_mlflow_server()
    
    # Create management commands script
    create_mlflow_commands_script()
    
    # Get current experiment summary
    print("\n Current Experiment Summary:")
    summary_df = get_experiment_summary()
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No experiments found. Run experiments first!")
    
    # Generate report
    print("\n Generating experiment report...")
    report = generate_experiment_report(output_file="mlflow_experiment_report.txt")
    
    print("\n Setup completed!")
    print(f" To start MLflow UI, run: python -c \"from mlflow_dashboard import start_mlflow_ui; start_mlflow_ui()\"")
    print(f"🔗 Or manually run: mlflow ui --host {config['host']} --port {config['port']}")


if __name__ == "__main__":
    main()
