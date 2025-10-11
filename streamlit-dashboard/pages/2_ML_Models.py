"""
ML Model Management Page for Streamlit Dashboard
Provides model training, evaluation, and monitoring capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import time

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "streamlit-dashboard"))

from utils.ml_integration import MLModelManager
from utils.data_loader import DataLoader

st.set_page_config(
    page_title="ML Models - CPS Dashboard",
    page_icon=None,  # Removed emoji
    layout="wide"
)

def main():
    """Main ML model management page"""
    
    # Initialize managers
    if 'ml_manager' not in st.session_state:
        st.session_state.ml_manager = MLModelManager()
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    st.title("ML Model Management")
    st.markdown("Train, evaluate, and monitor machine learning models")
    
    # Sidebar
    with st.sidebar:
        st.subheader("Model Controls")
        
        page_mode = st.selectbox(
            "Page Mode",
            ["Model Overview", "Model Training", "Model Evaluation", "Hyperparameter Tuning", "Model Monitoring"]
        )
        
        # Model selection with defensive handling
        available_models = st.session_state.ml_manager.get_available_models()
        if available_models:
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                index=0
            )
        else:
            st.info("No models available")
            selected_model = None
        
        if st.button("Refresh Models"):
            st.session_state.ml_manager._discover_models()
            st.rerun()
    
    # Main content based on page mode
    if page_mode == "Model Overview":
        render_model_overview()
    elif page_mode == "Model Training":
        render_model_training()
    elif page_mode == "Model Evaluation":
        render_model_evaluation(selected_model)
    elif page_mode == "Hyperparameter Tuning":
        render_hyperparameter_tuning()
    elif page_mode == "Model Monitoring":
        render_model_monitoring()

def render_model_overview():
    """Render model overview"""
    st.subheader("Model Overview")
    
    # Model status cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", "3", "+1")
    
    with col2:
        st.metric("Active Models", "2", "0")
    
    with col3:
        st.metric("Training Jobs", "1", "+1")
    
    with col4:
        st.metric("Best Accuracy", "94.2%", "+1.3%")
    
    # Model comparison table
    st.subheader("Model Comparison")
    
    performance_data = st.session_state.ml_manager.get_model_performance()
    
    # Convert performance data to DataFrame with defensive handling
    model_rows = []
    for model_name, metrics in performance_data.items():
        model_info = st.session_state.ml_manager.get_model_info(model_name)
        
        model_rows.append({
            "Model": model_name.replace('_', ' ').title(),
            "Status": "Active" if model_info and model_info.get('loaded') else "Idle",
            "RMSE": f"{metrics.get('rmse', 0):.3f}",
            "MAE": f"{metrics.get('mae', 0):.3f}",
            "R²": f"{metrics.get('r2', 0):.3f}",
            "Last Updated": model_info.get('last_modified', datetime.now()).strftime("%Y-%m-%d %H:%M") if model_info and model_info.get('last_modified') else "N/A",
            "Actions": "Train | Evaluate | Tune"
        })
    
    if model_rows:
        models_df = pd.DataFrame(model_rows)
        st.dataframe(models_df, use_container_width=True, hide_index=True)
    else:
        st.info("No models found. Start by training a new model.")
    
    # Performance comparison chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Comparison")
        
        if model_rows:
            # Create performance comparison chart
            metrics = ['RMSE', 'MAE', 'R²']
            fig = go.Figure()
            
            for metric in metrics:
                values = [float(row[metric]) for row in model_rows]
                model_names = [row['Model'] for row in model_rows]
                
                fig.add_trace(go.Scatter(
                    x=model_names,
                    y=values,
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Model Performance Metrics",
                xaxis_title="Models",
                yaxis_title="Metric Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Accuracy Trends")
        
        # Generate sample accuracy trends over time
        dates = pd.date_range(start=datetime.now()-timedelta(days=30), end=datetime.now(), freq='D')
        
        fig = go.Figure()
        
        for model_name in performance_data.keys():
            # Generate sample accuracy trend
            base_accuracy = performance_data[model_name].get('r2', 0.8)
            trend = base_accuracy + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.02, len(dates))
            trend = np.clip(trend, 0, 1)  # Keep between 0 and 1
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=trend,
                mode='lines',
                name=model_name.replace('_', ' ').title(),
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Accuracy Trends (30 Days)",
            xaxis_title="Date",
            yaxis_title="R² Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_model_training():
    """Render model training interface"""
    st.subheader("Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Training configuration
        st.subheader("Training Configuration")
        
        model_type = st.selectbox(
            "Model Type",
            ["Basic Forecaster", "XGBoost", "ARIMA", "Neural Network", "Random Forest"]
        )
        
        # Model-specific parameters
        if model_type == "Basic Forecaster":
            st.subheader("Basic Forecaster Parameters")
            n_lags = st.slider("Number of Lags", 1, 20, 5)
            
        elif model_type == "XGBoost":
            st.subheader("XGBoost Parameters")
            col_a, col_b = st.columns(2)
            with col_a:
                n_estimators = st.slider("N Estimators", 50, 500, 100)
                max_depth = st.slider("Max Depth", 3, 15, 6)
            with col_b:
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
                subsample = st.slider("Subsample", 0.5, 1.0, 0.8)
        
        elif model_type == "Neural Network":
            st.subheader("Neural Network Parameters")
            col_a, col_b = st.columns(2)
            with col_a:
                hidden_layers = st.slider("Hidden Layers", 1, 5, 2)
                neurons_per_layer = st.slider("Neurons per Layer", 16, 256, 64)
            with col_b:
                activation = st.selectbox("Activation", ["relu", "tanh", "sigmoid"])
                optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
        
        # Data configuration
        st.subheader("Data Configuration")
        col_a, col_b = st.columns(2)
        
        with col_a:
            train_size = st.slider("Training Data (%)", 60, 90, 80)
            validation_size = st.slider("Validation Data (%)", 5, 20, 10)
        
        with col_b:
            test_size = 100 - train_size - validation_size
            st.metric("Test Data (%)", f"{test_size}")
            
            data_source = st.selectbox(
                "Data Source",
                ["Temperature", "Humidity", "Pressure", "Multi-variate"]
            )
        
        # Advanced options
        with st.expander("Advanced Options"):
            cross_validation = st.checkbox("Enable Cross-Validation", value=True)
            if cross_validation:
                cv_folds = st.slider("CV Folds", 3, 10, 5)
            
            early_stopping = st.checkbox("Enable Early Stopping", value=True)
            if early_stopping:
                patience = st.slider("Patience", 5, 50, 10)
            
            feature_engineering = st.multiselect(
                "Feature Engineering",
                ["Lag Features", "Rolling Statistics", "Fourier Transform", "Polynomial Features"],
                default=["Lag Features"]
            )
        
        # Start training
        col_train, col_schedule = st.columns(2)
        
        with col_train:
            if st.button("Start Training"):
                start_training_job(model_type, {
                    'n_lags': n_lags if model_type == "Basic Forecaster" else None,
                    'n_estimators': n_estimators if model_type == "XGBoost" else None,
                    'train_size': train_size / 100,
                    'data_source': data_source
                })
        
        with col_schedule:
            if st.button("Schedule Training"):
                st.info("Training scheduled for later execution")
    
    with col2:
        st.subheader("Training Status")
        
        # Active training jobs
        if 'training_jobs' not in st.session_state:
            st.session_state.training_jobs = []
        
        if st.session_state.training_jobs:
            for job in st.session_state.training_jobs:
                with st.container():
                    st.write(f"**{job['model']} Training**")
                    
                    # Progress bar
                    progress = min(job.get('progress', 0), 100)
                    st.progress(progress / 100)
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.write(f"Status: {job['status']}")
                        st.write(f"Epoch: {job.get('epoch', 0)}/{job.get('total_epochs', 100)}")
                    
                    with col_info2:
                        st.write(f"Loss: {job.get('loss', 'N/A')}")
                        st.write(f"ETA: {job.get('eta', 'Calculating...')}")
                    
                    if st.button("Stop", key=f"stop_{job['id']}"):
                        stop_training_job(job['id'])
                    
                    st.markdown("---")
        else:
            st.info("No active training jobs")
        
        # Recent completions
        st.subheader("Recent Completions")
        
        completed_jobs = [
            {"model": "XGBoost", "time": "2 hours ago", "accuracy": "94.2%"},
            {"model": "Basic Forecaster", "time": "5 hours ago", "accuracy": "89.1%"},
            {"model": "ARIMA", "time": "1 day ago", "accuracy": "83.5%"}
        ]
        
        for job in completed_jobs:
            st.write(f"**{job['model']}** - {job['time']}")
            st.write(f"Accuracy: {job['accuracy']}")
            st.markdown("---")

def render_model_evaluation(selected_model):
    """Render model evaluation"""
    if not selected_model:
        st.warning("Please select a model from the sidebar")
        return
        
    st.subheader(f"Model Evaluation: {selected_model}")
    
    # Evaluation metrics
    col1, col2, col3 = st.columns(3)
    
    # Get model performance with defensive handling
    performance = st.session_state.ml_manager.get_model_performance(selected_model)
    metrics = {}
    
    if isinstance(performance, dict):
        if selected_model in performance:
            metrics = performance[selected_model]
        else:
            metrics = performance
    
    with col1:
        st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
    
    with col2:
        st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
    
    with col3:
        st.metric("R²", f"{metrics.get('r2', 0):.3f}")
    
    # Evaluation plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predictions vs Actual")
        
        # Generate sample evaluation data
        n_points = 100
        actual = 20 + 5 * np.sin(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 1, n_points)
        
        # Add some model error
        model_error = np.random.normal(0, metrics.get('rmse', 2), n_points)
        predicted = actual + model_error
        
        fig = go.Figure()
        
        # Perfect prediction line
        fig.add_trace(go.Scatter(
            x=actual,
            y=actual,
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        # Actual vs Predicted
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(size=6, opacity=0.7)
        ))
        
        fig.update_layout(
            title="Predictions vs Actual Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Residual Analysis")
        
        residuals = predicted - actual
        
        # Residual plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=actual,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(size=6, opacity=0.7)
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="Residual Plot",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series evaluation
    st.subheader("Time Series Performance")
    
    # Generate time series evaluation
    dates = pd.date_range(start=datetime.now()-timedelta(days=30), end=datetime.now(), freq='D')
    actual_ts = 20 + 5 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 1, len(dates))
    predicted_ts = actual_ts + np.random.normal(0, metrics.get('rmse', 2), len(dates))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_ts,
        mode='lines',
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted_ts,
        mode='lines',
        name='Predicted',
        line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        title="Time Series Predictions",
        xaxis_title="Date",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Evaluation actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Re-evaluate"):
            st.success("Model re-evaluation started")
    
    with col2:
        if st.button("Generate Report"):
            generate_evaluation_report(selected_model)
    
    with col3:
        if st.button("Deploy Model"):
            st.success("Model deployment initiated")

def render_hyperparameter_tuning():
    """Render hyperparameter tuning interface"""
    st.subheader("Hyperparameter Tuning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tuning configuration
        model_type = st.selectbox(
            "Model for Tuning",
            ["XGBoost", "Neural Network", "Random Forest", "SVM"]
        )
        
        tuning_method = st.selectbox(
            "Tuning Method",
            ["Grid Search", "Random Search", "Bayesian Optimization", "Optuna"]
        )
        
        # Parameter ranges
        st.subheader("Parameter Ranges")
        
        if model_type == "XGBoost":
            col_a, col_b = st.columns(2)
            
            with col_a:
                n_estimators_range = st.slider(
                    "N Estimators Range",
                    value=[50, 200],
                    min_value=10,
                    max_value=500
                )
                
                max_depth_range = st.slider(
                    "Max Depth Range",
                    value=[3, 10],
                    min_value=1,
                    max_value=20
                )
            
            with col_b:
                learning_rate_range = st.slider(
                    "Learning Rate Range",
                    value=[0.01, 0.3],
                    min_value=0.001,
                    max_value=1.0
                )
                
                subsample_range = st.slider(
                    "Subsample Range",
                    value=[0.6, 1.0],
                    min_value=0.1,
                    max_value=1.0
                )
        
        # Tuning settings
        col_a, col_b = st.columns(2)
        
        with col_a:
            n_trials = st.slider("Number of Trials", 10, 200, 50)
            cv_folds = st.slider("CV Folds", 3, 10, 5)
        
        with col_b:
            timeout = st.slider("Timeout (minutes)", 10, 300, 60)
            n_jobs = st.slider("Parallel Jobs", 1, 8, 4)
        
        # Start tuning
        if st.button("Start Hyperparameter Tuning"):
            start_hyperparameter_tuning(model_type, tuning_method, {
                'n_trials': n_trials,
                'cv_folds': cv_folds,
                'timeout': timeout
            })
    
    with col2:
        st.subheader("Tuning Progress")
        
        # Mock tuning progress
        if 'tuning_active' in st.session_state and st.session_state.tuning_active:
            progress = st.session_state.get('tuning_progress', 0)
            st.progress(progress / 100)
            
            st.write(f"Trial: {int(progress * n_trials / 100)}/{n_trials}")
            st.write(f"Best Score: {0.85 + progress/1000:.4f}")
            st.write(f"Current Trial Score: {np.random.uniform(0.7, 0.95):.4f}")
            
            if progress >= 100:
                st.success("Hyperparameter tuning completed!")
                st.session_state.tuning_active = False
        else:
            st.info("No active tuning job")
        
        # Best parameters history
        st.subheader("Best Parameters")
        
        if model_type == "XGBoost":
            best_params = {
                "n_estimators": 150,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "score": 0.942
            }
            
            for param, value in best_params.items():
                if param != "score":
                    st.write(f"**{param}**: {value}")
            
            st.metric("Best CV Score", f"{best_params['score']:.3f}")

def render_model_monitoring():
    """Render model monitoring dashboard"""
    st.subheader("Model Monitoring")
    
    # Model health overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Uptime", "99.9%", "+0.1%")
    
    with col2:
        st.metric("Predictions/Hour", "245", "+12")
    
    with col3:
        st.metric("Avg Response Time", "120ms", "-5ms")
    
    with col4:
        st.metric("Error Rate", "0.02%", "-0.01%")
    
    # Monitoring charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Volume")
        
        # Generate hourly prediction volume
        hours = list(range(24))
        volumes = [20 + 15 * np.sin(h * np.pi / 12) + np.random.randint(-5, 6) for h in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hours,
            y=volumes,
            name='Predictions per Hour',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Hourly Prediction Volume",
            xaxis_title="Hour of Day",
            yaxis_title="Predictions",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Response Times")
        
        # Generate response time data
        timestamps = pd.date_range(end=datetime.now(), periods=24, freq='H')
        response_times = 100 + 50 * np.sin(np.linspace(0, 2*np.pi, 24)) + np.random.normal(0, 10, 24)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=response_times,
            mode='lines+markers',
            name='Response Time',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title="Model Response Times",
            xaxis_title="Time",
            yaxis_title="Response Time (ms)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model drift detection
    st.subheader("Model Drift Detection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Feature drift visualization
        features = ['temperature', 'humidity', 'pressure']
        drift_scores = np.random.uniform(0, 1, len(features))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=features,
            y=drift_scores,
            marker_color=['red' if score > 0.7 else 'yellow' if score > 0.4 else 'green' for score in drift_scores],
            name='Drift Score'
        ))
        
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Drift Threshold")
        fig.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="Medium Drift Threshold")
        
        fig.update_layout(
            title="Feature Drift Detection",
            xaxis_title="Features",
            yaxis_title="Drift Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Drift Alerts")
        
        for i, (feature, score) in enumerate(zip(features, drift_scores)):
            if score > 0.7:
                alert_type = "High"
            elif score > 0.4:
                alert_type = "Medium"
            else:
                alert_type = "Low"
            
            st.write(f"**{feature}**: {alert_type}")
            st.write(f"Score: {score:.3f}")
            st.markdown("---")
        
        if st.button("Retrain Model"):
            st.success("Model retraining initiated")

def start_training_job(model_type, config):
    """Start a training job (simulated)"""
    if 'training_jobs' not in st.session_state:
        st.session_state.training_jobs = []
    
    job = {
        'id': len(st.session_state.training_jobs),
        'model': model_type,
        'status': 'Starting...',
        'progress': 0,
        'epoch': 0,
        'total_epochs': 100,
        'loss': 'N/A',
        'eta': 'Calculating...',
        'config': config
    }
    
    st.session_state.training_jobs.append(job)
    st.success(f"Training job started for {model_type}")
    
    # Simulate progress (in a real app, this would be handled differently)
    time.sleep(1)
    job['status'] = 'Training...'
    job['progress'] = 5

def stop_training_job(job_id):
    """Stop a training job"""
    if 'training_jobs' in st.session_state:
        for job in st.session_state.training_jobs:
            if job['id'] == job_id:
                job['status'] = 'Stopped'
                st.warning(f"Training job {job_id} stopped")
                break

def start_hyperparameter_tuning(model_type, method, config):
    """Start hyperparameter tuning (simulated)"""
    st.session_state.tuning_active = True
    st.session_state.tuning_progress = 0
    st.success(f"Hyperparameter tuning started for {model_type} using {method}")

def generate_evaluation_report(model_name):
    """Generate evaluation report"""
    st.success(f"Evaluation report generated for {model_name}")
    
    # In a real app, this would generate and download a PDF report
    with st.expander("Report Preview"):
        st.markdown(f"""
        # Model Evaluation Report: {model_name}
        
        ## Summary
        - **Model Type**: {model_name}
        - **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - **Dataset Size**: 10,000 samples
        
        ## Performance Metrics
        - **RMSE**: 1.87
        - **MAE**: 1.43
        - **R²**: 0.94
        
        ## Recommendations
        - Model performance is excellent
        - Consider deployment to production
        - Monitor for data drift
        """)

if __name__ == "__main__":
    main()