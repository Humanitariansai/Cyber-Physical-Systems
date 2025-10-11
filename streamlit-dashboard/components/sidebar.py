"""
Sidebar Component for Streamlit Dashboard
Provides navigation and control elements for the dashboard.
"""

import streamlit as st
from datetime import datetime, timedelta
import time

def render_sidebar():
    """Render the dashboard sidebar with navigation and controls"""
    
    with st.sidebar:
        # Dashboard logo/title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2>🌐 CPS Dashboard</h2>
            <p style="color: #666; font-size: 0.9rem;">Control Center</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # System status indicator
        st.subheader("🔧 System Status")
        
        # Real-time status
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("Uptime", "99.9%")
        with status_col2:
            st.metric("Active", "12/12")
        
        # System health indicator
        health_status = st.selectbox(
            "System Health",
            ["🟢 Healthy", "🟡 Warning", "🔴 Critical"],
            index=0,
            disabled=True
        )
        
        st.divider()
        
        # Data refresh controls
        st.subheader("🔄 Data Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("Auto Refresh", value=True)
        
        if auto_refresh:
            refresh_interval = st.selectbox(
                "Refresh Interval",
                ["30 seconds", "1 minute", "5 minutes", "10 minutes"],
                index=1
            )
            
            # Convert to seconds
            interval_map = {
                "30 seconds": 30,
                "1 minute": 60,
                "5 minutes": 300,
                "10 minutes": 600
            }
            
            interval_seconds = interval_map[refresh_interval]
            
            # Auto-refresh logic
            if 'last_refresh' not in st.session_state:
                st.session_state.last_refresh = time.time()
            
            current_time = time.time()
            if current_time - st.session_state.last_refresh >= interval_seconds:
                st.session_state.last_refresh = current_time
                st.rerun()
            
            # Show countdown
            time_since_refresh = int(current_time - st.session_state.last_refresh)
            time_until_refresh = interval_seconds - time_since_refresh
            
            st.info(f"Next refresh in: {time_until_refresh}s")
        
        # Manual refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.session_state.last_refresh = time.time()
            st.rerun()
        
        st.divider()
        
        # Time range selector
        st.subheader("📅 Time Range")
        
        time_range = st.selectbox(
            "Select Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"],
            index=2
        )
        
        if time_range == "Custom":
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
            end_date = st.date_input("End Date", datetime.now())
            
            # Store custom range in session state
            st.session_state.custom_start = start_date
            st.session_state.custom_end = end_date
        
        # Store selected time range
        st.session_state.selected_time_range = time_range
        
        st.divider()
        
        # Data export
        st.subheader("📊 Data Export")
        
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "JSON", "Excel", "PDF Report"]
        )
        
        if st.button("📥 Export Data", use_container_width=True):
            with st.spinner("Preparing export..."):
                time.sleep(2)  # Simulate export preparation
                st.success(f"Data exported as {export_format}")
        
        st.divider()
        
        # Model controls
        st.subheader("🤖 ML Models")
        
        # Model status
        models_status = {
            "Basic Forecaster": "🟢 Active",
            "XGBoost": "🟢 Active",
            "ARIMA": "🟡 Idle"
        }
        
        for model, status in models_status.items():
            st.text(f"{model}: {status}")
        
        # Quick model actions
        if st.button("🚀 Train New Model", use_container_width=True):
            st.session_state.show_training_modal = True
            st.success("Training initiated!")
        
        if st.button("📈 View MLflow", use_container_width=True):
            st.info("Opening MLflow dashboard...")
        
        st.divider()
        
        # Alerts and notifications
        st.subheader("🔔 Alerts")
        
        # Recent alerts
        alerts = [
            {"time": "2 min ago", "type": "info", "message": "Model training completed"},
            {"time": "15 min ago", "type": "warning", "message": "High CPU usage detected"},
            {"time": "1 hr ago", "type": "success", "message": "System backup completed"}
        ]
        
        for alert in alerts:
            alert_type_icons = {
                "info": "ℹ️",
                "warning": "⚠️",
                "success": "✅",
                "error": "❌"
            }
            
            icon = alert_type_icons.get(alert["type"], "📄")
            
            with st.expander(f"{icon} {alert['time']}", expanded=False):
                st.write(alert["message"])
        
        st.divider()
        
        # Dashboard settings
        st.subheader("⚙️ Settings")
        
        # Theme selector
        theme = st.selectbox(
            "Dashboard Theme",
            ["Light", "Dark", "Auto"],
            index=0
        )
        
        # Chart type preferences
        chart_style = st.selectbox(
            "Chart Style",
            ["Modern", "Classic", "Minimal"],
            index=0
        )
        
        # Notification settings
        notifications_enabled = st.toggle("Enable Notifications", value=True)
        
        if notifications_enabled:
            notification_types = st.multiselect(
                "Notification Types",
                ["System Alerts", "Model Updates", "Data Quality", "Performance"],
                default=["System Alerts", "Model Updates"]
            )
        
        # Save settings
        if st.button("💾 Save Settings", use_container_width=True):
            # Store settings in session state
            st.session_state.dashboard_settings = {
                "theme": theme,
                "chart_style": chart_style,
                "notifications_enabled": notifications_enabled,
                "notification_types": notification_types if notifications_enabled else []
            }
            st.success("Settings saved!")
        
        st.divider()
        
        # Footer information
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem; padding: 1rem 0;">
            <p>CPS Dashboard v1.0</p>
            <p>© 2025 Udisha Dutta Chowdhury</p>
            <p>Last updated: {}</p>
        </div>
        """.format(datetime.now().strftime("%H:%M:%S")), unsafe_allow_html=True)

def get_time_range_dates(time_range):
    """
    Convert time range selection to actual dates
    
    Args:
        time_range (str): Selected time range
    
    Returns:
        tuple: (start_date, end_date)
    """
    end_date = datetime.now()
    
    if time_range == "Last Hour":
        start_date = end_date - timedelta(hours=1)
    elif time_range == "Last 6 Hours":
        start_date = end_date - timedelta(hours=6)
    elif time_range == "Last 24 Hours":
        start_date = end_date - timedelta(days=1)
    elif time_range == "Last 7 Days":
        start_date = end_date - timedelta(days=7)
    elif time_range == "Last 30 Days":
        start_date = end_date - timedelta(days=30)
    elif time_range == "Custom":
        if hasattr(st.session_state, 'custom_start') and hasattr(st.session_state, 'custom_end'):
            start_date = datetime.combine(st.session_state.custom_start, datetime.min.time())
            end_date = datetime.combine(st.session_state.custom_end, datetime.max.time())
        else:
            start_date = end_date - timedelta(days=7)
    else:
        start_date = end_date - timedelta(days=1)
    
    return start_date, end_date

def show_training_modal():
    """Show model training configuration modal"""
    if st.session_state.get('show_training_modal', False):
        with st.modal("🚀 Start Model Training"):
            st.subheader("Training Configuration")
            
            model_type = st.selectbox(
                "Model Type",
                ["Basic Forecaster", "XGBoost", "ARIMA", "Neural Network"]
            )
            
            if model_type == "XGBoost":
                n_estimators = st.slider("N Estimators", 50, 500, 100)
                max_depth = st.slider("Max Depth", 3, 15, 6)
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
            
            training_data_size = st.slider("Training Data Size (%)", 60, 90, 80)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Start Training", type="primary"):
                    st.session_state.show_training_modal = False
                    st.success("Training started!")
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.show_training_modal = False
                    st.rerun()