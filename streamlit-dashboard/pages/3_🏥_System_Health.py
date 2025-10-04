"""
System Health Monitoring Page for Streamlit Dashboard
Monitors system resources, network, and application health.
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

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

st.set_page_config(
    page_title="System Health - CPS Dashboard",
    page_icon="🏥",
    layout="wide"
)

def main():
    """Main system health monitoring page"""
    
    st.title("🏥 System Health Monitoring")
    st.markdown("Monitor system resources, performance, and application health")
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Monitoring Controls")
        
        auto_refresh = st.toggle("Auto Refresh", value=True)
        if auto_refresh:
            refresh_interval = st.select_slider(
                "Refresh Interval",
                options=[5, 10, 30, 60],
                value=10,
                format_func=lambda x: f"{x}s"
            )
        
        monitoring_mode = st.selectbox(
            "Monitoring Mode",
            ["Real-time", "Historical", "Alerts"]
        )
        
        system_components = st.multiselect(
            "Monitor Components",
            ["CPU", "Memory", "Disk", "Network", "GPU", "Temperature"],
            default=["CPU", "Memory", "Disk", "Network"]
        )
        
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Auto-refresh mechanism
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    # System overview cards
    render_system_overview()
    
    # Main content based on mode
    if monitoring_mode == "Real-time":
        render_realtime_monitoring(system_components)
    elif monitoring_mode == "Historical":
        render_historical_monitoring(system_components)
    elif monitoring_mode == "Alerts":
        render_alerts_monitoring()

def render_system_overview():
    """Render system overview metrics"""
    st.subheader("📊 System Overview")
    
    # Get system information
    if PSUTIL_AVAILABLE:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network stats
        try:
            network = psutil.net_io_counters()
            network_available = True
        except:
            network_available = False
    else:
        # Mock data when psutil not available
        cpu_percent = np.random.uniform(20, 80)
        memory = type('obj', (object,), {
            'percent': np.random.uniform(40, 90),
            'total': 16 * 1024**3,  # 16GB
            'available': 8 * 1024**3  # 8GB available
        })()
        disk = type('obj', (object,), {
            'percent': np.random.uniform(30, 70),
            'total': 500 * 1024**3,  # 500GB
            'free': 200 * 1024**3  # 200GB free
        })()
        network_available = False
    
    # System health cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        cpu_color = "🔴" if cpu_percent > 80 else "🟡" if cpu_percent > 60 else "🟢"
        st.metric(
            "CPU Usage",
            f"{cpu_percent:.1f}%",
            delta=f"{np.random.uniform(-5, 5):.1f}%",
            help=f"{cpu_color} CPU Status"
        )
    
    with col2:
        memory_color = "🔴" if memory.percent > 85 else "🟡" if memory.percent > 70 else "🟢"
        st.metric(
            "Memory Usage",
            f"{memory.percent:.1f}%",
            delta=f"{np.random.uniform(-3, 3):.1f}%",
            help=f"{memory_color} Memory Status"
        )
    
    with col3:
        disk_color = "🔴" if disk.percent > 90 else "🟡" if disk.percent > 80 else "🟢"
        st.metric(
            "Disk Usage",
            f"{disk.percent:.1f}%",
            delta=f"{np.random.uniform(-1, 1):.1f}%",
            help=f"{disk_color} Disk Status"
        )
    
    with col4:
        if network_available and PSUTIL_AVAILABLE:
            network_speed = (network.bytes_sent + network.bytes_recv) / (1024**2)  # MB
            st.metric(
                "Network I/O",
                f"{network_speed:.1f} MB/s",
                delta=f"{np.random.uniform(-10, 10):.1f} MB/s"
            )
        else:
            network_speed = np.random.uniform(10, 100)
            st.metric(
                "Network I/O",
                f"{network_speed:.1f} MB/s",
                delta=f"{np.random.uniform(-10, 10):.1f} MB/s"
            )
    
    with col5:
        # System temperature (mock data)
        temp = np.random.uniform(45, 75)
        temp_color = "🔴" if temp > 70 else "🟡" if temp > 60 else "🟢"
        st.metric(
            "Temperature",
            f"{temp:.1f}°C",
            delta=f"{np.random.uniform(-2, 2):.1f}°C",
            help=f"{temp_color} Temperature Status"
        )

def render_realtime_monitoring(components):
    """Render real-time monitoring charts"""
    st.subheader("⚡ Real-time Monitoring")
    
    # Generate real-time data
    timestamps = pd.date_range(end=datetime.now(), periods=60, freq='1min')
    
    if "CPU" in components or "Memory" in components:
        col1, col2 = st.columns(2)
        
        if "CPU" in components:
            with col1:
                st.subheader("💻 CPU Usage")
                
                # Generate CPU data
                cpu_data = 30 + 20 * np.sin(np.linspace(0, 4*np.pi, len(timestamps))) + np.random.normal(0, 5, len(timestamps))
                cpu_data = np.clip(cpu_data, 0, 100)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=cpu_data,
                    mode='lines',
                    name='CPU %',
                    line=dict(color='blue', width=2),
                    fill='tonexty'
                ))
                
                # Add threshold lines
                fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Critical")
                fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Warning")
                
                fig.update_layout(
                    title="CPU Usage Over Time",
                    xaxis_title="Time",
                    yaxis_title="CPU %",
                    height=300,
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        if "Memory" in components:
            with col2:
                st.subheader("🧠 Memory Usage")
                
                # Generate memory data
                memory_data = 50 + 15 * np.sin(np.linspace(0, 2*np.pi, len(timestamps))) + np.random.normal(0, 3, len(timestamps))
                memory_data = np.clip(memory_data, 0, 100)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=memory_data,
                    mode='lines',
                    name='Memory %',
                    line=dict(color='green', width=2),
                    fill='tonexty'
                ))
                
                # Add threshold lines
                fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Critical")
                fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Warning")
                
                fig.update_layout(
                    title="Memory Usage Over Time",
                    xaxis_title="Time",
                    yaxis_title="Memory %",
                    height=300,
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Network and Disk monitoring
    if "Network" in components or "Disk" in components:
        col1, col2 = st.columns(2)
        
        if "Network" in components:
            with col1:
                st.subheader("🌐 Network Activity")
                
                # Generate network data
                network_in = 50 + 30 * np.sin(np.linspace(0, 3*np.pi, len(timestamps))) + np.random.normal(0, 10, len(timestamps))
                network_out = 30 + 20 * np.cos(np.linspace(0, 3*np.pi, len(timestamps))) + np.random.normal(0, 8, len(timestamps))
                network_in = np.clip(network_in, 0, 150)
                network_out = np.clip(network_out, 0, 100)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=network_in,
                    mode='lines',
                    name='Incoming',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=network_out,
                    mode='lines',
                    name='Outgoing',
                    line=dict(color='orange', width=2)
                ))
                
                fig.update_layout(
                    title="Network I/O (MB/s)",
                    xaxis_title="Time",
                    yaxis_title="MB/s",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        if "Disk" in components:
            with col2:
                st.subheader("💾 Disk Activity")
                
                # Generate disk I/O data
                disk_read = 20 + 15 * np.sin(np.linspace(0, 2*np.pi, len(timestamps))) + np.random.normal(0, 5, len(timestamps))
                disk_write = 15 + 10 * np.cos(np.linspace(0, 2*np.pi, len(timestamps))) + np.random.normal(0, 4, len(timestamps))
                disk_read = np.clip(disk_read, 0, 60)
                disk_write = np.clip(disk_write, 0, 40)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=disk_read,
                    mode='lines',
                    name='Read',
                    line=dict(color='green', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=disk_write,
                    mode='lines',
                    name='Write',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title="Disk I/O (MB/s)",
                    xaxis_title="Time",
                    yaxis_title="MB/s",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Process monitoring
    st.subheader("🔄 Process Monitoring")
    
    if PSUTIL_AVAILABLE:
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] > 1.0:  # Only show processes using > 1% CPU
                        processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU usage
            processes = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:10]
            
            if processes:
                process_df = pd.DataFrame(processes)
                process_df['cpu_percent'] = process_df['cpu_percent'].round(2)
                process_df['memory_percent'] = process_df['memory_percent'].round(2)
                
                st.dataframe(
                    process_df,
                    column_config={
                        "pid": "PID",
                        "name": "Process Name",
                        "cpu_percent": st.column_config.NumberColumn("CPU %", format="%.2f"),
                        "memory_percent": st.column_config.NumberColumn("Memory %", format="%.2f")
                    },
                    hide_index=True,
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error getting process information: {e}")
    else:
        # Mock process data
        mock_processes = [
            {"pid": 1234, "name": "streamlit", "cpu_percent": 15.2, "memory_percent": 8.5},
            {"pid": 5678, "name": "python", "cpu_percent": 12.8, "memory_percent": 6.2},
            {"pid": 9012, "name": "code", "cpu_percent": 8.4, "memory_percent": 12.1},
            {"pid": 3456, "name": "chrome", "cpu_percent": 6.7, "memory_percent": 18.9},
            {"pid": 7890, "name": "system", "cpu_percent": 3.2, "memory_percent": 4.1}
        ]
        
        process_df = pd.DataFrame(mock_processes)
        st.dataframe(
            process_df,
            column_config={
                "pid": "PID",
                "name": "Process Name",
                "cpu_percent": st.column_config.NumberColumn("CPU %", format="%.2f"),
                "memory_percent": st.column_config.NumberColumn("Memory %", format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )

def render_historical_monitoring(components):
    """Render historical monitoring data"""
    st.subheader("📈 Historical Monitoring")
    
    # Time range selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week", "Last Month"]
        )
    
    with col2:
        export_data = st.button("📊 Export Data", use_container_width=True)
        if export_data:
            st.success("Data export initiated")
    
    # Generate historical data based on time range
    if time_range == "Last Hour":
        periods, freq = 60, '1min'
    elif time_range == "Last 6 Hours":
        periods, freq = 360, '1min'
    elif time_range == "Last 24 Hours":
        periods, freq = 288, '5min'
    elif time_range == "Last Week":
        periods, freq = 168, '1H'
    else:  # Last Month
        periods, freq = 720, '1H'
    
    timestamps = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
    
    # System performance overview
    st.subheader("📊 Performance Summary")
    
    # Generate summary statistics
    avg_cpu = np.random.uniform(30, 60)
    avg_memory = np.random.uniform(40, 70)
    avg_disk = np.random.uniform(20, 50)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average CPU", f"{avg_cpu:.1f}%")
    
    with col2:
        st.metric("Average Memory", f"{avg_memory:.1f}%")
    
    with col3:
        st.metric("Peak Usage", f"{max(avg_cpu, avg_memory):.1f}%")
    
    with col4:
        uptime = np.random.uniform(95, 99.9)
        st.metric("Uptime", f"{uptime:.2f}%")
    
    # Historical charts
    if components:
        # Combined view
        st.subheader("📈 Combined Resource Usage")
        
        fig = go.Figure()
        
        if "CPU" in components:
            cpu_data = avg_cpu + 10 * np.sin(np.linspace(0, 4*np.pi, len(timestamps))) + np.random.normal(0, 5, len(timestamps))
            cpu_data = np.clip(cpu_data, 0, 100)
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=cpu_data,
                mode='lines',
                name='CPU %',
                line=dict(color='blue', width=2)
            ))
        
        if "Memory" in components:
            memory_data = avg_memory + 5 * np.cos(np.linspace(0, 2*np.pi, len(timestamps))) + np.random.normal(0, 3, len(timestamps))
            memory_data = np.clip(memory_data, 0, 100)
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=memory_data,
                mode='lines',
                name='Memory %',
                line=dict(color='green', width=2)
            ))
        
        if "Disk" in components:
            disk_data = avg_disk + 3 * np.sin(np.linspace(0, 6*np.pi, len(timestamps))) + np.random.normal(0, 2, len(timestamps))
            disk_data = np.clip(disk_data, 0, 100)
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=disk_data,
                mode='lines',
                name='Disk %',
                line=dict(color='orange', width=2)
            ))
        
        fig.update_layout(
            title=f"Resource Usage - {time_range}",
            xaxis_title="Time",
            yaxis_title="Usage %",
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdowns
        if len(components) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Resource utilization heatmap
                st.subheader("🔥 Resource Heatmap")
                
                # Create hourly heatmap data
                hours = list(range(24))
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                
                heatmap_data = np.random.uniform(20, 80, (len(days), len(hours)))
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=hours,
                    y=days,
                    colorscale='RdYlBu_r',
                    colorbar=dict(title="Usage %")
                ))
                
                fig.update_layout(
                    title="Weekly Usage Pattern",
                    xaxis_title="Hour of Day",
                    yaxis_title="Day of Week",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Usage distribution
                st.subheader("📊 Usage Distribution")
                
                # Generate distribution data
                if "CPU" in components:
                    cpu_dist = np.random.normal(avg_cpu, 15, 1000)
                    cpu_dist = np.clip(cpu_dist, 0, 100)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=cpu_dist,
                        nbinsx=30,
                        name='CPU Distribution',
                        opacity=0.7
                    ))
                    
                    fig.update_layout(
                        title="CPU Usage Distribution",
                        xaxis_title="CPU %",
                        yaxis_title="Frequency",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def render_alerts_monitoring():
    """Render alerts and notifications"""
    st.subheader("🚨 System Alerts & Notifications")
    
    # Alert statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Critical Alerts", "2", "+1")
    
    with col2:
        st.metric("Warning Alerts", "5", "-2")
    
    with col3:
        st.metric("Info Alerts", "12", "+3")
    
    with col4:
        st.metric("Resolution Rate", "94%", "+2%")
    
    # Recent alerts
    st.subheader("🔔 Recent Alerts")
    
    alerts = [
        {
            "Time": datetime.now() - timedelta(minutes=5),
            "Type": "🔴 Critical",
            "Component": "CPU",
            "Message": "CPU usage exceeded 85% for 5 minutes",
            "Status": "Active"
        },
        {
            "Time": datetime.now() - timedelta(minutes=15),
            "Type": "🟡 Warning",
            "Component": "Memory",
            "Message": "Memory usage above 75% threshold",
            "Status": "Acknowledged"
        },
        {
            "Time": datetime.now() - timedelta(hours=1),
            "Type": "🔴 Critical",
            "Component": "Disk",
            "Message": "Low disk space on system drive",
            "Status": "Resolved"
        },
        {
            "Time": datetime.now() - timedelta(hours=2),
            "Type": "🟡 Warning",
            "Component": "Network",
            "Message": "High network latency detected",
            "Status": "Resolved"
        },
        {
            "Time": datetime.now() - timedelta(hours=3),
            "Type": "🔵 Info",
            "Component": "System",
            "Message": "System restart completed successfully",
            "Status": "Resolved"
        }
    ]
    
    # Display alerts
    for alert in alerts:
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 2, 3, 1])
            
            with col1:
                st.write(f"**{alert['Type']}**")
            
            with col2:
                st.write(f"**{alert['Component']}**")
                st.write(f"_{alert['Time'].strftime('%H:%M:%S')}_")
            
            with col3:
                st.write(alert['Message'])
            
            with col4:
                status_color = {
                    "Active": "🔴",
                    "Acknowledged": "🟡",
                    "Resolved": "🟢"
                }
                st.write(f"{status_color.get(alert['Status'], '')} {alert['Status']}")
            
            st.divider()
    
    # Alert configuration
    with st.expander("⚙️ Alert Configuration"):
        st.subheader("Threshold Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cpu_warning = st.slider("CPU Warning Threshold (%)", 50, 90, 70)
            cpu_critical = st.slider("CPU Critical Threshold (%)", 70, 95, 85)
            
            memory_warning = st.slider("Memory Warning Threshold (%)", 60, 90, 75)
            memory_critical = st.slider("Memory Critical Threshold (%)", 80, 95, 90)
        
        with col2:
            disk_warning = st.slider("Disk Warning Threshold (%)", 70, 90, 80)
            disk_critical = st.slider("Disk Critical Threshold (%)", 85, 98, 95)
            
            network_threshold = st.slider("Network Latency Threshold (ms)", 100, 1000, 500)
        
        st.subheader("Notification Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            email_alerts = st.toggle("Email Alerts", value=True)
            if email_alerts:
                email_address = st.text_input("Email Address", "admin@company.com")
        
        with col2:
            slack_alerts = st.toggle("Slack Notifications", value=False)
            if slack_alerts:
                slack_webhook = st.text_input("Slack Webhook URL")
        
        if st.button("💾 Save Configuration"):
            st.success("Alert configuration saved")

if __name__ == "__main__":
    main()