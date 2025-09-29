# SARIMA Forecasting Dashboard - Frontend Only (Uses Pre-trained Models)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import io

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import data utilities
from utils.data_loader import load_wind_data, get_location_list

# Configure page
st.set_page_config(
    page_title="SARIMA Forecasting Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'sarima_initialized' not in st.session_state:
        st.session_state.sarima_initialized = True
        st.session_state.loaded_models = {}

initialize_session_state()

# Load data with caching
@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process_sarima_data():
    """Load and prepare wind data for SARIMA analysis."""
    try:
        data = load_wind_data()
        df = data.copy()
        
        # Ensure proper date handling
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')
        
        # Add time-based features
        df['year'] = df['date'].dt.year.astype('Int64')
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['quarter'] = df['date'].dt.quarter
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_sarima_metadata(df):
    """Get metadata for SARIMA analysis."""
    if df.empty:
        return [], [], []
    
    cities = get_location_list(df)
    years = sorted([int(y) for y in df['year'].dropna().unique() if y > 1900])
    
    # Available wind parameters
    wind_params = [col for col in df.columns if 'wind_speed' in col or 'wind_power_density' in col]
    
    return cities, years, wind_params

@st.cache_data(ttl=3600, show_spinner=False)
def load_available_models(models_dir: str = "models"):
    """Load information about available pre-trained models."""
    try:
        if not os.path.exists(models_dir):
            return {}
        
        available_models = {}
        for filename in os.listdir(models_dir):
            if filename.endswith('.pkl') and filename.startswith('sarima_'):
                try:
                    # Parse filename: sarima_cityname_parameter.pkl
                    parts = filename.replace('sarima_', '').replace('.pkl', '').split('_')
                    if len(parts) >= 2:
                        city = '_'.join(parts[:-1]).replace('_', ' ')
                        parameter = parts[-1]
                        
                        if city not in available_models:
                            available_models[city] = []
                        available_models[city].append({
                            'parameter': parameter,
                            'filename': filename,
                            'filepath': os.path.join(models_dir, filename)
                        })
                except Exception:
                    continue
        
        return available_models
    except Exception as e:
        st.error(f"Error loading model information: {str(e)}")
        return {}

def load_model(filepath: str) -> Optional[Dict]:
    """Load a pre-trained SARIMA model."""
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"Error loading model from {filepath}: {str(e)}")
        return None

def create_forecast_plot(historical_data: pd.Series, model_data: Dict, 
                        show_historical_days: int = 90) -> go.Figure:
    """Create interactive forecast plot."""
    try:
        # Get recent historical data for context
        recent_data = historical_data.tail(show_historical_days)
        
        # Get forecast data
        forecast_dates = pd.to_datetime(model_data['forecast_dates'])
        forecast_values = model_data['forecast_values']
        lower_ci = model_data['confidence_intervals']['lower']
        upper_ci = model_data['confidence_intervals']['upper']
        
        # Create plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data.values,
            mode='lines',
            name='Historical Data',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f} m/s<extra></extra>'
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2),
            marker=dict(size=4),
            hovertemplate='Date: %{x}<br>Forecast: %{y:.2f} m/s<extra></extra>'
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=upper_ci,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=lower_ci,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255,0,0,0.2)',
            fill='tonexty',
            name='95% Confidence Interval',
            hovertemplate='Date: %{x}<br>Lower CI: %{y:.2f} m/s<extra></extra>'
        ))
        
        # Add vertical line at forecast start
        if len(recent_data) > 0:
            fig.add_vline(
                x=recent_data.index[-1],
                line_dash="dash",
                line_color="gray",
                annotation_text="Forecast Start"
            )
        
        fig.update_layout(
            title=f"Wind Speed Forecast - {model_data['city']} ({model_data['parameter']})",
            xaxis_title="Date",
            yaxis_title="Wind Speed (m/s)",
            hovermode='x unified',
            legend=dict(x=0, y=1),
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating forecast plot: {str(e)}")
        return go.Figure()

def create_residuals_plot(model_data: Dict) -> go.Figure:
    """Create residuals analysis plot."""
    try:
        residuals = model_data['residuals']
        fitted_values = model_data['fitted_values']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Residuals vs Time', 'Residuals vs Fitted', 'Residuals Histogram', 'Q-Q Plot'],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Residuals vs Time
        fig.add_trace(go.Scatter(
            y=residuals,
            mode='lines',
            name='Residuals',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Residuals vs Fitted
        fig.add_trace(go.Scatter(
            x=fitted_values,
            y=residuals,
            mode='markers',
            name='Residuals vs Fitted',
            marker=dict(color='red', size=4, opacity=0.6)
        ), row=1, col=2)
        
        # Residuals Histogram
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Histogram',
            marker_color='green'
        ), row=2, col=1)
        
        # Add normal curve overlay for Q-Q plot approximation
        residuals_array = np.array(residuals)
        sorted_residuals = np.sort(residuals_array)
        theoretical_quantiles = np.linspace(0.01, 0.99, len(sorted_residuals))
        theoretical_values = np.percentile(residuals_array, theoretical_quantiles * 100)
        
        fig.add_trace(go.Scatter(
            x=theoretical_values,
            y=sorted_residuals,
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color='purple', size=4)
        ), row=2, col=2)
        
        fig.update_layout(
            title="Model Diagnostics - Residuals Analysis",
            showlegend=False,
            height=600
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating residuals plot: {str(e)}")
        return go.Figure()

def create_seasonal_decomposition_plot(historical_data: pd.Series, parameter: str) -> go.Figure:
    """Create seasonal decomposition plot."""
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Perform seasonal decomposition
        if len(historical_data) >= 730:  # At least 2 years of data
            decomposition = seasonal_decompose(
                historical_data.dropna(),
                model='additive',
                period=365
            )
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                vertical_spacing=0.05
            )
            
            # Original series
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data.values,
                mode='lines',
                name='Original',
                line=dict(color='blue')
            ), row=1, col=1)
            
            # Trend
            fig.add_trace(go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend.values,
                mode='lines',
                name='Trend',
                line=dict(color='orange')
            ), row=2, col=1)
            
            # Seasonal
            fig.add_trace(go.Scatter(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal.values,
                mode='lines',
                name='Seasonal',
                line=dict(color='green')
            ), row=3, col=1)
            
            # Residual
            fig.add_trace(go.Scatter(
                x=decomposition.resid.index,
                y=decomposition.resid.values,
                mode='lines',
                name='Residual',
                line=dict(color='red')
            ), row=4, col=1)
            
            fig.update_layout(
                title=f"Seasonal Decomposition - {parameter}",
                showlegend=False,
                height=800
            )
            
            return fig
        else:
            st.warning("Insufficient data for seasonal decomposition (minimum 2 years required)")
            return go.Figure()
            
    except Exception as e:
        st.error(f"Error creating seasonal decomposition: {str(e)}")
        return go.Figure()

# Load data
df = load_and_process_sarima_data()
cities, years, wind_params = get_sarima_metadata(df)
available_models = load_available_models()

# Check if data and models are available
if df.empty or not cities:
    st.error("No wind data available. Please check your data source.")
    st.stop()

if not available_models:
    st.error("""
    No pre-trained SARIMA models found in the 'models/' directory.
    
    Please:
    1. Run the training script locally: `python sarima_training_script.py --data your_data.csv`
    2. Copy the generated model files to the 'models/' directory
    3. Refresh this dashboard
    """)
    st.stop()

# Main dashboard
st.markdown('<h1 style="text-align: center;">SARIMA Forecasting Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;">Wind Energy Forecasting with Pre-trained Models</h3>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.title("Forecasting Configuration")
    
    # Model selection
    st.subheader("Model Selection")
    
    # Available cities with models
    cities_with_models = list(available_models.keys())
    selected_city = st.selectbox(
        "Select Location", 
        cities_with_models, 
        key="forecast_city"
    )
    
    # Available parameters for selected city
    if selected_city in available_models:
        available_params = [model['parameter'] for model in available_models[selected_city]]
        selected_param = st.selectbox(
            "Select Parameter",
            available_params,
            key="forecast_param"
        )
        
        # Get model info
        selected_model_info = next(
            model for model in available_models[selected_city] 
            if model['parameter'] == selected_param
        )
    else:
        st.error("No models available for selected city")
        st.stop()
    
    # Display configuration
    st.subheader("Display Options")
    show_historical_days = st.slider(
        "Historical Context (days)", 
        30, 365, 90,
        help="Number of historical days to show for context"
    )
    
    show_confidence_intervals = st.checkbox(
        "Show Confidence Intervals", 
        value=True,
        help="Display 95% confidence intervals for forecasts"
    )
    
    show_model_diagnostics = st.checkbox(
        "Show Model Diagnostics",
        value=False,
        help="Display model validation and diagnostic plots"
    )
    
    # Load selected model
    if st.button("Load Model & Generate Forecast", type="primary"):
        with st.spinner("Loading model and generating forecast..."):
            model_data = load_model(selected_model_info['filepath'])
            if model_data:
                st.session_state.current_model = model_data
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model")

# Main content area
if 'current_model' in st.session_state:
    model_data = st.session_state.current_model
    
    # Model information header
    st.subheader("Model Information")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Location", model_data['city'])
    
    with col2:
        st.metric("Parameter", model_data['parameter'])
    
    with col3:
        order = model_data['order']
        st.metric("SARIMA Order", f"({order[0]},{order[1]},{order[2]})")
    
    with col4:
        st.metric("AIC Score", f"{model_data['aic']:.2f}")
    
    with col5:
        seasonal_order = model_data['seasonal_order']
        st.metric("Seasonal Order", f"({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]})")
    
    # Model accuracy metrics
    if model_data.get('accuracy_metrics'):
        accuracy = model_data['accuracy_metrics']
        
        st.subheader("Model Accuracy")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE", f"{accuracy.get('mae', 0):.3f}")
        
        with col2:
            st.metric("RMSE", f"{accuracy.get('rmse', 0):.3f}")
        
        with col3:
            st.metric("MAPE", f"{accuracy.get('mape', 0):.1f}%")
    
    # Get historical data for context
    try:
        city_data = df[df['location'] == model_data['city']].copy()
        city_data = city_data.set_index('date').sort_index()
        
        if model_data['parameter'] in city_data.columns:
            historical_data = city_data[model_data['parameter']].dropna()
        else:
            st.error(f"Parameter {model_data['parameter']} not found in historical data")
            st.stop()
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        st.stop()
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Forecast Visualization", 
        "Statistical Analysis", 
        "Model Diagnostics", 
        "Data Export"
    ])
    
    with tab1:
        st.subheader("Wind Speed Forecast")
        
        # Main forecast plot
        forecast_fig = create_forecast_plot(historical_data, model_data, show_historical_days)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Forecast statistics
        st.subheader("Forecast Statistics")
        
        forecast_values = np.array(model_data['forecast_values'])
        lower_ci = np.array(model_data['confidence_intervals']['lower'])
        upper_ci = np.array(model_data['confidence_intervals']['upper'])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Mean Forecast", f"{forecast_values.mean():.2f} m/s")
        
        with col2:
            st.metric("Min Forecast", f"{forecast_values.min():.2f} m/s")
        
        with col3:
            st.metric("Max Forecast", f"{forecast_values.max():.2f} m/s")
        
        with col4:
            st.metric("Forecast Std", f"{forecast_values.std():.2f} m/s")
        
        with col5:
            uncertainty = (upper_ci - lower_ci).mean()
            st.metric("Avg Uncertainty", f"{uncertainty:.2f} m/s")
        
        # Weekly forecast breakdown
        st.subheader("Weekly Forecast Breakdown")
        
        forecast_df = pd.DataFrame({
            'Date': pd.to_datetime(model_data['forecast_dates']),
            'Forecast': forecast_values,
            'Lower_CI': lower_ci,
            'Upper_CI': upper_ci
        })
        
        # Add week information
        forecast_df['Week'] = forecast_df['Date'].dt.isocalendar().week
        forecast_df['WeekStart'] = forecast_df['Date'] - pd.to_timedelta(forecast_df['Date'].dt.dayofweek, unit='D')
        
        weekly_summary = forecast_df.groupby('WeekStart').agg({
            'Forecast': ['mean', 'min', 'max'],
            'Lower_CI': 'mean',
            'Upper_CI': 'mean'
        }).round(2)
        
        weekly_summary.columns = ['Mean', 'Min', 'Max', 'Lower CI', 'Upper CI']
        weekly_summary.index = weekly_summary.index.strftime('%Y-%m-%d')
        
        st.dataframe(weekly_summary, use_container_width=True)
    
    with tab2:
        st.subheader("Statistical Analysis")
        
        # Historical data statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Historical Data Statistics:**")
            
            hist_stats = {
                'Metric': ['Count', 'Mean', 'Std', 'Min', 'Max', 'Median'],
                'Value': [
                    f"{len(historical_data):,}",
                    f"{historical_data.mean():.2f} m/s",
                    f"{historical_data.std():.2f} m/s",
                    f"{historical_data.min():.2f} m/s", 
                    f"{historical_data.max():.2f} m/s",
                    f"{historical_data.median():.2f} m/s"
                ]
            }
            
            st.dataframe(pd.DataFrame(hist_stats), use_container_width=True)
        
        with col2:
            st.write("**Model Information:**")
            
            model_info = {
                'Metric': ['Training Data', 'Test Data', 'AIC', 'BIC', 'Log Likelihood'],
                'Value': [
                    f"{model_data.get('training_data_length', 'N/A'):,}",
                    f"{model_data.get('test_data_length', 'N/A'):,}",
                    f"{model_data.get('aic', 0):.2f}",
                    f"{model_data.get('bic', 0):.2f}",
                    f"{model_data.get('log_likelihood', 0):.2f}"
                ]
            }
            
            st.dataframe(pd.DataFrame(model_info), use_container_width=True)
        
        # Seasonal decomposition
        if len(historical_data) >= 730:
            st.subheader("Seasonal Decomposition")
            decomp_fig = create_seasonal_decomposition_plot(historical_data, model_data['parameter'])
            if decomp_fig.data:
                st.plotly_chart(decomp_fig, use_container_width=True)
        
        # Historical vs Forecast comparison
        st.subheader("Historical vs Forecast Distribution")
        
        fig_dist = go.Figure()
        
        # Historical distribution
        fig_dist.add_trace(go.Histogram(
            x=historical_data.values,
            name='Historical',
            opacity=0.7,
            nbinsx=30,
            marker_color='blue'
        ))
        
        # Forecast distribution
        fig_dist.add_trace(go.Histogram(
            x=forecast_values,
            name='Forecast',
            opacity=0.7,
            nbinsx=15,
            marker_color='red'
        ))
        
        fig_dist.update_layout(
            title="Distribution Comparison: Historical vs Forecast",
            xaxis_title="Wind Speed (m/s)",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab3:
        st.subheader("Model Diagnostics")
        
        if show_model_diagnostics:
            # Residuals analysis
            if model_data.get('residuals'):
                residuals_fig = create_residuals_plot(model_data)
                if residuals_fig.data:
                    st.plotly_chart(residuals_fig, use_container_width=True)
                
                # Residual statistics
                residuals = np.array(model_data['residuals'])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Residuals Mean", f"{residuals.mean():.4f}")
                
                with col2:
                    st.metric("Residuals Std", f"{residuals.std():.4f}")
                
                with col3:
                    from scipy import stats
                    skewness = stats.skew(residuals)
                    st.metric("Skewness", f"{skewness:.4f}")
                
                with col4:
                    kurtosis = stats.kurtosis(residuals)
                    st.metric("Kurtosis", f"{kurtosis:.4f}")
            
            # Stationarity information
            if model_data.get('stationarity'):
                st.subheader("Stationarity Tests")
                stationarity = model_data['stationarity']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Augmented Dickey-Fuller Test:**")
                    st.write(f"- P-value: {stationarity.get('adf_pvalue', 'N/A'):.4f}")
                    adf_result = '✅ Stationary' if stationarity.get('adf_stationary', False) else '❌ Non-stationary'
                    st.write(f"- Result: {adf_result}")
                
                with col2:
                    st.write("**KPSS Test:**")
                    st.write(f"- P-value: {stationarity.get('kpss_pvalue', 'N/A'):.4f}")
                    kpss_result = '✅ Stationary' if stationarity.get('kpss_stationary', False) else '❌ Non-stationary'
                    st.write(f"- Result: {kpss_result}")
        else:
            st.info("Enable 'Show Model Diagnostics' in the sidebar to view detailed diagnostic plots.")
    
    with tab4:
        st.subheader("Data Export")
        
        # Forecast data export
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Export Forecast Data**")
            
            export_df = pd.DataFrame({
                'Date': model_data['forecast_dates'],
                'Forecast': [f"{v:.3f}" for v in model_data['forecast_values']],
                'Lower_CI': [f"{v:.3f}" for v in model_data['confidence_intervals']['lower']],
                'Upper_CI': [f"{v:.3f}" for v in model_data['confidence_intervals']['upper']]
            })
            
            csv_forecast = export_df.to_csv(index=False)
            st.download_button(
                "Download Forecast CSV",
                csv_forecast,
                file_name=f"forecast_{model_data['city'].replace(' ', '_')}_{model_data['parameter']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.write("**Export Model Summary**")
            
            summary_data = {
                'Model': 'SARIMA',
                'Location': model_data['city'],
                'Parameter': model_data['parameter'],
                'SARIMA_Order': str(model_data['order']),
                'Seasonal_Order': str(model_data['seasonal_order']),
                'AIC': f"{model_data['aic']:.2f}",
                'BIC': f"{model_data['bic']:.2f}",
                'Training_Date': model_data.get('training_date', 'N/A'),
                'Forecast_Horizon': len(model_data['forecast_values']),
                'Mean_Forecast': f"{np.mean(model_data['forecast_values']):.2f}",
                'Export_Date': datetime.now().isoformat()
            }
            
            summary_df = pd.DataFrame([summary_data])
            csv_summary = summary_df.to_csv(index=False)
            
            st.download_button(
                "Download Model Summary CSV",
                csv_summary,
                file_name=f"model_summary_{model_data['city'].replace(' ', '_')}_{model_data['parameter']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Training data comparison
        st.subheader("Training vs Forecast Comparison")
        
        comparison_data = []
        
        # Add historical statistics
        comparison_data.append({
            'Dataset': 'Historical',
            'Mean': f"{historical_data.mean():.2f}",
            'Std': f"{historical_data.std():.2f}",
            'Min': f"{historical_data.min():.2f}",
            'Max': f"{historical_data.max():.2f}",
            'Count': len(historical_data)
        })
        
        # Add forecast statistics
        comparison_data.append({
            'Dataset': 'Forecast',
            'Mean': f"{np.mean(model_data['forecast_values']):.2f}",
            'Std': f"{np.std(model_data['forecast_values']):.2f}",
            'Min': f"{np.min(model_data['forecast_values']):.2f}",
            'Max': f"{np.max(model_data['forecast_values']):.2f}",
            'Count': len(model_data['forecast_values'])
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

else:
    # Initial state - show available models
    st.subheader("Available Pre-trained Models")
    
    # Create model summary table
    model_summary_data = []
    for city, models in available_models.items():
        for model in models:
            model_summary_data.append({
                'City': city,
                'Parameter': model['parameter'],
                'Filename': model['filename']
            })
    
    if model_summary_data:
        summary_df = pd.DataFrame(model_summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        st.info("👈 Select a model from the sidebar and click 'Load Model & Generate Forecast' to begin analysis.")
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Available Cities", len(available_models))
        
        with col2:
            total_models = sum(len(models) for models in available_models.values())
            st.metric("Total Models", total_models)
        
        with col3:
            unique_params = set()
            for models in available_models.values():
                for model in models:
                    unique_params.add(model['parameter'])
            st.metric("Parameters", len(unique_params))
    
    else:
        st.warning("No pre-trained models found in the models directory.")

# Technical information
with st.expander("Model Information & Usage Guide"):
    st.markdown(f"""
    ### SARIMA Forecasting Dashboard
    
    This dashboard displays forecasts from **pre-trained SARIMA models** that were trained locally using the training script.
    
    #### Model Training Process
    
    **Step 1: Local Training**
    ```bash
    # Run the training script on your local machine
    python sarima_training_script.py --data your_wind_data.csv --output models
    ```
    
    **Step 2: Model Files**
    - Models are saved as `.pkl` files in the `models/` directory
    - Each model file contains: fitted model, forecasts, diagnostics, accuracy metrics
    - File naming: `sarima_[city]_[parameter].pkl`
    
    **Step 3: Dashboard Deployment** 
    - Copy model files to your deployment environment
    - Dashboard automatically detects and loads available models
    - No training required on the frontend
    
    #### Currently Available Models
    
    **Cities with Models:** {len(available_models)}
    **Total Models:** {sum(len(models) for models in available_models.values())}
    
    **Available Locations:**
    {', '.join(list(available_models.keys())[:10])}{'...' if len(available_models) > 10 else ''}
    
    #### Features
    
    **Forecasting:**
    - 30-day ahead forecasts with confidence intervals
    - Interactive visualizations with historical context
    - Weekly forecast summaries and statistics
    
    **Analysis:**
    - Model diagnostics and residual analysis
    - Seasonal decomposition (when sufficient data)
    - Statistical comparisons between historical and forecast data
    
    **Export:**
    - CSV export of forecast data and model summaries
    - Downloadable visualizations
    - Comprehensive model performance metrics
    
    #### Technical Specifications
    
    **Model Architecture:** SARIMA (Seasonal AutoRegressive Integrated Moving Average)
    **Optimization:** AIC-based parameter selection
    **Validation:** Train/test split with accuracy metrics (MAE, RMSE, MAPE)
    **Confidence Intervals:** 95% prediction intervals
    **Seasonal Period:** 365 days (annual seasonality)
    
    #### Performance Metrics
    
    Models are evaluated using:
    - **MAE**: Mean Absolute Error
    - **RMSE**: Root Mean Square Error  
    - **MAPE**: Mean Absolute Percentage Error
    - **AIC**: Akaike Information Criterion
    - **Residual Analysis**: Autocorrelation and normality tests
    """)

# Footer
st.markdown("---")
try:
    st.markdown(
        f"""
        <div style='text-align: center; padding: 2rem; 
                    background: linear-gradient(45deg, #1e3c72, #2a5298); 
                    border-radius: 15px; 
                    color: white; 
                    font-family: Arial, sans-serif;'>
            <h4 style='color: #f9a825;'>SARIMA Forecasting Dashboard</h4>
            <p><strong>Available Models:</strong> {sum(len(models) for models in available_models.values())} | 
            <strong>Cities:</strong> {len(available_models)} | 
            <strong>Forecast Horizon:</strong> 30 days</p>
            <p><em>Pre-trained Models • Real-time Analytics • Built with Streamlit</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )
except Exception:
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem; color: #666;'>
            <p><em>SARIMA Forecasting Dashboard</em></p>
        </div>
        """, 
        unsafe_allow_html=True
    )