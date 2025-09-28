import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import io
from typing import List, Dict, Any

# Import data utilities
from utils.data_loader import load_wind_data, get_location_list

# Configure page
st.set_page_config(
    page_title="Wind Analytics Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_and_process_data():
    """Load and prepare wind data for analysis."""
    data = load_wind_data()
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month_name'] = df['date'].dt.strftime('%B')
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week'] = df['date'].dt.isocalendar().week
    return df

# Initialize data
df = load_and_process_data()
cities = get_location_list(df)
years_available = sorted(df['date'].dt.year.unique().tolist())
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
def create_wind_rose(data: pd.DataFrame, speed_col: str, direction_col: str, title: str):
    """Create wind rose chart."""
    # Bin wind directions into 16 sectors
    direction_bins = np.arange(0, 361, 22.5)
    data['dir_bin'] = pd.cut(data[direction_col], direction_bins, labels=False)
    
    # Create wind rose data
    rose_data = []
    for i in range(16):
        sector_data = data[data['dir_bin'] == i]
        if not sector_data.empty:
            rose_data.append({
                'direction': i * 22.5,
                'frequency': len(sector_data) / len(data) * 100,
                'avg_speed': sector_data[speed_col].mean()
            })
    
    if not rose_data:
        return go.Figure().add_annotation(text="No data available for wind rose")
    
    rose_df = pd.DataFrame(rose_data)
    
    fig = go.Figure()
    
    # Add wind rose bars
    fig.add_trace(go.Barpolar(
        r=rose_df['frequency'],
        theta=rose_df['direction'],
        width=22.5,
        marker_color=rose_df['avg_speed'],
        marker_colorscale='Viridis',
        marker_colorbar=dict(title="Wind Speed (m/s)"),
        hovertemplate='<b>Direction:</b> %{theta}°<br><b>Frequency:</b> %{r:.1f}%<br><b>Avg Speed:</b> %{marker.color:.1f} m/s<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, rose_df['frequency'].max() * 1.1]),
            angularaxis=dict(direction="clockwise", rotation=90)  # Changed from 'start' to 'rotation'
        )
    )
    
    return fig


def create_comparison_chart(data: pd.DataFrame, cities_list: List[str], metric: str, chart_type: str = "line"):
    """Create comparison charts for multiple cities."""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3[:len(cities_list)]
    
    for i, city in enumerate(cities_list):
        city_data = data[data['location'] == city].copy()
        
        if chart_type == "line":
            fig.add_trace(go.Scatter(
                x=city_data['date'],
                y=city_data[metric],
                mode='lines',
                name=city,
                line=dict(color=colors[i])
            ))
        elif chart_type == "box":
            fig.add_trace(go.Box(
                y=city_data[metric],
                name=city,
                marker_color=colors[i]
            ))
    
    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} Comparison",
        xaxis_title="Date" if chart_type == "line" else "",
        yaxis_title=metric.replace('_', ' ').title(),
        hovermode='x unified' if chart_type == "line" else 'closest'
    )
    
    return fig

def calculate_wind_statistics(data: pd.DataFrame, speed_cols: List[str]):
    """Calculate comprehensive wind statistics."""
    stats = {}
    
    for col in speed_cols:
        if col in data.columns:
            stats[col] = {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'q25': data[col].quantile(0.25),
                'q75': data[col].quantile(0.75),
                'calm_percentage': (data[col] < 2).mean() * 100,  # Wind speeds < 2 m/s
                'optimal_percentage': ((data[col] >= 3) & (data[col] <= 25)).mean() * 100  # Turbine operational range
            }
    
    return stats

def export_data(data: pd.DataFrame, file_format: str, filename: str):
    """Export data in various formats."""
    if file_format == "CSV":
        return data.to_csv(index=False)
    elif file_format == "Excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='Wind_Data', index=False)
        return output.getvalue()
    elif file_format == "JSON":
        return data.to_json(orient='records', date_format='iso')

# Sidebar Configuration
with st.sidebar:
    st.title("Analytics Configuration")
    
    # Analysis Mode
    analysis_mode = st.radio(
        "Analysis Mode:",
        ["Single City", "Multi-City Comparison"],
        horizontal=True
    )
    
    # City Selection
    if analysis_mode == "Single City":
        selected_cities = [st.selectbox("Select City", cities, index=0)]
    else:
        selected_cities = st.multiselect(
            "Select Cities (up to 8)",
            cities,
            default=cities[:3] if len(cities) >= 3 else cities,
            max_selections=8
        )
    
    # Wind Speed Parameters
    st.subheader("Wind Parameters")
    height_options = ["2m", "10m", "50m"]
    selected_height = st.selectbox("Wind Measurement Height", height_options, index=1)
    
    speed_type_options = ["Average", "Maximum", "Minimum"]
    selected_speed_type = st.selectbox("Speed Type", speed_type_options, index=0)
    
    # Additional Parameters
    show_power_density = st.checkbox("Show Power Density Analysis", value=True)
    show_turbine_operational = st.checkbox("Show Turbine Operational Status", value=True)
    show_seasonal_analysis = st.checkbox("Show Seasonal Analysis", value=True)
    
    # Time Filter
    st.subheader("Time Period")
    time_filter = st.selectbox(
        "Filter Type",
        ["All Data", "Specific Year", "Date Range", "Seasonal", "Monthly"],
        index=0
    )
    
    # Filter Configuration
    if time_filter == "Specific Year":
        selected_year = st.selectbox("Select Year", years_available, index=len(years_available)-1)
        filter_data = df[df['date'].dt.year == selected_year]
    elif time_filter == "Date Range":
        min_date, max_date = df['date'].min().date(), df['date'].max().date()
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if len(date_range) == 2:
            filter_data = df[(df['date'].dt.date >= date_range[0]) & 
                           (df['date'].dt.date <= date_range[1])]
        else:
            filter_data = df
    elif time_filter == "Seasonal":
        selected_season = st.selectbox("Select Season", seasons)
        filter_data = df[df['season'] == selected_season]
    elif time_filter == "Monthly":
        selected_month = st.selectbox("Select Month", months)
        filter_data = df[df['month_name'] == selected_month]
    else:
        filter_data = df
    
    # Export Options
    st.subheader("Export Options")
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])

# Main Dashboard
st.markdown('<h1 class="main-header"> Wind Energy Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Comprehensive Wind Resource Assessment</p>', unsafe_allow_html=True)

st.markdown("---")

# Data validation
if not selected_cities:
    st.warning("Please select at least one city to begin analysis.")
    st.stop()

# Prepare column names
speed_col = f"wind_speed_{selected_height}"
if selected_speed_type != "Average":
    speed_col += f"_{selected_speed_type.lower()}"

direction_col = f"wind_direction_{selected_height}"
power_density_col = f"wind_power_density_{selected_height}"
resource_class_col = f"wind_resource_class_{selected_height}"
turbine_operational_col = f"turbine_operational_{selected_height}"

# Filter data for selected cities
city_data = filter_data[filter_data['location'].isin(selected_cities)].copy()

if city_data.empty:
    st.error("No data available for selected criteria. Please adjust your filters.")
    st.stop()

# Key Metrics Row
st.subheader("Key Performance Metrics")

if analysis_mode == "Single City":
    city = selected_cities[0]
    city_subset = city_data[city_data['location'] == city]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_speed = city_subset[speed_col].mean()
        st.metric("Average Wind Speed", f"{avg_speed:.2f} m/s")
    
    with col2:
        max_speed = city_subset[speed_col].max()
        st.metric("Peak Wind Speed", f"{max_speed:.2f} m/s")
    
    with col3:
        if power_density_col in city_subset.columns:
            avg_power = city_subset[power_density_col].mean()
            st.metric("Avg Power Density", f"{avg_power:.1f} W/m²")
        else:
            st.metric("Power Density", "N/A")
    
    with col4:
        operational_days = city_subset[turbine_operational_col].value_counts().get('Operating', 0)
        operational_pct = (operational_days / len(city_subset)) * 100
        st.metric("Operational Days", f"{operational_pct:.1f}%")
    
    with col5:
        resource_class = city_subset[resource_class_col].mode().iloc[0] if not city_subset[resource_class_col].mode().empty else "Unknown"
        st.metric("Resource Class", resource_class)

else:
    # Multi-city summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cities Analyzed", len(selected_cities))
    
    with col2:
        total_records = len(city_data)
        st.metric("Total Data Points", f"{total_records:,}")
    
    with col3:
        avg_speed_all = city_data[speed_col].mean()
        st.metric("Overall Avg Speed", f"{avg_speed_all:.2f} m/s")
    
    with col4:
        date_range_days = (city_data['date'].max() - city_data['date'].min()).days
        st.metric("Analysis Period", f"{date_range_days} days")

# Main Analysis Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Time Series Analysis", 
    "Wind Resource Assessment", 
    "Statistical Analysis", 
    "Wind Rose Analysis", 
    "Data Table"
])

with tab1:
    st.subheader("Wind Speed Time Series")
    
    if analysis_mode == "Single City":
        city = selected_cities[0]
        city_subset = city_data[city_data['location'] == city].sort_values('date')
        
        # Time series chart
        fig = px.line(
            city_subset, 
            x='date', 
            y=speed_col,
            title=f"Wind Speed Time Series - {city} ({selected_height} {selected_speed_type})",
            labels={speed_col: 'Wind Speed (m/s)', 'date': 'Date'}
        )
        fig.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional time series
        if show_power_density and power_density_col in city_subset.columns:
            fig_power = px.line(
                city_subset,
                x='date',
                y=power_density_col,
                title=f"Power Density Time Series - {city}",
                labels={power_density_col: 'Power Density (W/m²)', 'date': 'Date'}
            )
            fig_power.update_traces(line_color='#ff7f0e')
            st.plotly_chart(fig_power, use_container_width=True)
    
    else:
        # Multi-city comparison
        fig = create_comparison_chart(city_data, selected_cities, speed_col, "line")
        st.plotly_chart(fig, use_container_width=True)
        
        if show_power_density and power_density_col in city_data.columns:
            fig_power = create_comparison_chart(city_data, selected_cities, power_density_col, "line")
            st.plotly_chart(fig_power, use_container_width=True)
    
    # Seasonal Analysis
    if show_seasonal_analysis:
        st.subheader("Seasonal Wind Patterns")
        
        seasonal_data = city_data.groupby(['location', 'season'])[speed_col].agg(['mean', 'std']).reset_index()
        
        fig_seasonal = px.bar(
            seasonal_data,
            x='season',
            y='mean',
            color='location' if analysis_mode == "Multi-City Comparison" else None,
            title="Seasonal Average Wind Speeds",
            labels={'mean': 'Average Wind Speed (m/s)', 'season': 'Season'},
            barmode='group' if analysis_mode == "Multi-City Comparison" else 'relative'
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)

with tab2:
    st.subheader("Wind Resource Classification")
    
    # Resource class distribution
    if resource_class_col in city_data.columns:
        resource_summary = city_data.groupby(['location', resource_class_col]).size().reset_index(name='count')
        resource_summary['percentage'] = resource_summary.groupby('location')['count'].transform(lambda x: x / x.sum() * 100)
        
        fig_resource = px.pie(
            resource_summary[resource_summary['location'] == selected_cities[0]] if analysis_mode == "Single City" else resource_summary,
            values='count',
            names=resource_class_col,
            title="Wind Resource Classification Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_resource, use_container_width=True)
    
    # Turbine operational status
    if show_turbine_operational and turbine_operational_col in city_data.columns:
        operational_summary = city_data.groupby(['location', turbine_operational_col]).size().reset_index(name='days')
        operational_summary['percentage'] = operational_summary.groupby('location')['days'].transform(lambda x: x / x.sum() * 100)
        
        fig_operational = px.bar(
            operational_summary,
            x='location',
            y='percentage',
            color=turbine_operational_col,
            title="Turbine Operational Status (%)",
            labels={'percentage': 'Percentage of Days (%)', 'location': 'City'},
            color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1']
        )
        st.plotly_chart(fig_operational, use_container_width=True)
    
    # Power density analysis
    if show_power_density and power_density_col in city_data.columns:
        st.subheader("Power Density Analysis")
        
        # Box plot for power density comparison
        fig_power_box = px.box(
            city_data,
            x='location',
            y=power_density_col,
            title="Power Density Distribution by City",
            labels={power_density_col: 'Power Density (W/m²)', 'location': 'City'}
        )
        st.plotly_chart(fig_power_box, use_container_width=True)

with tab3:
    st.subheader("Statistical Summary")
    
    # Calculate statistics
    speed_columns = [col for col in city_data.columns if col.startswith('wind_speed_') and not col.endswith('_max') and not col.endswith('_min')]
    
    for city in selected_cities:
        city_subset = city_data[city_data['location'] == city]
        
        st.write(f"#### {city}")
        
        # Create statistics table
        stats_data = []
        for col in speed_columns:
            if col in city_subset.columns:
                stats = {
                    'Parameter': col.replace('wind_speed_', '').replace('_', ' ').title(),
                    'Mean (m/s)': f"{city_subset[col].mean():.2f}",
                    'Median (m/s)': f"{city_subset[col].median():.2f}",
                    'Std Dev (m/s)': f"{city_subset[col].std():.2f}",
                    'Min (m/s)': f"{city_subset[col].min():.2f}",
                    'Max (m/s)': f"{city_subset[col].max():.2f}",
                    'Calm Days (%)': f"{((city_subset[col] < 2).mean() * 100):.1f}",
                    'Operational Days (%)': f"{(((city_subset[col] >= 3) & (city_subset[col] <= 25)).mean() * 100):.1f}"
                }
                stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Wind speed distribution histogram
        fig_hist = px.histogram(
            city_subset,
            x=speed_col,
            nbins=30,
            title=f"Wind Speed Distribution - {city}",
            labels={speed_col: 'Wind Speed (m/s)', 'count': 'Frequency'},
            marginal="box"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

with tab4:
    st.subheader("Wind Rose Analysis")
    
    if selected_height != "2m" and direction_col in city_data.columns:
        for city in selected_cities:
            city_subset = city_data[city_data['location'] == city]
            if not city_subset.empty and not city_subset[direction_col].isna().all():
                wind_rose_fig = create_wind_rose(
                    city_subset, 
                    speed_col, 
                    direction_col, 
                    f"Wind Rose - {city}"
                )
                st.plotly_chart(wind_rose_fig, use_container_width=True)
            else:
                st.warning(f"No wind direction data available for {city}")
    else:
        st.info("Wind direction data is not available for 2m height measurements.")

with tab5:
    st.subheader("Raw Data Table")
    
    # Display options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_all_columns = st.checkbox("Show All Columns", value=False)
    
    with col2:
        rows_to_show = st.selectbox("Rows to Display", [100, 500, 1000, "All"], index=0)
    
    with col3:
        sort_by = st.selectbox("Sort By", ['date', 'location', speed_col], index=0)
    
    # Prepare display data
    display_data = city_data.copy()
    
    if not show_all_columns:
        essential_columns = [
            'date', 'location', speed_col, direction_col,
            power_density_col, resource_class_col, turbine_operational_col,
            'temperature_2m', 'relative_humidity_2m', 'precipitation'
        ]
        display_data = display_data[[col for col in essential_columns if col in display_data.columns]]
    
    display_data = display_data.sort_values(sort_by)
    
    if rows_to_show != "All":
        display_data = display_data.head(rows_to_show)
    
    st.dataframe(display_data, use_container_width=True)
    
    # Export functionality
    st.subheader("Export Data")
    
    export_data_filtered = city_data.copy()
    
    if st.button("Generate Export File"):
        export_content = export_data(export_data_filtered, export_format, f"wind_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if export_format == "CSV":
            st.download_button(
                label=f"Download CSV",
                data=export_content,
                file_name=f"wind_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            st.download_button(
                label=f"Download Excel",
                data=export_content,
                file_name=f"wind_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif export_format == "JSON":
            st.download_button(
                label=f"Download JSON",
                data=export_content,
                file_name=f"wind_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; padding: 2rem; 
                background: linear-gradient(45deg, #222, #444); 
                border-radius: 15px; 
                margin-top: 2rem; 
                color: #eee; 
                font-family: Arial, sans-serif;'>
        <h4 style='color: #f9a825;'>Wind Energy Analytics Dashboard
</h4>
        <p><strong>Data Updated:</strong> {df['date'].max().strftime('%B %d, %Y')} | 
        <strong>Analysis Period:</strong> {df['date'].min().strftime('%Y')} - {df['date'].max().strftime('%Y')} | 
        <p><em>Powered by NASA POWER Data • Built with Streamlit & Plotly</em></p>
    </div>
    """,

    unsafe_allow_html=True
)
