# Energy Production Dashboard - Final Optimized Version

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List

# Import data utilities
from utils.data_loader import load_wind_data, get_location_list

# Configure page
st.set_page_config(
    page_title="Energy Production Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'production_initialized' not in st.session_state:
        st.session_state.production_initialized = True

initialize_session_state()

# Load data with caching
@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process_production_data():
    """Load and prepare wind data for energy production analysis."""
    try:
        data = load_wind_data()
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['year'] = df['date'].dt.year.astype('Int64')
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.strftime('%B')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False) 
def get_production_metadata(df):
    """Get metadata for production analysis."""
    if df.empty:
        return [], []
    
    cities = get_location_list(df)
    years = sorted([int(y) for y in df['year'].dropna().unique() if y > 1900])
    return cities, years

# Optimized Wind Energy Calculator
class WindEnergyCalculator:
    """
    Optimized wind energy production calculator.
    
    Sources:
    - Wind Power Formula: P = 0.5 × ρ × A × v³ (Basic wind energy physics)
    - Power Law: v_h = v_r × (h/h_r)^α (IEC 61400-12-1 Standard)
    - Capacity Factor: CF = Actual Energy / (Rated Power × Hours) × 100%
    """
    
    def __init__(self):
        # Optimized turbine database - essential models only
        self.turbine_models = {
            '1.5MW Turbine': {
                'rated_power': 1500,  # kW
                'rotor_diameter': 82,  # meters
                'hub_height': 80,  # meters
                'cut_in': 3.0,  # m/s
                'rated_speed': 12.0,  # m/s
                'cut_out': 25.0,  # m/s
            },
            '2.0MW Turbine': {
                'rated_power': 2000,  # kW
                'rotor_diameter': 90,  # meters
                'hub_height': 85,  # meters
                'cut_in': 3.0,  # m/s
                'rated_speed': 12.5,  # m/s
                'cut_out': 25.0,  # m/s
            },
            '3.0MW Turbine': {
                'rated_power': 3000,  # kW
                'rotor_diameter': 112,  # meters
                'hub_height': 90,  # meters
                'cut_in': 3.0,  # m/s
                'rated_speed': 13.0,  # m/s
                'cut_out': 25.0,  # m/s
            }
        }
        
        # Terrain types for height adjustment
        self.terrain_alpha = {
            'water': 0.10,
            'open': 0.15,
            'rough': 0.25,
            'urban': 0.35
        }
    
    def calculate_power_output(self, wind_speed: float, turbine_model: str) -> float:
        """
        Calculate power output using optimized power curve.
        
        Args:
            wind_speed: Wind speed at hub height (m/s)
            turbine_model: Selected turbine model
            
        Returns:
            Power output in kW
        """
        if turbine_model not in self.turbine_models:
            turbine_model = '2.0MW Turbine'
        
        specs = self.turbine_models[turbine_model]
        
        # Handle operational limits
        if wind_speed < specs['cut_in'] or wind_speed > specs['cut_out']:
            return 0.0
        
        # Power calculation: cubic relationship until rated speed
        if wind_speed <= specs['rated_speed']:
            power_ratio = ((wind_speed - specs['cut_in']) / 
                          (specs['rated_speed'] - specs['cut_in'])) ** 3
            return specs['rated_power'] * power_ratio
        
        # Constant rated power between rated and cut-out
        return specs['rated_power']
    
    def adjust_wind_speed_for_height(self, wind_speed: float, from_height: float, 
                                   to_height: float, terrain: str = 'open') -> float:
        """
        Adjust wind speed from measurement height to hub height using power law.
        
        Args:
            wind_speed: Measured wind speed (m/s)
            from_height: Measurement height (m)
            to_height: Target hub height (m)
            terrain: Terrain type
            
        Returns:
            Adjusted wind speed at hub height (m/s)
        """
        alpha = self.terrain_alpha.get(terrain, 0.15)
        
        if from_height <= 0 or to_height <= 0 or wind_speed < 0:
            return wind_speed
        
        adjusted_speed = wind_speed * ((to_height / from_height) ** alpha)
        return max(0, min(50, adjusted_speed))
    
    def calculate_capacity_factor(self, actual_energy_mwh: float, rated_power_mw: float, 
                                hours: int) -> float:
        """Calculate capacity factor percentage."""
        if rated_power_mw <= 0 or hours <= 0:
            return 0.0
        
        max_possible_energy = rated_power_mw * hours
        capacity_factor = (actual_energy_mwh / max_possible_energy) * 100
        
        return max(0, min(100, capacity_factor))

# Initialize calculator
wind_calc = WindEnergyCalculator()

# Load data
df = load_and_process_production_data()
cities, years = get_production_metadata(df)

# Check if data available
if df.empty or not cities:
    st.error("No wind data available. Please check your data source.")
    st.stop()

# Main dashboard
st.markdown('<h1 style="text-align: center;">Wind Energy Production Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;">Energy Production Analysis & Performance Assessment</h3>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.title("Configuration")
    
    # Location selection
    st.subheader("Location")
    analysis_type = st.radio("Analysis Type", ["Single Location", "Compare Locations"], key="prod_analysis")
    
    if analysis_type == "Single Location":
        selected_cities = [st.selectbox("Select Location", cities, key="prod_city")]
    else:
        selected_cities = st.multiselect("Select Locations (max 5)", cities, 
                                       default=cities[:3] if len(cities) >= 3 else [cities[0]], 
                                       max_selections=5, key="prod_cities")
    
    # Wind farm configuration
    st.subheader("Wind Farm Setup")
    turbine_model = st.selectbox("Turbine Model", 
                               list(wind_calc.turbine_models.keys()), 
                               index=1, key="prod_turbine")
    
    num_turbines = st.number_input("Number of Turbines", min_value=1, max_value=100, 
                                 value=20, key="prod_num_turbines")
    
    # Wind data configuration
    st.subheader("Wind Data")
    measurement_height = st.selectbox("Measurement Height", 
                                    ["10m", "50m", "100m"], 
                                    index=1, key="prod_height")
    
    terrain_type = st.selectbox("Terrain Type",
                              ["water", "open", "rough", "urban"],
                              index=1, key="prod_terrain")
    
    # Time period
    st.subheader("Time Period")
    if years:
        selected_year = st.selectbox("Analysis Year", years, 
                                   index=len(years)-1, key="prod_year")
        filter_data = df[df['year'] == selected_year]
    else:
        filter_data = df

# Data validation
if not selected_cities:
    st.warning("Please select at least one location.")
    st.stop()

# Filter data for selected cities
city_data = filter_data[filter_data['location'].isin(selected_cities)].copy()

if city_data.empty:
    st.error("No data available for selected criteria.")
    st.stop()

# Get measurement height and wind speed column
height_num = int(measurement_height.replace('m', ''))
wind_speed_col = f"wind_speed_{measurement_height}"

if wind_speed_col not in city_data.columns:
    st.error(f"Wind speed data for {measurement_height} not available.")
    st.stop()

# Get turbine specifications
turbine_specs = wind_calc.turbine_models[turbine_model]
rated_power_mw = (turbine_specs['rated_power'] * num_turbines) / 1000

# Display turbine info
st.sidebar.info(f"""
**Turbine Specifications:**
- Power: {turbine_specs['rated_power']:,} kW
- Rotor: {turbine_specs['rotor_diameter']} m
- Hub Height: {turbine_specs['hub_height']} m
- Cut-in: {turbine_specs['cut_in']} m/s
- Rated: {turbine_specs['rated_speed']} m/s
- Cut-out: {turbine_specs['cut_out']} m/s

**Wind Farm Total:**
- Capacity: {rated_power_mw:.1f} MW
- Turbines: {num_turbines}
""")

# Calculate production for each location
production_summary = []

for city in selected_cities:
    city_subset = city_data[city_data['location'] == city].copy()
    
    if city_subset.empty:
        continue
    
    # Adjust wind speed to hub height
    city_subset['wind_speed_hub'] = city_subset[wind_speed_col].apply(
        lambda ws: wind_calc.adjust_wind_speed_for_height(
            ws, height_num, turbine_specs['hub_height'], terrain_type
        )
    )
    
    # Calculate power output
    city_subset['single_turbine_power_kw'] = city_subset['wind_speed_hub'].apply(
        lambda ws: wind_calc.calculate_power_output(ws, turbine_model)
    )
    
    # Calculate total wind farm power and energy
    city_subset['total_power_kw'] = city_subset['single_turbine_power_kw'] * num_turbines
    city_subset['total_power_mw'] = city_subset['total_power_kw'] / 1000
    
    # Energy calculation (assuming hourly data)
    city_subset['hourly_energy_mwh'] = city_subset['total_power_mw']
    
    # Calculate summary statistics
    total_hours = len(city_subset)
    total_energy_mwh = city_subset['hourly_energy_mwh'].sum()
    
    # Capacity factor
    capacity_factor = wind_calc.calculate_capacity_factor(
        total_energy_mwh, rated_power_mw, total_hours
    )
    
    # Scale to annual if less than full year
    if total_hours < 8760:
        annual_energy_mwh = total_energy_mwh * (8760 / total_hours)
    else:
        annual_energy_mwh = total_energy_mwh
    
    # Calculate key performance metrics
    avg_power_output_mw = city_subset['total_power_mw'].mean()
    max_power_output_mw = city_subset['total_power_mw'].max()
    
    # Wind speed statistics
    avg_wind_measured = city_subset[wind_speed_col].mean()
    avg_wind_hub = city_subset['wind_speed_hub'].mean()
    
    production_summary.append({
        'Location': city,
        'Annual Energy (MWh)': annual_energy_mwh,
        'Capacity Factor (%)': capacity_factor,
        'Avg Power Output (MW)': avg_power_output_mw,
        'Max Power Output (MW)': max_power_output_mw,
        'Avg Wind Speed (m/s)': avg_wind_measured,
        'Avg Hub Wind Speed (m/s)': avg_wind_hub,
        'Operating Hours': total_hours,
        'Energy Yield (MWh/MW)': annual_energy_mwh / rated_power_mw if rated_power_mw > 0 else 0
    })

# Create summary dataframe
summary_df = pd.DataFrame(production_summary)

if summary_df.empty:
    st.error("No production data calculated. Check your inputs.")
    st.stop()

# Display results
st.subheader("Wind Farm Performance Summary")

if analysis_type == "Single Location":
    city_results = summary_df.iloc[0]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Annual Energy", 
                f"{city_results['Annual Energy (MWh)']:,.0f} MWh")
    
    with col2:
        st.metric("Capacity Factor", 
                f"{city_results['Capacity Factor (%)']:.1f}%")
    
    with col3:
        st.metric("Average Power", 
                f"{city_results['Avg Power Output (MW)']:.1f} MW")
    
    with col4:
        st.metric("Wind Farm Capacity", 
                f"{rated_power_mw:.1f} MW")
    
    with col5:
        st.metric("Energy Yield", 
                f"{city_results['Energy Yield (MWh/MW)']:,.0f} MWh/MW")

else:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Locations Compared", len(selected_cities))
    
    with col2:
        st.metric("Total Energy", 
                f"{summary_df['Annual Energy (MWh)'].sum():,.0f} MWh")
    
    with col3:
        st.metric("Best Capacity Factor", 
                f"{summary_df['Capacity Factor (%)'].max():.1f}%")
    
    with col4:
        st.metric("Average CF", 
                f"{summary_df['Capacity Factor (%)'].mean():.1f}%")

# Detailed results table
st.subheader("Detailed Performance Results")
st.dataframe(summary_df.round(2), use_container_width=True)

# Analysis tabs - simplified to 3 essential tabs
tab1, tab2, tab3 = st.tabs([
    "Production Analysis", 
    "Power Curves & Performance", 
    "Location Comparison"
])

with tab1:
    st.subheader("Energy Production Analysis")
    
    if analysis_type == "Single Location":
        city = selected_cities[0]
        city_subset = city_data[city_data['location'] == city].copy()
        
        # Add calculated columns
        city_subset['wind_speed_hub'] = city_subset[wind_speed_col].apply(
            lambda ws: wind_calc.adjust_wind_speed_for_height(
                ws, height_num, turbine_specs['hub_height'], terrain_type
            )
        )
        city_subset['total_power_mw'] = city_subset['wind_speed_hub'].apply(
            lambda ws: wind_calc.calculate_power_output(ws, turbine_model) * num_turbines / 1000
        )
        
        # Monthly production pattern
        if not city_subset.empty:
            monthly_data = city_subset.groupby('month_name').agg({
                'total_power_mw': 'mean',
                'wind_speed_hub': 'mean'
            }).reset_index()
            
            # Ensure correct month order
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_data['month_name'] = pd.Categorical(monthly_data['month_name'], categories=month_order, ordered=True)
            monthly_data = monthly_data.sort_values('month_name')
            
            # Monthly power output chart
            fig_monthly = px.bar(monthly_data, x='month_name', y='total_power_mw',
                               title=f"Average Monthly Power Output - {city}",
                               labels={'total_power_mw': 'Average Power (MW)', 'month_name': 'Month'},
                               color='total_power_mw',
                               color_continuous_scale='Viridis')
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Power output vs wind speed scatter (sample for performance)
            sample_size = min(1000, len(city_subset))
            sample_data = city_subset.sample(sample_size) if len(city_subset) > sample_size else city_subset
            
            fig_scatter = px.scatter(sample_data, 
                                   x='wind_speed_hub', y='total_power_mw',
                                   title=f"Power Output vs Wind Speed - {city}",
                                   labels={'wind_speed_hub': 'Hub Wind Speed (m/s)', 
                                         'total_power_mw': 'Power Output (MW)'},
                                   opacity=0.6,
                                   color='total_power_mw',
                                   color_continuous_scale='Plasma')
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Wind speed distribution
            fig_wind_dist = px.histogram(city_subset, x='wind_speed_hub', nbins=30,
                                       title=f"Hub Height Wind Speed Distribution - {city}",
                                       labels={'wind_speed_hub': 'Wind Speed at Hub Height (m/s)', 
                                             'count': 'Frequency'},
                                       marginal="box")
            st.plotly_chart(fig_wind_dist, use_container_width=True)
        else:
            st.warning(f"No data available for {city}")
    
    else:
        # Multi-location comparison
        fig_comparison = px.bar(summary_df, x='Location', y='Annual Energy (MWh)',
                              color='Capacity Factor (%)', 
                              title="Annual Energy Production Comparison",
                              color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Capacity factor comparison
        fig_cf = px.bar(summary_df, x='Location', y='Capacity Factor (%)',
                       title="Capacity Factor Comparison",
                       color='Capacity Factor (%)',
                       color_continuous_scale='RdYlGn')
        fig_cf.add_hline(y=35, line_dash="dash", line_color="red", 
                        annotation_text="Good Site Threshold (35%)")
        st.plotly_chart(fig_cf, use_container_width=True)

with tab2:
    st.subheader("Turbine Power Curve & Performance")
    
    # Generate power curve data
    wind_speeds = np.arange(0, 30, 0.5)
    power_outputs = [wind_calc.calculate_power_output(ws, turbine_model) for ws in wind_speeds]
    
    curve_df = pd.DataFrame({
        'Wind Speed (m/s)': wind_speeds,
        'Power Output (kW)': power_outputs,
        'Power Output (MW)': np.array(power_outputs) / 1000
    })
    
    # Power curve chart
    fig_curve = px.line(curve_df, x='Wind Speed (m/s)', y='Power Output (kW)',
                       title=f"Power Curve - {turbine_model}",
                       labels={'Power Output (kW)': 'Single Turbine Power Output (kW)'})
    
    # Add operating regions
    fig_curve.add_vline(x=turbine_specs['cut_in'], line_dash="dash", line_color="green",
                       annotation_text=f"Cut-in ({turbine_specs['cut_in']} m/s)")
    fig_curve.add_vline(x=turbine_specs['rated_speed'], line_dash="dash", line_color="orange",
                       annotation_text=f"Rated ({turbine_specs['rated_speed']} m/s)")
    fig_curve.add_vline(x=turbine_specs['cut_out'], line_dash="dash", line_color="red",
                       annotation_text=f"Cut-out ({turbine_specs['cut_out']} m/s)")
    
    st.plotly_chart(fig_curve, use_container_width=True)
    
    # Wind farm total power curve
    curve_df['Total Farm Power (MW)'] = curve_df['Power Output (MW)'] * num_turbines
    
    fig_farm_curve = px.line(curve_df, x='Wind Speed (m/s)', y='Total Farm Power (MW)',
                           title=f"Wind Farm Power Curve ({num_turbines} Turbines)",
                           labels={'Total Farm Power (MW)': 'Total Wind Farm Power (MW)'})
    fig_farm_curve.add_hline(y=rated_power_mw, line_dash="dash", line_color="red",
                           annotation_text=f"Rated Capacity ({rated_power_mw:.1f} MW)")
    st.plotly_chart(fig_farm_curve, use_container_width=True)
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Turbine Performance")
        rotor_area = np.pi * (turbine_specs['rotor_diameter']/2)**2
        specific_power = (turbine_specs['rated_power'] * 1000) / rotor_area
        
        perf_data = {
            'Metric': ['Rotor Swept Area', 'Specific Power', 'Cut-in Wind Speed', 
                      'Rated Wind Speed', 'Cut-out Wind Speed'],
            'Value': [
                f"{rotor_area:.0f} m²",
                f"{specific_power:.0f} W/m²",
                f"{turbine_specs['cut_in']} m/s",
                f"{turbine_specs['rated_speed']} m/s",
                f"{turbine_specs['cut_out']} m/s"
            ]
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
    
    with col2:
        st.subheader("Wind Farm Performance")
        if not summary_df.empty:
            avg_cf = summary_df['Capacity Factor (%)'].mean()
            
            # Performance rating
            if avg_cf >= 40:
                rating = "Excellent"
            elif avg_cf >= 30:
                rating = "Good"
            elif avg_cf >= 20:
                rating = "Fair"
            else:
                rating = "Poor"
            
            st.metric("Average Capacity Factor", f"{avg_cf:.1f}%")
            st.metric("Performance Rating", rating)
            st.metric("Total Annual Energy", f"{summary_df['Annual Energy (MWh)'].sum():,.0f} MWh")

with tab3:
    st.subheader("Location Performance Comparison")
    
    if len(selected_cities) > 1:
        # Capacity factor vs wind speed relationship
        fig_cf_wind = px.scatter(summary_df, x='Avg Hub Wind Speed (m/s)', y='Capacity Factor (%)',
                               color='Location', size='Annual Energy (MWh)',
                               title="Capacity Factor vs Average Wind Speed",
                               hover_data=['Location', 'Energy Yield (MWh/MW)'])
        fig_cf_wind.add_vline(x=turbine_specs['rated_speed'], line_dash="dash", line_color="red",
                             annotation_text="Turbine Rated Speed")
        st.plotly_chart(fig_cf_wind, use_container_width=True)
        
        # Performance ranking table
        st.subheader("Location Ranking")
        
        ranking_metric = st.selectbox("Rank by:", 
                                   ["Capacity Factor (%)", "Annual Energy (MWh)", "Energy Yield (MWh/MW)"], 
                                   key="ranking_metric")
        
        ranked_df = summary_df.sort_values(ranking_metric, ascending=False).reset_index(drop=True)
        ranked_df.index += 1
        ranked_df.index.name = 'Rank'
        
        # Display ranking
        display_cols = ['Location', ranking_metric, 'Avg Hub Wind Speed (m/s)']
        st.dataframe(ranked_df[display_cols].round(2), use_container_width=True)
        
        # Best vs worst comparison
        if len(ranked_df) >= 2:
            best_location = ranked_df.iloc[0]
            worst_location = ranked_df.iloc[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **Best Performing Location:**
                - {best_location['Location']}
                - {ranking_metric}: {best_location[ranking_metric]:.1f}
                - Avg Wind Speed: {best_location['Avg Hub Wind Speed (m/s)']:.1f} m/s
                """)
            
            with col2:
                st.warning(f"""
                **Lowest Performing Location:**
                - {worst_location['Location']}
                - {ranking_metric}: {worst_location[ranking_metric]:.1f}
                - Avg Wind Speed: {worst_location['Avg Hub Wind Speed (m/s)']:.1f} m/s
                """)
    
    else:
        st.info("Select multiple locations to enable comparison analysis.")
    
    # Industry benchmarks
    st.subheader("Industry Benchmarks")
    
    avg_cf = summary_df['Capacity Factor (%)'].mean()
    
    benchmark_data = pd.DataFrame({
        'Technology': ['Your Wind Farm', 'Onshore Wind (Excellent)', 'Onshore Wind (Good)', 
                      'Onshore Wind (Average)', 'Solar PV', 'Natural Gas'],
        'Typical Capacity Factor (%)': [avg_cf, 45, 35, 25, 25, 85],
        'Performance Level': ['Current', 'Best Sites', 'Good Sites', 'Average Sites', 'Reference', 'Reference']
    })
    
    fig_benchmark = px.bar(benchmark_data, x='Technology', y='Typical Capacity Factor (%)',
                         color='Performance Level',
                         title="Capacity Factor Benchmarking")
    st.plotly_chart(fig_benchmark, use_container_width=True)

# Export functionality
st.subheader("Export Results")

col1, col2 = st.columns(2)

with col1:
    if st.button("Export Summary"):
        csv_data = summary_df.to_csv(index=False)
        st.download_button(
            "Download Summary CSV",
            csv_data,
            file_name=f"wind_production_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Export Power Curve Data"):
        wind_speeds = np.arange(0, 30, 0.5)
        power_outputs = [wind_calc.calculate_power_output(ws, turbine_model) for ws in wind_speeds]
        curve_data = pd.DataFrame({
            'Wind Speed (m/s)': wind_speeds,
            'Single Turbine Power (kW)': power_outputs,
            'Wind Farm Power (MW)': np.array(power_outputs) * num_turbines / 1000
        })
        
        csv_curve = curve_data.to_csv(index=False)
        st.download_button(
            "Download Power Curve CSV",
            csv_curve,
            file_name=f"power_curve_{turbine_model.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Technical notes
with st.expander("Technical Notes & Calculation Methods"):
    st.markdown(f"""
    ### Calculation Methods
    
    **Wind Speed Height Adjustment:**
    - Formula: v_hub = v_measured × (h_hub / h_measured)^α
    - Alpha values by terrain:
      - Water: 0.10 (very smooth)
      - Open: 0.15 (agricultural, grassland)
      - Rough: 0.25 (forests, suburbs)
      - Urban: 0.35 (cities, industrial)
    - Source: IEC 61400-12-1 Standard
    
    **Power Output Calculation:**
    - Cubic relationship: P ∝ v³ between cut-in and rated wind speed
    - Constant rated power between rated and cut-out wind speed
    - Zero power below cut-in and above cut-out wind speeds
    
    **Capacity Factor:**
    - Formula: CF = Actual Energy / (Rated Power × Hours) × 100%
    - Represents turbine utilization efficiency
    - Industry benchmark: 25-45% for onshore wind
    
    **Energy Yield:**
    - Annual energy production per MW of installed capacity
    - Useful metric for comparing different locations
    - Typical range: 2,000-4,000 MWh/MW/year
    
    ### Current Analysis Configuration
    - **Turbine Model:** {turbine_model}
    - **Number of Turbines:** {num_turbines}
    - **Total Capacity:** {rated_power_mw:.1f} MW
    - **Analysis Year:** {selected_year if years else 'All available data'}
    - **Measurement Height:** {measurement_height}
    - **Hub Height:** {turbine_specs['hub_height']} m
    - **Terrain Type:** {terrain_type.title()}
    
    ### Key Assumptions
    - Hourly wind data intervals
    - No wake effects between turbines (simplified)
    - 100% turbine availability (no maintenance downtime)
    - Standard atmospheric conditions
    - No grid curtailment or transmission losses
    """)


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
        <h4 style='color: #f9a825;'>Wind Energy Production Dashboard</h4>
        <p><strong>Data Updated:</strong> {str(df['date'].max())[0:10]} | 
        <strong>Analysis Period:</strong> {df['date'].min().strftime('%Y')} - {df['date'].max().strftime('%Y')} | 
        <p><em>Powered by NASA POWER Data • Built with Streamlit & Plotly</em></p>
    </div>
    """,

    unsafe_allow_html=True
)
