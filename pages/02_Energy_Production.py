import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Tuple, Any
from utils.data_loader import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure page with enhanced settings
st.set_page_config(
    page_title="🌪️ Wind Energy Feasibility Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/wind-energy-dashboard',
        'Report a Bug': 'https://github.com/your-repo/wind-energy-dashboard/issues',
        'About': "# Wind Energy Feasibility Dashboard\nProfessional wind resource assessment platform powered by NASA POWER data."
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
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

df = load_wind_data()
cities = get_location_list(df)
years_available = sorted(df['date'].dt.year.unique().tolist())

def main():
    """Main dashboard application."""
    
    # Header with enhanced styling
    st.markdown('<h1 class="main-header"> Wind Energy Production Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Wind Resource Assessment & Site Selection Platform</p>', unsafe_allow_html=True)
    




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

if __name__ == "__main__":
    main()