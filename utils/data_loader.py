import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime, timedelta

@st.cache_data(ttl=3600)
def load_wind_data(file_path: str = "data/selected_cities/selected_cities.csv") -> pd.DataFrame:
    """Load and preprocess wind data with caching."""
    try:
        df = pd.read_csv(file_path)
        # Add date column if missing for basic functionality
        if 'date' not in df.columns:
            # Create dummy date for testing
            df['date'] = pd.date_range('2014-01-01', periods=len(df), freq='D')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(['name', 'date']).reset_index(drop=True) # Use 'name' instead of 'location'
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def get_location_list(df: pd.DataFrame) -> List[str]:
    """Get unique locations from dataset."""
    return sorted(df['location'].unique().tolist())

def filter_data_by_location(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """Filter data by location."""
    return df[df['location'] == location].copy()

def filter_data_by_date_range(df: pd.DataFrame, start_date: datetime, 
                             end_date: datetime) -> pd.DataFrame:
    """Filter data by date range."""
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

def calculate_wind_statistics(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive wind statistics."""
    stats = {}
    
    for height in ['2m', '10m', '50m']:
        wind_col = f'wind_speed_{height}'
        if wind_col in df.columns:
            wind_data = df[wind_col].dropna()
            stats[height] = {
                'mean': wind_data.mean(),
                'median': wind_data.median(),
                'std': wind_data.std(),
                'min': wind_data.min(),
                'max': wind_data.max(),
                'percentile_25': wind_data.quantile(0.25),
                'percentile_75': wind_data.quantile(0.75),
                'coefficient_of_variation': wind_data.std() / wind_data.mean()
            }
    
    return stats

def prepare_data_for_sarima(df: pd.DataFrame, location: str, 
                           wind_height: str = '10m') -> pd.Series:
    """Prepare time series data for SARIMA modeling."""
    location_data = df[df['location'] == location].copy()
    location_data = location_data.set_index('date')
    
    wind_col = f'wind_speed_{wind_height}'
    if wind_col not in location_data.columns:
        raise ValueError(f"Column {wind_col} not found in data")
    
    # Resample to monthly averages for better SARIMA performance
    monthly_data = location_data[wind_col].resample('M').mean()
    
    # Handle missing values
    monthly_data = monthly_data.fillna(monthly_data.mean())
    
    return monthly_data