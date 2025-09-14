import streamlit as st
import pandas as pd
from datetime import datetime

# import cities data
data = pd.read_csv("data/processed/comprehensive_wind_data_all_cities_2014_2024.csv")
data["date"] = pd.to_datetime(data["date"], errors="coerce", dayfirst=True)
data = data.dropna(subset=["date"])
cities = data.location.dropna().unique()

df = data.sort_values(["location", "date"]).copy()
years_available = sorted(df["date"].dt.year.unique())


# set page
st.set_page_config(page_title="Wind Energy ", layout="wide")

def get_lat_long(city_name: str, df: pd.DataFrame):
    city_coords = df.loc[df["location"] == city_name, ["latitude", "longitude"]].dropna()
    if not city_coords.empty:
        lat = city_coords.iloc[0]["latitude"]
        lon = city_coords.iloc[0]["longitude"]
        return lat, lon
    else:
        return None, None

# setting required values to NONE
selected_speed = None
selected_speed_type = None
# Sidebar (Left Nav)
with st.sidebar:
    st.title("Select Parameters")
    selected_city = st.selectbox("City", cities, index=None, placeholder="Select City")
    datatype = st.selectbox("Data type",["Wind Speed"],index=None, placeholder="Select Data Type")
    if datatype == "Wind Speed":
        selected_speed = st.radio("Wind Speed At:", ["2m", "10m", "50m"], index=None, horizontal=True)
        selected_speed_type = st.radio("Wind Speed Type:", ["Avg", "Max", "Min"], index=0, horizontal=True).lower()

    filter_mode = st.selectbox("Time filter", ["All", "Specific year", "Custom"], index=None, placeholder="Select Filter")
    selected_year, custom_range = None, None
    if filter_mode == "Specific year":
        selected_year = st.selectbox("Year", years_available, index=len(years_available)-1)
    elif filter_mode == "Custom":
        min_d, max_d = df["date"].min().date(), df["date"].max().date()
        custom_range = st.date_input("Select range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

# Main Page
st.title("Wind Energy Feasibility Dashboard")
st.markdown('<p style="font-size:24px;">A Data-Driven Approach to Site Selection.</p>', unsafe_allow_html=True)
st.markdown("---")

today = df["date"].max().date()
start, end = df["date"].min().date(), today
if filter_mode == "Specific year" and selected_year is not None:
    start, end = datetime(selected_year, 1, 1).date(), datetime(selected_year, 12, 31).date()
elif filter_mode == "Custom" and isinstance(custom_range, (list, tuple)) and len(custom_range) == 2:
    start, end = custom_range if custom_range[0] <= custom_range[1] else (custom_range[1], custom_range[0])

if selected_speed is not None:
    if selected_speed_type == "avg":
        col = f"wind_speed_{selected_speed}"
    else:
        col = f"wind_speed_{selected_speed}_{selected_speed_type}"
else:
    col = None


if selected_city is not None and selected_speed is not None and col is not None and filter_mode is not None:
    mask = (
        (df["location"] == selected_city) &
        (df["date"].dt.date >= start) &
        (df["date"].dt.date <= end)
    )
    city_df = (
        df.loc[mask, ["date", col]]
        .set_index("date")
        .rename(columns={col: f"Wind ({selected_speed})"})
        .sort_index()
    )
    latitude, longitude = get_lat_long(selected_city, data)
    st.subheader(f"Historical Wind Data of {selected_city} ({latitude},{longitude})")
    if city_df.empty:
        st.warning("No data for the selected city and time range.")
    else:
        st.write(f"{(selected_speed_type).capitalize()} Wind Speed")
        st.line_chart(city_df)
        st.caption(f"{selected_city} • {start} → {end}")
        if selected_speed is not "2m":
            dir_col = f"wind_direction_{selected_speed}"
            direction_df = (
                df.loc[mask, ["date", dir_col]]
                .set_index("date")
                .rename(columns={col: f"Wind Direction ({selected_speed})"})
                .sort_index())
            st.write("Avg Wind Direction")
            st.line_chart(direction_df)
        else:
            st.warning("Direction Data not available for 2m height")


    st.info("Last Updated on 01/09/2025")
else:
    st.info("Select all the fields to view historical data.")
