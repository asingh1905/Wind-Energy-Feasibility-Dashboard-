import streamlit as st
import pandas as pd
from datetime import datetime

# import cities data
data = pd.read_csv("data/processed/all_cities_INDIA cleaned.csv")
data["date"] = pd.to_datetime(data["date"], errors="coerce", dayfirst=True)
data = data.dropna(subset=["date"])
cities = data.city.dropna().unique()

df = data.sort_values(["city", "date"]).copy()
years_available = sorted(df["date"].dt.year.unique())


# set page
st.set_page_config(page_title="Wind Energy ", layout="wide")

# Sidebar (Left Nav)
with st.sidebar:
    st.title("Select Parameters")
    selected_city = st.selectbox("City", cities, index=None, placeholder="Select City")
    selected_speed = st.selectbox("Wind Speed At", ["10m", "50m"], index=None, placeholder="Select Wind Speed")

    filter_mode = st.selectbox("Time filter", ["All", "Specific year", "Custom"], index=None, placeholder="Select Filter")
    selected_year, custom_range = None, None
    if filter_mode == "Specific year":
        selected_year = st.selectbox("Year", years_available, index=len(years_available)-1)
    elif filter_mode == "Custom":
        min_d, max_d = df["date"].min().date(), df["date"].max().date()
        custom_range = st.date_input("Select range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

# Main Page
st.title("Wind Energy Feasibility Dashboard")
st.write("A Data-Driven Approach to Site Selection.")

today = df["date"].max().date()
start, end = df["date"].min().date(), today
if filter_mode == "Specific year" and selected_year is not None:
    start, end = datetime(selected_year, 1, 1).date(), datetime(selected_year, 12, 31).date()
elif filter_mode == "Custom" and isinstance(custom_range, (list, tuple)) and len(custom_range) == 2:
    start, end = custom_range if custom_range[0] <= custom_range[1] else (custom_range[1], custom_range[0])

if selected_speed is not None:
    col = "ws10" if selected_speed == "10m" else "ws50"
else:
    col = None

if selected_city is not None and selected_speed is not None and col is not None and filter_mode is not None:
    mask = (
        (df["city"] == selected_city) &
        (df["date"].dt.date >= start) &
        (df["date"].dt.date <= end)
    )
    city_df = (
        df.loc[mask, ["date", col]]
        .set_index("date")
        .rename(columns={col: f"Wind ({selected_speed})"})
        .sort_index()
    )

    st.subheader("Historical Wind")
    if city_df.empty:
        st.warning("No data for the selected city and time range.")
    else:
        st.line_chart(city_df)
        st.caption(f"{selected_city} • {start} → {end}")
else:
    st.info("Select all the fields to view historical data.")
