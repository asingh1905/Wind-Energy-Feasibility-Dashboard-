"""
WindDataCollector: Fetches and parses historical wind data from NASA POWER API
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WindDataCollector:
    """
    Collects historical wind data for given locations using NASA POWER API.
    Saves raw JSON and consolidated CSV.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # NASA POWER API base URL and default parameters
        self.nasa_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.nasa_base_params = {
            "community": "RE",           # Renewable Energy community
            "format": "JSON",
            "parameters": "WS10M,WS50M"  # Wind speed at 10m & 50m
        }
        self.request_delay = 1  # seconds between API calls

    def load_locations(self, locations_file: str) -> pd.DataFrame:
        """
        Load locations CSV with columns: name, latitude, longitude, state, type
        """
        df = pd.read_csv(locations_file)
        required = {"name", "latitude", "longitude"}
        if not required.issubset(df.columns):
            raise ValueError(f"Locations file must contain columns: {required}")
        return df

    def fetch_nasa_data(
        self,
        name: str,
        lat: float,
        lon: float,
        start: str,
        end: str
    ) -> Optional[Dict]:
        """
        Fetch JSON from NASA POWER API for one location and date range.
        Saves the raw JSON file.
        """
        params = dict(self.nasa_base_params)
        params.update({
            "latitude": lat,
            "longitude": lon,
            "start": start,
            "end": end
        })
        try:
            logger.info(f"Fetching NASA data for {name} ({start}->{end})")
            resp = requests.get(self.nasa_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            raw_file = os.path.join(
                self.raw_dir,
                f"nasa_{name}_{start}_{end}.json"
            )
            with open(raw_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved raw JSON to {raw_file}")
            time.sleep(self.request_delay)
            return data

        except Exception as e:
            logger.error(f"Error fetching {name}: {e}")
            return None

    def parse_nasa_data(self, data: Dict, name: str) -> pd.DataFrame:
        """
        Parse NASA JSON into DataFrame with columns:
         date, wind_speed_10m, wind_speed_50m, location, latitude, longitude
        """
        try:
            params = data["properties"]["parameter"]
            df = pd.DataFrame(params).reset_index()
            df.rename(columns={"index": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df.rename(columns={
                "WS10M": "wind_speed_10m",
                "WS50M": "wind_speed_50m"
            }, inplace=True)

            coord = data["geometry"]["coordinates"]
            df["location"] = name
            df["latitude"] = coord[1]
            df["longitude"] = coord[0]
            return df

        except Exception as e:
            logger.error(f"Error parsing data for {name}: {e}")
            return pd.DataFrame()

    def collect_for_location(
        self,
        row: pd.Series,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """
        Collect and parse data year by year for one location.
        """
        name = row["name"]
        lat = row["latitude"]
        lon = row["longitude"]
        all_dfs = []

        for year in range(start_year, end_year + 1):
            start = f"{year}0101"
            end = f"{year}1231"
            raw = self.fetch_nasa_data(name, lat, lon, start, end)
            if raw:
                parsed = self.parse_nasa_data(raw, name)
                if not parsed.empty:
                    parsed["year"] = year
                    all_dfs.append(parsed)

        if all_dfs:
            df_loc = pd.concat(all_dfs, ignore_index=True)
            return df_loc
        else:
            return pd.DataFrame()

    def collect_all(
        self,
        locations_file: str,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """
        Collect data for all locations and save consolidated CSV.
        """
        locs = self.load_locations(locations_file)
        all_data = []

        for _, row in locs.iterrows():
            df_loc = self.collect_for_location(row, start_year, end_year)
            if not df_loc.empty:
                all_data.append(df_loc)

        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            df_all.sort_values(["location", "date"], inplace=True)
            output = os.path.join(self.processed_dir, "wind_data_all_cities.csv")
            df_all.to_csv(output, index=False)
            logger.info(f"Saved consolidated data to {output}")
            return df_all
        else:
            logger.error("No data collected for any location")
            return pd.DataFrame()
