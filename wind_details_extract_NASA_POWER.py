# Wind Energy Extractor

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import time
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlignedWindEnergyExtractor:
    def __init__(self, max_concurrent=25):
        self.max_concurrent = max_concurrent
        self.max_params_per_call = 15
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.session = None
        
        # NASA POWER API parameters (same as yours)
        self.nasa_parameters = [
            'WS2M', 'WS10M', 'WS50M',           # wind speeds
            'WS2M_MAX', 'WS2M_MIN',             # 2m wind extremes  
            'WS10M_MAX', 'WS10M_MIN',           # 10m wind extremes
            'WS50M_MAX', 'WS50M_MIN',           # 50m wind extremes
            'WD10M', 'WD50M',                   # wind directions
            'T2M',                              # temperature 2m
            'PS',                               # surface pressure (kPa)
            'RH2M',                             # relative humidity 2m
            'PRECTOTCORR'                       # precipitation
        ]
        
        # YOUR CALCULATION CONSTANTS (exact match)
        self.air_density_standard = 1.225  # kg/m³ at sea level, 15°C
        self.gas_constant = 287.05          # J/(kg·K) for dry air
        self.cut_in_speed = 3.0             # m/s (YOUR cut-in speed)
        self.cut_out_speed = 25.0           # m/s (YOUR cut-out speed)
        self.rated_wind_speed = 12.0        # m/s (YOUR rated wind speed)
    
    async def create_session(self):
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(
            limit=35, limit_per_host=15, keepalive_timeout=30
        )
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    def split_parameters(self, parameters: List[str]) -> List[List[str]]:
        chunks = []
        for i in range(0, len(parameters), self.max_params_per_call):
            chunks.append(parameters[i:i + self.max_params_per_call])
        return chunks
    
    def calculate_air_density(self, temperature_c: float, pressure_kpa: float, humidity_pct: float = 0) -> float:
        """
        YOUR EXACT CALCULATION METHOD
        More accurate than using standard air density
        """
        try:
            # Convert temperature to Kelvin
            temp_k = temperature_c + 273.15
            
            # Convert pressure from kPa to Pa (CORRECTED!)
            pressure_pa = pressure_kpa * 1000  # NASA POWER PS is in kPa
            
            # For dry air approximation (humidity effect is small) - YOUR METHOD
            air_density = pressure_pa / (self.gas_constant * temp_k)
            
            return air_density if air_density > 0 else self.air_density_standard
        except:
            return self.air_density_standard

    def calculate_wind_power_density(self, wind_speed: float, air_density: float = None) -> float:
        """
        YOUR EXACT METHOD: WPD = 0.5 * ρ * v³
        """
        if air_density is None:
            air_density = self.air_density_standard
        return 0.5 * air_density * (wind_speed ** 3)

    def classify_wind_resource(self, avg_wind_speed: float, wind_power_density: float) -> str:
        """
        YOUR EXACT CLASSIFICATION SYSTEM
        """
        if avg_wind_speed >= 7.0 and wind_power_density >= 500:
            return "Excellent"
        elif avg_wind_speed >= 6.0 and wind_power_density >= 300:
            return "Good"
        elif avg_wind_speed >= 5.0 and wind_power_density >= 200:
            return "Fair"
        elif avg_wind_speed >= 4.0:
            return "Marginal"
        else:
            return "Poor"
    
    def get_turbine_operational_status(self, wind_speed: float) -> str:
        """
        YOUR EXACT OPERATIONAL STATUS SYSTEM
        """
        if self.cut_in_speed <= wind_speed <= self.cut_out_speed:
            return 'Operating'
        elif wind_speed < self.cut_in_speed:
            return 'Below cut-in'
        else:
            return 'Above cut-out'
    
    async def fetch_city_batch(self, city_info: Dict, param_chunk: List[str], 
                             start_date: str, end_date: str, semaphore) -> Dict:
        async with semaphore:
            try:
                params = {
                    'parameters': ','.join(param_chunk),
                    'community': 'RE',
                    'longitude': city_info['longitude'],
                    'latitude': city_info['latitude'],
                    'start': start_date,
                    'end': end_date,
                    'format': 'JSON'
                }
                
                async with self.session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'city': city_info['city'],
                            'data': data,
                            'success': True
                        }
                    else:
                        return {'city': city_info['city'], 'success': False, 'error': f"HTTP {response.status}"}
                        
            except Exception as e:
                return {'city': city_info['city'], 'success': False, 'error': str(e)}
    
    async def fetch_city_complete(self, city_info: Dict, start_date: str, end_date: str, 
                                semaphore) -> Dict:
        param_chunks = self.split_parameters(self.nasa_parameters)
        
        tasks = [
            self.fetch_city_batch(city_info, chunk, start_date, end_date, semaphore)
            for chunk in param_chunks
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined_data = {'city': city_info['city'], 'parameters': {}}
        success_count = 0
        
        for result in results:
            if isinstance(result, dict) and result.get('success'):
                if 'data' in result and 'properties' in result['data']:
                    param_data = result['data']['properties']['parameter']
                    combined_data['parameters'].update(param_data)
                    success_count += 1
        
        combined_data['success'] = success_count > 0
        combined_data['latitude'] = city_info['latitude']
        combined_data['longitude'] = city_info['longitude']
        return combined_data
    
    def process_city_data(self, raw_data: Dict) -> pd.DataFrame:
        """
        Process raw NASA POWER data using YOUR EXACT calculation methods
        """
        if not raw_data.get('success') or not raw_data.get('parameters'):
            return pd.DataFrame()
        
        try:
            first_param = list(raw_data['parameters'].keys())[0]
            dates = list(raw_data['parameters'][first_param].keys())
            
            rows = []
            for date_str in dates:
                row = {
                    'date': date_str,
                    'location': raw_data['city'],
                    'latitude': raw_data['latitude'],
                    'longitude': raw_data['longitude']
                }
                
                params = raw_data['parameters']
                
                # Extract NASA POWER parameters with YOUR column names
                row['wind_speed_2m'] = params.get('WS2M', {}).get(date_str, np.nan)
                row['wind_speed_10m'] = params.get('WS10M', {}).get(date_str, np.nan)
                row['wind_speed_50m'] = params.get('WS50M', {}).get(date_str, np.nan)
                
                row['wind_speed_2m_max'] = params.get('WS2M_MAX', {}).get(date_str, np.nan)
                row['wind_speed_2m_min'] = params.get('WS2M_MIN', {}).get(date_str, np.nan)
                row['wind_speed_10m_max'] = params.get('WS10M_MAX', {}).get(date_str, np.nan)
                row['wind_speed_10m_min'] = params.get('WS10M_MIN', {}).get(date_str, np.nan)
                row['wind_speed_50m_max'] = params.get('WS50M_MAX', {}).get(date_str, np.nan)
                row['wind_speed_50m_min'] = params.get('WS50M_MIN', {}).get(date_str, np.nan)
                
                row['wind_direction_10m'] = params.get('WD10M', {}).get(date_str, np.nan)
                row['wind_direction_50m'] = params.get('WD50M', {}).get(date_str, np.nan)
                
                row['temperature_2m'] = params.get('T2M', {}).get(date_str, np.nan)
                row['surface_pressure'] = params.get('PS', {}).get(date_str, np.nan)  # Keep in kPa for YOUR calculation
                row['relative_humidity_2m'] = params.get('RH2M', {}).get(date_str, np.nan)
                row['precipitation'] = params.get('PRECTOTCORR', {}).get(date_str, np.nan)
                
                # Calculate air density using YOUR EXACT METHOD
                temp_c = row['temperature_2m'] if not pd.isna(row['temperature_2m']) else 15.0
                pressure_kpa = row['surface_pressure'] if not pd.isna(row['surface_pressure']) else 101.325
                humidity = row['relative_humidity_2m'] if not pd.isna(row['relative_humidity_2m']) else 50
                
                row['air_density'] = self.calculate_air_density(temp_c, pressure_kpa, humidity)
                
                # Calculate wind power densities using YOUR METHOD
                ws_10m = row['wind_speed_10m'] if not pd.isna(row['wind_speed_10m']) else 0
                ws_50m = row['wind_speed_50m'] if not pd.isna(row['wind_speed_50m']) else 0
                
                row['wind_power_density_10m'] = self.calculate_wind_power_density(ws_10m, row['air_density'])
                row['wind_power_density_50m'] = self.calculate_wind_power_density(ws_50m, row['air_density'])
                
                # Wind resource classification using YOUR SYSTEM
                row['wind_resource_class_10m'] = self.classify_wind_resource(ws_10m, row['wind_power_density_10m'])
                row['wind_resource_class_50m'] = self.classify_wind_resource(ws_50m, row['wind_power_density_50m'])
                
                # Turbine operational status using YOUR SYSTEM
                row['turbine_operational_10m'] = self.get_turbine_operational_status(ws_10m)
                row['turbine_operational_50m'] = self.get_turbine_operational_status(ws_50m)
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['season'] = df['month'].apply(
                lambda m: 'Winter' if m in [12, 1, 2] else
                         'Spring' if m in [3, 4, 5] else
                         'Summer' if m in [6, 7, 8] else 'Autumn'
            )
            
            # Reorder columns to match your exact specification
            column_order = [
                'date', 'wind_speed_10m', 'wind_speed_50m', 'wind_speed_2m',
                'wind_speed_10m_max', 'wind_speed_10m_min', 'wind_speed_50m_max', 'wind_speed_50m_min',
                'wind_speed_2m_max', 'wind_speed_2m_min', 'wind_direction_10m', 'wind_direction_50m',
                'temperature_2m', 'surface_pressure', 'relative_humidity_2m', 'precipitation',
                'location', 'latitude', 'longitude', 'air_density',
                'wind_power_density_10m', 'wind_resource_class_10m', 'turbine_operational_10m',
                'wind_power_density_50m', 'wind_resource_class_50m', 'turbine_operational_50m',
                'month', 'season', 'year'
            ]
            
            df = df[column_order]
            
            logger.info(f"Processed {len(df)} records for {raw_data['city']} with YOUR calculation methods")
            return df
            
        except Exception as e:
            logger.error(f"Error processing data for {raw_data['city']}: {str(e)}")
            return pd.DataFrame()
    
    async def extract_multiple_cities(self, cities_data: List[Dict], 
                                    start_date: str = "20140101", 
                                    end_date: str = "20241231") -> pd.DataFrame:
        """Extract data using async speed with YOUR calculation accuracy"""
        
        await self.create_session()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        try:
            logger.info(f"🚀 Starting ALIGNED extraction for {len(cities_data)} cities...")
            logger.info(f"📊 Using YOUR calculation methods with async speed")
            start_time = time.time()
            
            tasks = [
                self.fetch_city_complete(city_info, start_date, end_date, semaphore)
                for city_info in cities_data
            ]
            
            results = []
            completed = 0
            
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                completed += 1
                
                if completed % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(cities_data) - completed) / rate
                    logger.info(f"Progress: {completed}/{len(cities_data)} cities "
                              f"({rate:.1f} cities/sec, ETA: {eta/60:.1f}min)")
            
            all_dfs = []
            successful_cities = 0
            
            for raw_data in results:
                if raw_data.get('success'):
                    df = self.process_city_data(raw_data)
                    if not df.empty:
                        all_dfs.append(df)
                        successful_cities += 1
                else:
                    logger.warning(f"Failed: {raw_data.get('city', 'unknown')}")
            
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                total_time = time.time() - start_time
                
                logger.info(f"✅ SUCCESS! {len(combined_df)} records from {successful_cities} cities")
                logger.info(f"⏱️ Time: {total_time/60:.1f} minutes ({total_time/len(cities_data):.1f}s/city)")
                logger.info(f"🧮 Using YOUR exact calculation methods!")
                
                return combined_df
            else:
                logger.error("❌ No data extracted successfully")
                return pd.DataFrame()
                
        finally:
            await self.close_session()
    
    def save_data(self, df: pd.DataFrame, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"aligned_wind_energy_data_{timestamp}.csv"
        
        os.makedirs('data/processed', exist_ok=True)
        filepath = f"data/processed/{filename}"
        
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Data saved to {filepath}")
        return filepath

# Supporting functions (same as before)
def load_cities_from_csv(csv_path: str, limit: int = None) -> List[Dict]:
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    
    cities = []
    for _, row in df.iterrows():
        cities.append({
            'city': str(row['city']).strip(),
            'latitude': float(row['lat']),
            'longitude': float(row['lng'])
        })
    return cities

def create_city_batches(cities_list: List[Dict], batch_size: int = 100) -> List[List[Dict]]:
    batches = []
    for i in range(0, len(cities_list), batch_size):
        batches.append(cities_list[i:i + batch_size])
    return batches

async def main_aligned_extraction(cities_csv_path: str, batch_size: int = 150, max_concurrent: int = 30):
    """Main extraction pipeline using YOUR calculation methods"""
    logger.info("🔄 ALIGNED Wind Energy Data Extraction")
    logger.info("📊 Using YOUR calculation methods with async speed")
    
    all_cities = load_cities_from_csv(cities_csv_path)
    logger.info(f"📍 Loaded {len(all_cities)} cities")
    
    extractor = AlignedWindEnergyExtractor(max_concurrent=max_concurrent)
    
    city_batches = create_city_batches(all_cities, batch_size)
    logger.info(f"🔄 Processing in {len(city_batches)} batches")
    
    all_results = []
    
    for i, batch in enumerate(city_batches):
        logger.info(f"📦 Batch {i+1}/{len(city_batches)} ({len(batch)} cities)")
        
        try:
            batch_df = await extractor.extract_multiple_cities(batch)
            
            if not batch_df.empty:
                batch_filename = f"aligned_wind_batch_{i+1:03d}.csv"
                extractor.save_data(batch_df, batch_filename)
                all_results.append(batch_df)
                
                # Sample statistics to verify calculations
                logger.info(f"✅ Batch {i+1}: {len(batch_df)} records")
                logger.info(f"   📊 Avg air density: {batch_df['air_density'].mean():.3f} kg/m³")
                logger.info(f"   🌬️ Avg wind 10m: {batch_df['wind_speed_10m'].mean():.2f} m/s")
                logger.info(f"   ⚡ Avg power density: {batch_df['wind_power_density_10m'].mean():.1f} W/m²")
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Batch {i+1} failed: {str(e)}")
            continue
    
    if all_results:
        logger.info("🔗 Combining all batches...")
        final_df = pd.concat(all_results, ignore_index=True)
        
        final_path = extractor.save_data(final_df, "aligned_wind_energy_complete_29_params.csv")
        
        logger.info(f"🎉 ALIGNED EXTRACTION COMPLETE!")
        logger.info(f"📊 Dataset: {len(final_df):,} records from {final_df['location'].nunique()} cities")
        logger.info(f"🧮 Calculations match YOUR synchronous code!")
        logger.info(f"💾 Saved to: {final_path}")
        
        return final_path
    else:
        logger.error("❌ EXTRACTION FAILED")
        return None

# Example usage
if __name__ == "__main__":
    # Optimized configuration for 16GB RAM, Ryzen 5 H-series
    CITIES_CSV = r"C:\Users\Anshuman Singh\Desktop\q.csv"
    BATCH_SIZE = 150         # Adjusted for your system
    MAX_CONCURRENT = 30      # Conservative for stability
    
    result = asyncio.run(main_aligned_extraction(CITIES_CSV, BATCH_SIZE, MAX_CONCURRENT))
    
    if result:
        print(f"\n✅ SUCCESS! Aligned dataset ready!")
        print(f"📂 File: {result}")
        print(f"🔄 Calculations now match your synchronous code!")
    else:
        print("\n❌ EXTRACTION FAILED!")