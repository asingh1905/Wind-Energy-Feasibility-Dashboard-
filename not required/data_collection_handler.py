import pandas as pd
import os
import time
from datetime import datetime
from src.data_collection import WindDataCollector
import logging

# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)


def run_production_data_collection():
    """
    Run complete data collection with robust error handling and progress tracking
    """

    print(" WIND ENERGY DASHBOARD - PRODUCTION DATA COLLECTION\n\n")

    # Configuration
    locations_file = 'data/raw/selected_cities.csv'
    START_YEAR = 2014  # 10+ years of data
    END_YEAR = 2024
    BATCH_SIZE = 3  # Process 3 cities at a time to be API-friendly

    # Verify locations file exists
    if not os.path.exists(locations_file):
        print(f"Locations file not found: {locations_file}")
        return

    # Load and display locations
    locations_df = pd.read_csv(locations_file)
    total_cities = len(locations_df)

    print(f" Cities to process: {total_cities}")
    print(f" Data period: {START_YEAR} - {END_YEAR} ({END_YEAR - START_YEAR + 1} years)")
    print(f" Batch size: {BATCH_SIZE} cities at a time")
    print(f"️Estimated time: {total_cities * 2} - {total_cities * 4} minutes")

    print(f"\nCities selected:")
    for i, row in locations_df.iterrows():
        print(f"  {i + 1:2d}. {row['name']:20s} ({row['state']}, {row['type']})")

    # Confirm before starting
    print(f"\n This will make ~{total_cities * (END_YEAR - START_YEAR + 1)} API requests")
    print(" The process will be respectful of API limits with delays")

    proceed = input(f"\nProceed with data collection? (y/N): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print(" Collection cancelled by user")
        return

    # Initialize collector
    collector = WindDataCollector()

    # Track progress
    successful_cities = []
    failed_cities = []
    all_data = []

    start_time = datetime.now()

    print(f"\n Starting data collection at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Process cities in batches
    for batch_start in range(0, total_cities, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_cities)
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (total_cities + BATCH_SIZE - 1) // BATCH_SIZE

        batch_cities = locations_df.iloc[batch_start:batch_end]

        print(f"\n BATCH {batch_num}/{total_batches}")
        print(f" Processing cities {batch_start + 1}-{batch_end}")
        print("-" * 40)

        batch_start_time = datetime.now()

        for idx, city_row in batch_cities.iterrows():
            city_name = city_row['name']
            city_progress = f"[{idx + 1}/{total_cities}]"

            print(f"\n{city_progress} {city_name}")
            print(f"{city_row['latitude']:.3f}, {city_row['longitude']:.3f}")
            print(f"{city_row['state']} ({city_row['type']})")

            try:
                # Collect data for this city
                city_data = collector.collect_for_location(
                    city_row,
                    start_year=START_YEAR,
                    end_year=END_YEAR
                )

                if not city_data.empty:
                    all_data.append(city_data)
                    successful_cities.append(city_name)

                    # Show success stats
                    records_count = len(city_data)
                    date_range = f"{city_data['date'].min().strftime('%Y-%m-%d')} to {city_data['date'].max().strftime('%Y-%m-%d')}"
                    avg_wind_speed = city_data['wind_speed_10m'].mean()

                    print(f"    SUCCESS: {records_count:,} records")
                    print(f"    Date range: {date_range}")
                    print(f"    Avg wind speed: {avg_wind_speed:.2f} m/s")

                else:
                    failed_cities.append(city_name)
                    print(f"    FAILED: No data collected")

            except Exception as e:
                failed_cities.append(city_name)
                print(f"    ERROR: {str(e)[:60]}...")
                logging.error(f"Error collecting data for {city_name}: {e}")

        batch_duration = datetime.now() - batch_start_time

        print(f"\n Batch {batch_num} Summary:")
        print(f"    Duration: {batch_duration}")
        print(f"    Successful: {len([c for c in successful_cities if c in batch_cities['name'].values])}")
        print(f"    Failed: {len([c for c in failed_cities if c in batch_cities['name'].values])}")

        # Save intermediate results
        if all_data:
            intermediate_df = pd.concat(all_data, ignore_index=True)
            intermediate_file = f'data/processed/wind_data_batch_{batch_num}.csv'
            intermediate_df.to_csv(intermediate_file, index=False)
            print(f"    Intermediate data saved: {intermediate_file}")

        # Rest between batches (except for last batch)
        if batch_end < total_cities:
            rest_time = 60  # 1 minute rest between batches
            print(f"    Resting {rest_time} seconds before next batch...")
            time.sleep(rest_time)

    # Final consolidation
    total_duration = datetime.now() - start_time

    print(f"\n FINAL COLLECTION SUMMARY")
    print("=" * 60)
    print(f" Total duration: {total_duration}")
    print(f" Cities processed: {total_cities}")
    print(f" Successful: {len(successful_cities)}")
    print(f" Failed: {len(failed_cities)}")

    if failed_cities:
        print(f"\n Failed cities:")
        for city in failed_cities:
            print(f"   - {city}")

    if all_data:
        # Create final consolidated dataset
        print(f"\n Creating final dataset...")

        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sort_values(['location', 'date']).reset_index(drop=True)

        # Save final dataset
        final_output = 'data/processed/wind_data_all_cities_complete.csv'
        final_df.to_csv(final_output, index=False)

        # Generate comprehensive summary
        generate_final_summary(final_df, successful_cities, failed_cities)

        print(f" DATA COLLECTION COMPLETED!")
        print(f" Final dataset: {final_output}")
        print(f" Total records: {len(final_df):,}")
        print(f" Date range: {final_df['date'].min()} to {final_df['date'].max()}")

        return final_df
    else:
        print(f" No data was collected successfully")
        return None


def generate_final_summary(df, successful_cities, failed_cities):
    """Generate comprehensive data collection summary"""

    summary_stats = {
        'collection_info': {
            'total_records': len(df),
            'successful_cities': len(successful_cities),
            'failed_cities': len(failed_cities),
            'success_rate': (len(successful_cities) / (len(successful_cities) + len(failed_cities))) * 100,
            'collection_date': datetime.now().isoformat()
        },
        'data_coverage': {
            'cities': df['location'].nunique(),
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat(),
                'total_days': (df['date'].max() - df['date'].min()).days
            }
        },
        'data_quality': {
            'completeness': {
                'wind_speed_10m': f"{(df['wind_speed_10m'].notna().sum() / len(df)) * 100:.1f}%",
                'wind_speed_50m': f"{(df['wind_speed_50m'].notna().sum() / len(df)) * 100:.1f}%",
                'temperature': f"{(df['temperature_2m'].notna().sum() / len(df)) * 100:.1f}%"
            }
        },
        'wind_statistics': {
            'avg_wind_speed_10m': f"{df['wind_speed_10m'].mean():.2f} m/s",
            'max_wind_speed_10m': f"{df['wind_speed_10m'].max():.2f} m/s",
            'avg_wind_speed_50m': f"{df['wind_speed_50m'].mean():.2f} m/s",
            'max_wind_speed_50m': f"{df['wind_speed_50m'].max():.2f} m/s"
        },
        'city_list': {
            'successful': successful_cities,
            'failed': failed_cities
        }
    }

    # Save detailed summary
    import json
    summary_file = 'data/processed/collection_summary_detailed.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)

    print(f" Detailed summary saved: {summary_file}")


if __name__ == "__main__":
    run_production_data_collection()
