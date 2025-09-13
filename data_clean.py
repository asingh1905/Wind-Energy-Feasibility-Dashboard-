import pandas as pd
import os

def convert_cities_data(input_file):

    # Create directories
    os.makedirs('data/raw', exist_ok=True)

    # Load your data
    df = pd.read_csv(input_file)

    # Clean and prepare data
    locations_df = pd.DataFrame({
        'name': df['City'].str.replace(' ', '_').str.replace(',', ''),
        'latitude': pd.to_numeric(df['Latitude'], errors='coerce'),
        'longitude': pd.to_numeric(df['Longitude'], errors='coerce'),
        'state': df['State'],
        'type': 'inland'  # Default, will update based on state
    })

    # Define location types based on states
    coastal_states = [
        'Gujarat', 'Maharashtra', 'Goa', 'Karnātaka', 'Kerala', 'Tamil Nādu',
        'Andhra Pradesh', 'Telangana', 'Odisha', 'West Bengal', 'Tripura'
    ]

    hilly_states = [
        'Himachal Pradesh', 'Uttarakhand', 'Jammu and Kashmir',
        'Arunachal Pradesh', 'Sikkim', 'Meghalaya', 'Mizoram', 'Nagaland', 'Manipur'
    ]

    desert_states = ['Rājasthān', 'Haryana', 'Punjab']

    # Update location types
    for idx, row in locations_df.iterrows():
        state = row['state']
        if state in coastal_states:
            locations_df.at[idx, 'type'] = 'coastal'
        elif state in hilly_states:
            locations_df.at[idx, 'type'] = 'hilly'
        elif state in desert_states:
            locations_df.at[idx, 'type'] = 'desert'

    # Remove invalid coordinates and duplicates
    locations_df = locations_df.dropna(subset=['latitude', 'longitude'])

    # Filter to valid Indian coordinates
    locations_df = locations_df[
        (locations_df['latitude'].between(6, 40)) &
        (locations_df['longitude'].between(68, 98))
        ]

    # Remove duplicates (keep first occurrence)
    locations_df = locations_df.drop_duplicates(subset=['name'], keep='first')

    # Save file
    all_locations_file = 'data/raw/all_cities.csv'
    locations_df.to_csv(all_locations_file, index=False)

def main():
    input_file="data/Indian cities data csv utf-8.csv"
    # Convert data
    convert_cities_data(input_file)
    print(f"\n Conversion completed!")


if __name__ == "__main__":
    main()
