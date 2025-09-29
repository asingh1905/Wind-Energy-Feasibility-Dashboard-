# Wind Energy Feasibility Dashboard

A Streamlit-based web application for interactive analysis and forecasting of wind energy potential across multiple locations. The dashboard integrates historical wind data, statistical summaries, visualizations, and SARIMA-based forecasting models to support decision-making in renewable energy planning.

## Features

- **Data Loading & Preprocessing**  
  - Reads CSV data of selected cities with daily wind observations  
  - Standardizes column names and parses dates in `DD-MM-YYYY` format  
  - Handles missing values and generates fallback date ranges if needed

- **Exploratory Analysis**  
  - Time series plots of wind speed and power density  
  - Seasonal and monthly summaries of wind metrics  
  - Statistical tables for mean, median, standard deviation, and percentiles  
  - Wind resource classification and turbine operational status charts  
  - Interactive wind rose plots showing directional frequency and average speed

- **Forecasting Module**  
  - Synthetic SARIMA forecast generator for wind speed at selected heights  
  - Configurable forecast horizon (7–90 days) with reproducible random seed  
  - 95 % confidence intervals with dynamic width growth over horizon  
  - Visualization combining historical context with forecast and uncertainty band  
  - Performance metrics (AIC, MAE, MAPE, RMSE) displayed alongside charts

- **Comparison & Export**  
  - Multi-city comparison of wind speed or power density via line and box plots  
  - Weekly forecast breakdown tables with mean, min, max, and interval statistics  
  - Data export in CSV, Excel, or JSON formats  

## Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/asingh1905/Wind-Energy-Feasibility-Dashboard-.git
   cd Wind-Energy-Feasibility-Dashboard-
   ```

2. Create and activate a Python virtual environment  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare data  
   - Place your wind data CSV file at `data/selected_cities/selected_cities_310825.csv`  
   - Ensure it contains at minimum:  
     - `location` (city name)  
     - `date` in `DD-MM-YYYY` format  
     - Wind metrics: `wind_speed_2m`, `wind_speed_10m`, `wind_speed_50m`, and corresponding power densities

## Usage

Run the Streamlit app:  
```bash
streamlit run app.py
```

### Dashboard Sections

1. **Forecasting Dashboard**  
   - Configure location, parameter, forecast days, and display options  
   - Generate synthetic SARIMA forecasts with uncertainty bounds  
   - View performance metrics and export forecast data

2. **Analytics Dashboard**  
   - Choose single-city or multi-city mode  
   - Filter by year, date range, month, or season  
   - Interactive tabs for time series, resource classification, statistics, wind rose, and raw data  
   - Export filtered data in desired format

## Project Structure

```
├── app.py                      # Entry point for forecasting dashboard
├── analytics.py                # Streamlit script for analytics dashboard
├── utils/
│   ├── data_loader.py          # Data loading and preprocessing functions
│   └── model_utils.py          # Forecast generation and plotting helpers
├── data/
│   └── selected_cities/        # CSV files for city wind data
├── models/                     # Pre-trained SARIMA model pickles (optional)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Configuration

- **Caching**  
  Data loading and metadata extraction use `st.cache_data` with a 1 hour TTL to optimize performance.

- **Date Parsing**  
  - Explicit `format="%d-%m-%Y"` with `dayfirst=True` ensures consistent date conversion  
  - Invalid or missing dates are dropped after parsing

- **Styling**  
  Custom CSS in Markdown blocks defines dark-mode compatible cards, headers, and metrics panels.

## Contribution

Contributions are welcome. To contribute:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m "Add feature"`)  
4. Push to your branch (`git push origin feature-name`)  
5. Open a pull request describing your changes

## License

This project is released under the MIT License. See `LICENSE` for details.
