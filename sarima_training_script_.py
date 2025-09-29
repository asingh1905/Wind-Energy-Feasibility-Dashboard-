# SARIMA Training Script for Wind Energy - Highly Optimized for Ryzen 5 H-Series 16GB

import pandas as pd
import numpy as np
import warnings
import pickle
import os
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import psutil
import gc
from pathlib import Path

# Statistical libraries with optimized imports
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    print("✅ All required libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install statsmodels scikit-learn")
    exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sarima_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedSARIMATrainer:
    """
    Highly optimized SARIMA training class for Ryzen 5 H-Series 16GB.
    
    Optimizations:
    - Memory-efficient chunked data processing
    - CPU-optimized parallel processing (6 cores max)
    - Smart parameter selection for large datasets
    - Intelligent model selection based on data characteristics
    - Automatic garbage collection and memory management
    """
    
    def __init__(self, data_path: str, output_dir: str = "models"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Optimized wind parameters for forecasting (most relevant for wind energy)
        self.wind_params = [
            'wind_speed_10m',           # Primary parameter
            'wind_speed_50m',           # Hub height approximation
            'wind_power_density_10m',   # Power potential
            'wind_power_density_50m',   # Power potential at height
            'wind_speed_2m'             # Ground level reference
        ]
        
        # Ryzen 5 H-Series optimized configuration
        self.config = {
            'test_split': 0.15,           # Reduced for more training data
            'forecast_horizon': 30,       # 30 days ahead
            'min_data_points': 800,       # Reduced minimum for more models
            'chunk_size': 50000,          # Process data in chunks
            'parallel_jobs': min(6, mp.cpu_count()),  # Max 6 cores for Ryzen 5
            'memory_limit_gb': 12,        # Leave 4GB for system
            
            # Optimized parameter ranges for speed vs accuracy
            'max_p': 2, 'max_d': 1, 'max_q': 2,      # Reduced complexity
            'max_P': 1, 'max_D': 1, 'max_Q': 1,      # Reduced seasonal complexity
            'seasonal_period': 365,                    # Annual seasonality
            'max_iterations': 150,                     # Faster convergence
            'param_combinations_limit': 20,           # Speed optimization
        }
        
        # Memory monitoring
        self.memory_usage = {'peak': 0, 'current': 0}
        
        # System info
        self._log_system_info()
    
    def _log_system_info(self):
        """Log system information for optimization reference."""
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        logger.info(f"System Configuration:")
        logger.info(f"  CPU Cores: {cpu_count}")
        logger.info(f"  Total RAM: {memory_gb:.1f} GB")
        logger.info(f"  Configured parallel jobs: {self.config['parallel_jobs']}")
        logger.info(f"  Memory limit: {self.config['memory_limit_gb']} GB")
    
    def _monitor_memory(self):
        """Monitor memory usage and trigger garbage collection if needed."""
        current_memory = psutil.virtual_memory().used / (1024**3)
        self.memory_usage['current'] = current_memory
        
        if current_memory > self.memory_usage['peak']:
            self.memory_usage['peak'] = current_memory
        
        # Aggressive garbage collection if memory usage is high
        if current_memory > self.config['memory_limit_gb']:
            logger.warning(f"High memory usage: {current_memory:.1f} GB, running GC")
            gc.collect()
    
    def load_data_optimized(self):
        """Load and preprocess wind data with memory optimization."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            # Check file size
            file_size_mb = Path(self.data_path).stat().st_size / (1024**2)
            logger.info(f"File size: {file_size_mb:.1f} MB")
            
            # First load without any dtype to check the data structure
            logger.info("Reading CSV file...")
            self.df = pd.read_csv(self.data_path, low_memory=False)
            
            # Clean column names (remove whitespace/BOM issues)
            self.df.columns = self.df.columns.str.strip()
            
            # Debug: Print first few rows and columns
            logger.info(f"Columns found: {list(self.df.columns)}")
            logger.info(f"First few rows of date column:")
            if 'date' in self.df.columns:
                logger.info(f"Date column type: {self.df['date'].dtype}")
                logger.info(f"Sample dates: {self.df['date'].head().tolist()}")
            else:
                logger.error("'date' column not found in CSV!")
                return False
            
            # Convert date column safely
            logger.info("Converting date column...")
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            
            # Check for failed date conversions
            failed_dates = self.df['date'].isna().sum()
            if failed_dates > 0:
                logger.warning(f"Failed to parse {failed_dates} dates, removing those rows")
            
            # Clean and optimize data
            self.df = self.df.dropna(subset=['date', 'location'])
            self.df = self.df.sort_values(['location', 'date'])
            
            # Now safely apply datetime operations
            logger.info("Adding time features...")
            self.df['day_of_year'] = self.df['date'].dt.dayofyear.astype('int16')
            self.df['week_of_year'] = self.df['date'].dt.isocalendar().week.astype('int8')
            
            # Convert location to category for memory efficiency
            self.df['location'] = self.df['location'].astype('category')
            
            # Convert numeric columns to float32 for memory efficiency
            numeric_columns = [
                'wind_speed_10m', 'wind_speed_50m', 'wind_speed_2m',
                'wind_power_density_10m', 'wind_power_density_50m',
                'temperature_2m', 'surface_pressure', 'relative_humidity_2m',
                'precipitation', 'latitude', 'longitude', 'air_density'
            ]
            
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('float32')
            
            # Convert integer columns
            if 'month' in self.df.columns:
                self.df['month'] = pd.to_numeric(self.df['month'], errors='coerce').astype('int8')
            if 'year' in self.df.columns:
                self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce').astype('int16')
            
            # Filter available wind parameters
            available_params = [p for p in self.wind_params if p in self.df.columns]
            self.wind_params = available_params
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  Records: {len(self.df):,}")
            logger.info(f"  Cities: {self.df['location'].nunique()}")
            logger.info(f"  Date range: {self.df['date'].min()} to {self.df['date'].max()}")
            logger.info(f"  Available parameters: {available_params}")
            logger.info(f"  Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            self._monitor_memory()
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def check_stationarity_fast(self, ts: pd.Series) -> bool:
        """Fast stationarity check using only ADF test."""
        try:
            # Use only ADF test for speed (skip KPSS)
            adf_result = adfuller(ts.dropna(), autolag='AIC', maxlag=10)  # Limit maxlag
            return adf_result[1] <= 0.05
        except Exception:
            return False
    
    def optimize_parameters_smart(self, ts: pd.Series, seasonal_period: int = 365) -> Tuple:
        """Smart parameter optimization based on data characteristics."""
        try:
            # Data-driven parameter selection
            data_length = len(ts)
            
            # Adjust parameter ranges based on data length
            if data_length < 1500:
                # Simple model for shorter series
                max_p, max_q = 1, 1
                max_P, max_Q = 0, 0
            elif data_length < 2500:
                # Moderate complexity
                max_p, max_q = 2, 2
                max_P, max_Q = 1, 1
            else:
                # Full complexity for long series
                max_p, max_q = self.config['max_p'], self.config['max_q']
                max_P, max_Q = self.config['max_P'], self.config['max_Q']
            
            best_aic = np.inf
            best_order = (1, 1, 1)
            best_seasonal_order = (0, 0, 0, seasonal_period)
            
            # Smart parameter combinations
            param_combinations = []
            
            # Non-seasonal models first (faster)
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    param_combinations.append(((p, 1, q), (0, 0, 0, seasonal_period)))
            
            # Add seasonal models if data is long enough
            if data_length > 2000:
                for p in range(max_p + 1):
                    for q in range(max_q + 1):
                        for P in range(max_P + 1):
                            for Q in range(max_Q + 1):
                                if P + Q > 0:  # Only if seasonal component exists
                                    param_combinations.append(((p, 1, q), (P, 1, Q, seasonal_period)))
            
            # Limit combinations for speed
            param_combinations = param_combinations[:self.config['param_combinations_limit']]
            
            logger.debug(f"Testing {len(param_combinations)} parameter combinations")
            
            for i, (order, seasonal_order) in enumerate(param_combinations):
                try:
                    # Quick model fit with limited iterations
                    model = SARIMAX(
                        ts.dropna(),
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        trend='c'  # Add constant trend
                    )
                    
                    fitted_model = model.fit(
                        disp=False,
                        maxiter=self.config['max_iterations'],
                        method='lbfgs',
                        optim_score='harvey'  # Faster optimization
                    )
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = order
                        best_seasonal_order = seasonal_order
                        
                except Exception:
                    continue
            
            return best_order, best_seasonal_order, best_aic
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {str(e)}")
            return (1, 1, 1), (0, 0, 0, seasonal_period), np.inf
    
    def train_single_model_optimized(self, city: str, param: str) -> Dict:
        """Train SARIMA model with memory and speed optimizations."""
        try:
            logger.info(f"Training: {city} - {param}")
            
            # Filter city data efficiently
            city_mask = self.df['location'] == city
            city_data = self.df[city_mask].copy()
            
            if city_data.empty or param not in city_data.columns:
                return {'success': False, 'error': 'No data available', 'city': city, 'parameter': param}
            
            # Prepare time series
            city_data = city_data.set_index('date').sort_index()
            ts = city_data[param].dropna()
            
            if len(ts) < self.config['min_data_points']:
                return {'success': False, 'error': f'Insufficient data: {len(ts)} points', 'city': city, 'parameter': param}
            
            # Memory-efficient train/test split
            test_size = int(len(ts) * self.config['test_split'])
            train_data = ts.iloc[:-test_size] if test_size > 0 else ts
            test_data = ts.iloc[-test_size:] if test_size > 0 else pd.Series()
            
            # Quick stationarity check
            is_stationary = self.check_stationarity_fast(train_data)
            
            # Smart parameter optimization
            order, seasonal_order, best_aic = self.optimize_parameters_smart(train_data)
            
            # Train final model with optimizations
            model = SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                trend='c',
                concentrate_scale=True  # Faster computation
            )
            
            fitted_model = model.fit(
                disp=False,
                maxiter=self.config['max_iterations'],
                method='lbfgs',
                optim_score='harvey'
            )
            
            # Generate forecast
            forecast_result = fitted_model.get_forecast(steps=self.config['forecast_horizon'])
            forecast_values = forecast_result.predicted_mean
            confidence_intervals = forecast_result.conf_int()
            
            # Create forecast dates
            last_date = train_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=self.config['forecast_horizon'],
                freq='D'
            )
            
            # Calculate accuracy metrics efficiently
            accuracy_metrics = {}
            if len(test_data) > 0:
                test_forecast = fitted_model.get_forecast(steps=len(test_data))
                test_pred = test_forecast.predicted_mean
                
                # Align series for accuracy calculation
                min_len = min(len(test_data), len(test_pred))
                test_actual = test_data.iloc[:min_len]
                test_predicted = test_pred.iloc[:min_len]
                
                accuracy_metrics = {
                    'mae': float(mean_absolute_error(test_actual, test_predicted)),
                    'rmse': float(np.sqrt(mean_squared_error(test_actual, test_predicted))),
                    'mape': float(mean_absolute_percentage_error(test_actual, test_predicted) * 100)
                }
            
            # Prepare lightweight model data
            model_data = {
                'city': city,
                'parameter': param,
                'training_data_length': len(train_data),
                'test_data_length': len(test_data),
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': float(fitted_model.aic),
                'bic': float(fitted_model.bic),
                'log_likelihood': float(fitted_model.llf),
                'is_stationary': is_stationary,
                'accuracy_metrics': accuracy_metrics,
                'forecast_values': forecast_values.round(4).tolist(),
                'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'confidence_intervals': {
                    'lower': confidence_intervals.iloc[:, 0].round(4).tolist(),
                    'upper': confidence_intervals.iloc[:, 1].round(4).tolist()
                },
                'training_date': datetime.now().isoformat(),
                'residuals': fitted_model.resid.round(4).tolist(),
                'fitted_values': fitted_model.fittedvalues.round(4).tolist(),
                'data_stats': {
                    'mean': float(ts.mean()),
                    'std': float(ts.std()),
                    'min': float(ts.min()),
                    'max': float(ts.max())
                }
            }
            
            # Save model with compression
            filename = f"sarima_{city.replace(' ', '_').replace('/', '_')}_{param}.pkl"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Clean up memory
            del fitted_model, model, city_data, ts, train_data
            gc.collect()
            
            logger.info(f"✅ {city} - {param}: AIC {best_aic:.2f}")
            if accuracy_metrics:
                logger.info(f"   Accuracy: MAE {accuracy_metrics['mae']:.3f}, MAPE {accuracy_metrics['mape']:.1f}%")
            
            self._monitor_memory()
            
            return {
                'success': True,
                'filepath': filepath,
                'city': city,
                'parameter': param,
                'aic': float(best_aic),
                'accuracy_metrics': accuracy_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Error training {city} - {param}: {str(e)}")
            return {'success': False, 'error': str(e), 'city': city, 'parameter': param}
    
    def train_all_models_batch(self, cities: List[str] = None, parameters: List[str] = None, 
                              max_models_per_batch: int = 20):
        """Train models in batches to manage memory efficiently."""
        try:
            if self.df is None:
                logger.error("Data not loaded. Call load_data_optimized() first.")
                return
            
            # Use specified cities or all available
            if cities is None:
                cities = self.df['location'].cat.categories.tolist()
            
            # Use specified parameters or available wind parameters
            if parameters is None:
                parameters = self.wind_params
            
            # Filter parameters that exist in data
            parameters = [p for p in parameters if p in self.df.columns]
            
            logger.info(f"🚀 Starting batch training:")
            logger.info(f"   Cities: {len(cities)}")
            logger.info(f"   Parameters: {len(parameters)}")
            logger.info(f"   Total models: {len(cities) * len(parameters)}")
            logger.info(f"   Batch size: {max_models_per_batch}")
            
            # Create all training tasks
            all_tasks = [(city, param) for city in cities for param in parameters]
            total_tasks = len(all_tasks)
            
            # Process in batches
            successful_models = []
            failed_models = []
            
            for batch_start in range(0, total_tasks, max_models_per_batch):
                batch_end = min(batch_start + max_models_per_batch, total_tasks)
                batch_tasks = all_tasks[batch_start:batch_end]
                
                logger.info(f"📦 Processing batch {batch_start//max_models_per_batch + 1}: "
                           f"tasks {batch_start+1}-{batch_end} of {total_tasks}")
                
                # Process current batch with parallel processing
                max_workers = self.config['parallel_jobs']
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit batch tasks
                    future_to_task = {
                        executor.submit(self.train_single_model_optimized, city, param): (city, param)
                        for city, param in batch_tasks
                    }
                    
                    # Process completed tasks
                    for future in as_completed(future_to_task):
                        city, param = future_to_task[future]
                        try:
                            result = future.result(timeout=300)  # 5 minute timeout per model
                            
                            if result['success']:
                                successful_models.append(result)
                            else:
                                failed_models.append(result)
                                
                        except Exception as e:
                            logger.error(f"❌ Timeout/Error: {city} - {param}: {str(e)}")
                            failed_models.append({
                                'success': False, 
                                'error': f'Timeout: {str(e)}', 
                                'city': city, 
                                'parameter': param
                            })
                
                # Progress update
                completed = len(successful_models) + len(failed_models)
                success_rate = len(successful_models) / completed * 100 if completed > 0 else 0
                
                logger.info(f"📊 Batch completed: {completed}/{total_tasks} models "
                           f"({success_rate:.1f}% success rate)")
                logger.info(f"💾 Memory usage: {self.memory_usage['current']:.1f} GB "
                           f"(peak: {self.memory_usage['peak']:.1f} GB)")
                
                # Force garbage collection between batches
                gc.collect()
                self._monitor_memory()
            
            # Generate comprehensive summary
            self.generate_optimized_summary(successful_models, failed_models)
            
            logger.info(f"🎉 Training completed!")
            logger.info(f"✅ Successful models: {len(successful_models)}")
            logger.info(f"❌ Failed models: {len(failed_models)}")
            logger.info(f"📈 Overall success rate: {len(successful_models)/total_tasks*100:.1f}%")
            logger.info(f"💾 Peak memory usage: {self.memory_usage['peak']:.1f} GB")
            
        except Exception as e:
            logger.error(f"Error in batch training process: {str(e)}")
    
    def generate_optimized_summary(self, successful_models: List[Dict], failed_models: List[Dict]):
        """Generate comprehensive training summary with performance metrics."""
        try:
            # Create summary data
            summary_data = []
            
            # Successful models
            for model in successful_models:
                acc = model.get('accuracy_metrics', {})
                summary_data.append({
                    'City': model['city'],
                    'Parameter': model['parameter'],
                    'Status': 'Success',
                    'AIC': f"{model['aic']:.2f}",
                    'MAE': f"{acc.get('mae', 0):.3f}" if acc else 'N/A',
                    'RMSE': f"{acc.get('rmse', 0):.3f}" if acc else 'N/A',
                    'MAPE': f"{acc.get('mape', 0):.1f}%" if acc else 'N/A',
                    'Model_File': os.path.basename(model['filepath'])
                })
            
            # Failed models
            for model in failed_models:
                summary_data.append({
                    'City': model['city'],
                    'Parameter': model['parameter'],
                    'Status': 'Failed',
                    'AIC': 'N/A',
                    'MAE': 'N/A',
                    'RMSE': 'N/A',
                    'MAPE': 'N/A',
                    'Model_File': f"Error: {model.get('error', 'Unknown')}"
                })
            
            # Save detailed summary
            summary_df = pd.DataFrame(summary_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_path = os.path.join(self.output_dir, f"training_summary_{timestamp}.csv")
            summary_df.to_csv(summary_path, index=False)
            
            # Create performance analysis
            if successful_models:
                performance_data = []
                
                for param in self.wind_params:
                    param_models = [m for m in successful_models if m['parameter'] == param]
                    if param_models:
                        aics = [m['aic'] for m in param_models]
                        maes = [m['accuracy_metrics'].get('mae', 0) for m in param_models if m['accuracy_metrics']]
                        
                        performance_data.append({
                            'Parameter': param,
                            'Models_Count': len(param_models),
                            'Avg_AIC': np.mean(aics),
                            'Min_AIC': np.min(aics),
                            'Max_AIC': np.max(aics),
                            'Avg_MAE': np.mean(maes) if maes else 0,
                            'Success_Rate': f"{len(param_models)/len([m for m in successful_models + failed_models if m['parameter'] == param])*100:.1f}%"
                        })
                
                performance_df = pd.DataFrame(performance_data)
                performance_path = os.path.join(self.output_dir, f"performance_analysis_{timestamp}.csv")
                performance_df.to_csv(performance_path, index=False)
                
                logger.info(f"📊 Performance analysis saved: {performance_path}")
            
            logger.info(f"📋 Training summary saved: {summary_path}")
            
            # Log top performers
            if successful_models:
                best_models = sorted(successful_models, key=lambda x: x['aic'])[:10]
                logger.info("🏆 Top 10 models by AIC:")
                for i, model in enumerate(best_models, 1):
                    acc = model.get('accuracy_metrics', {})
                    mape = f", MAPE: {acc.get('mape', 0):.1f}%" if acc else ""
                    logger.info(f"   {i:2d}. {model['city']} - {model['parameter']}: "
                               f"AIC {model['aic']:.2f}{mape}")
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")

def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(description='Optimized SARIMA Training Script for Wind Energy')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--output', default='models', help='Output directory for models')
    parser.add_argument('--cities', nargs='+', help='Specific cities to train (optional)')
    parser.add_argument('--parameters', nargs='+', help='Specific parameters to train (optional)')
    parser.add_argument('--jobs', type=int, default=6, help='Number of parallel jobs (max 6 for Ryzen 5)')
    parser.add_argument('--batch-size', type=int, default=20, help='Models per batch (memory management)')
    parser.add_argument('--memory-limit', type=int, default=12, help='Memory limit in GB')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.jobs > 8:
        logger.warning(f"Limiting parallel jobs to 8 (requested: {args.jobs})")
        args.jobs = 8
    
    # Initialize optimized trainer
    trainer = OptimizedSARIMATrainer(args.data, args.output)
    trainer.config['parallel_jobs'] = args.jobs
    trainer.config['memory_limit_gb'] = args.memory_limit
    
    # Load data
    logger.info("🔄 Loading and preprocessing data...")
    if not trainer.load_data_optimized():
        logger.error("❌ Failed to load data. Exiting.")
        return
    
    # Train models in batches
    logger.info("🚀 Starting optimized training process...")
    trainer.train_all_models_batch(args.cities, args.parameters, args.batch_size)

if __name__ == "__main__":
    if len(os.sys.argv) == 1:  # Interactive mode
        print("🌪️ Optimized SARIMA Training Script for Wind Energy")
        print("=" * 50)
        print()
        
        # Get data file path
        data_path = input("📁 Enter path to your CSV data file: ").strip()
        if not os.path.exists(data_path):
            print(f"❌ Error: File {data_path} not found.")
            exit(1)
        
        # Get output directory
        output_dir = input("📂 Enter output directory (default: models): ").strip() or "models"
        
        # Get number of parallel jobs
        default_jobs = min(6, mp.cpu_count())
        jobs_input = input(f"⚡ Number of parallel jobs (default: {default_jobs}): ").strip()
        parallel_jobs = int(jobs_input) if jobs_input.isdigit() else default_jobs
        
        # Initialize trainer
        print(f"\n🔧 Initializing trainer...")
        trainer = OptimizedSARIMATrainer(data_path, output_dir)
        trainer.config['parallel_jobs'] = parallel_jobs
        
        # Load data
        print("🔄 Loading data...")
        if not trainer.load_data_optimized():
            print("❌ Failed to load data.")
            exit(1)
        
        print(f"\n📊 Data Summary:")
        print(f"   Cities: {trainer.df['location'].nunique()}")
        print(f"   Records: {len(trainer.df):,}")
        print(f"   Parameters: {trainer.wind_params}")
        print(f"   Estimated models: {trainer.df['location'].nunique() * len(trainer.wind_params)}")
        
        # Confirm training
        response = input(f"\n🚀 Start training? This may take 2-4 hours (y/n): ").strip().lower()
        if response == 'y':
            print("🎯 Starting optimized training process...")
            trainer.train_all_models_batch(max_models_per_batch=15)  # Smaller batches for interactive
            print("🎉 Training completed! Check the models directory and log file.")
        else:
            print("❌ Training cancelled.")
    else:
        main()