"""
Data Preprocessing Module
Implements comprehensive data cleaning, validation, and preprocessing for BTCUSD data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from scipy import stats


class DataPreprocessor:
    """Comprehensive data preprocessing for BTCUSD forex data"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocessing parameters from config
        self.outlier_threshold = config.get('data', {}).get('outlier_threshold', 4)
        self.max_gap_minutes = config.get('data', {}).get('max_gap_minutes', 30)
        
    def preprocess_data(self, input_file: str = None) -> Optional[pd.DataFrame]:
        """Main preprocessing pipeline"""
        
        try:
            self.logger.info("Starting data preprocessing pipeline...")
            
            # Load raw data
            data = self._load_raw_data(input_file)
            if data is None:
                return None
            
            self.logger.info(f"Loaded raw data: {len(data)} records")
            
            # Step 1: Basic data validation and cleaning
            data = self._basic_validation(data)
            if data is None:
                return None
            
            # Step 2: Handle missing values and gaps
            data = self._handle_missing_values(data)
            
            # Step 3: Detect and handle outliers
            data = self._handle_outliers(data)
            
            # Step 4: Data quality checks
            data = self._quality_checks(data)
            
            # Step 5: Final validation
            data = self._final_validation(data)
            
            # Save preprocessed data
            self._save_preprocessed_data(data)
            
            self.logger.info(f"Data preprocessing completed: {len(data)} clean records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {e}")
            return None
    
    def _load_raw_data(self, input_file: str = None) -> Optional[pd.DataFrame]:
        """Load raw data from file"""
        
        try:
            if input_file:
                data_file = Path(input_file)
            else:
                # Look for latest raw data file
                data_file = self.raw_dir / "BTCUSD_5min_latest.csv"
                if not data_file.exists():
                    # Look for any raw data file
                    raw_files = list(self.raw_dir.glob("BTCUSD_*.csv"))
                    if not raw_files:
                        self.logger.error("No raw data files found")
                        return None
                    data_file = max(raw_files, key=lambda x: x.stat().st_mtime)
            
            if not data_file.exists():
                self.logger.error(f"Data file not found: {data_file}")
                return None
            
            # Load data
            data = pd.read_csv(data_file)
            
            # Convert timestamp
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {e}")
            return None
    
    def _basic_validation(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Basic data validation and cleaning"""
        
        try:
            self.logger.info("Performing basic data validation...")
            
            # Check required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Remove rows with invalid OHLC relationships
            initial_count = len(data)
            
            # High should be >= Open, Close, Low
            # Low should be <= Open, Close, High
            valid_ohlc = (
                (data['high'] >= data['open']) &
                (data['high'] >= data['close']) &
                (data['high'] >= data['low']) &
                (data['low'] <= data['open']) &
                (data['low'] <= data['close']) &
                (data['low'] <= data['high']) &
                (data['volume'] >= 0)
            )
            
            data = data[valid_ohlc].copy()
            
            removed_count = initial_count - len(data)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} rows with invalid OHLC relationships")
            
            # Remove duplicate timestamps
            data = data.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            # Sort by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Basic validation completed: {len(data)} valid records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error in basic validation: {e}")
            return None
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and data gaps"""
        
        try:
            self.logger.info("Handling missing values and gaps...")
            
            # Check for missing values in OHLCV columns
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_counts = data[ohlcv_columns].isnull().sum()
            
            if missing_counts.sum() > 0:
                self.logger.info(f"Missing values found: {missing_counts.to_dict()}")
                
                # Forward fill missing values (carry last observation forward)
                data[ohlcv_columns] = data[ohlcv_columns].fillna(method='ffill')
                
                # Backward fill any remaining missing values at the beginning
                data[ohlcv_columns] = data[ohlcv_columns].fillna(method='bfill')
                
                # If still missing, fill with median values
                for col in ohlcv_columns:
                    if data[col].isnull().sum() > 0:
                        median_value = data[col].median()
                        data[col] = data[col].fillna(median_value)
                        self.logger.info(f"Filled remaining missing values in {col} with median: {median_value}")
            
            # Handle time gaps
            data = self._handle_time_gaps(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {e}")
            return data
    
    def _handle_time_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle gaps in time series data"""
        
        try:
            # Calculate time differences
            data['time_diff'] = data['timestamp'].diff()
            
            # Identify large gaps (more than max_gap_minutes)
            large_gaps = data['time_diff'] > pd.Timedelta(minutes=self.max_gap_minutes)
            gap_count = large_gaps.sum()
            
            if gap_count > 0:
                self.logger.info(f"Found {gap_count} large time gaps (>{self.max_gap_minutes} minutes)")
                
                # Option 1: Fill small gaps with interpolated data
                # Option 2: Mark gaps for special handling
                # For now, we'll just log the gaps and continue
                
                gap_locations = data[large_gaps]['timestamp'].tolist()
                for gap_time in gap_locations[:5]:  # Log first 5 gaps
                    self.logger.info(f"Large gap detected at: {gap_time}")
            
            # Remove the temporary time_diff column
            data = data.drop('time_diff', axis=1)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error handling time gaps: {e}")
            return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in price data"""
        
        try:
            self.logger.info("Detecting and handling outliers...")
            
            price_columns = ['open', 'high', 'low', 'close']
            initial_count = len(data)
            
            for column in price_columns:
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(data[column]))
                
                # Identify outliers
                outliers = z_scores > self.outlier_threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    self.logger.info(f"Found {outlier_count} outliers in {column}")
                    
                    # Option 1: Remove outliers (aggressive)
                    # Option 2: Cap outliers (conservative)
                    # We'll use capping to preserve data points
                    
                    # Calculate percentiles for capping
                    lower_percentile = data[column].quantile(0.01)
                    upper_percentile = data[column].quantile(0.99)
                    
                    # Cap outliers
                    data[column] = data[column].clip(lower=lower_percentile, upper=upper_percentile)
            
            # Handle volume outliers separately (can be more extreme)
            volume_z_scores = np.abs(stats.zscore(data['volume']))
            volume_outliers = volume_z_scores > (self.outlier_threshold + 1)  # More lenient for volume
            
            if volume_outliers.sum() > 0:
                self.logger.info(f"Found {volume_outliers.sum()} volume outliers")
                # Cap volume outliers
                volume_upper = data['volume'].quantile(0.995)
                data['volume'] = data['volume'].clip(upper=volume_upper)
            
            self.logger.info(f"Outlier handling completed: {len(data)} records retained")
            return data
            
        except Exception as e:
            self.logger.error(f"Error handling outliers: {e}")
            return data
    
    def _quality_checks(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform additional data quality checks"""
        
        try:
            self.logger.info("Performing data quality checks...")
            
            # Check for reasonable price ranges (BTCUSD should be > $1000 and < $1,000,000)
            price_columns = ['open', 'high', 'low', 'close']
            
            for column in price_columns:
                unreasonable_prices = (data[column] < 1000) | (data[column] > 1000000)
                if unreasonable_prices.sum() > 0:
                    self.logger.warning(f"Found {unreasonable_prices.sum()} unreasonable prices in {column}")
                    # Remove rows with unreasonable prices
                    data = data[~unreasonable_prices]
            
            # Check for zero or negative volumes
            zero_volume = data['volume'] <= 0
            if zero_volume.sum() > 0:
                self.logger.info(f"Found {zero_volume.sum()} zero/negative volume records")
                # Set minimum volume to 1
                data.loc[zero_volume, 'volume'] = 1
            
            # Check data continuity
            time_diffs = data['timestamp'].diff().dt.total_seconds() / 60  # Convert to minutes
            expected_interval = 5  # 5-minute data
            
            irregular_intervals = np.abs(time_diffs - expected_interval) > 1  # Allow 1 minute tolerance
            irregular_count = irregular_intervals.sum()
            
            if irregular_count > 0:
                self.logger.info(f"Found {irregular_count} irregular time intervals")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in quality checks: {e}")
            return data
    
    def _final_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final validation before saving"""
        
        try:
            self.logger.info("Performing final validation...")
            
            # Ensure no missing values in critical columns
            critical_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_critical = data[critical_columns].isnull().sum().sum()
            
            if missing_critical > 0:
                self.logger.error(f"Critical missing values found: {missing_critical}")
                # Remove rows with any missing critical values
                data = data.dropna(subset=critical_columns)
            
            # Final sort and reset index
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Add data quality metrics
            data_quality = {
                'total_records': len(data),
                'date_range': {
                    'start': data['timestamp'].min().isoformat(),
                    'end': data['timestamp'].max().isoformat()
                },
                'price_range': {
                    'min': float(data['close'].min()),
                    'max': float(data['close'].max()),
                    'mean': float(data['close'].mean())
                },
                'volume_stats': {
                    'min': float(data['volume'].min()),
                    'max': float(data['volume'].max()),
                    'mean': float(data['volume'].mean())
                }
            }
            
            self.logger.info(f"Data quality summary: {data_quality}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in final validation: {e}")
            return data
    
    def _save_preprocessed_data(self, data: pd.DataFrame):
        """Save preprocessed data to file"""
        
        try:
            if data is None or data.empty:
                self.logger.error("No data to save")
                return
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"BTCUSD_5min_preprocessed_{timestamp}.csv"
            filepath = self.processed_dir / filename
            
            # Save data
            data.to_csv(filepath, index=False)
            self.logger.info(f"Preprocessed data saved to: {filepath}")
            
            # Also save as latest
            latest_filepath = self.processed_dir / "BTCUSD_5min_preprocessed.csv"
            data.to_csv(latest_filepath, index=False)
            self.logger.info(f"Latest preprocessed data saved to: {latest_filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving preprocessed data: {e}")
    
    def get_preprocessing_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        
        try:
            if data is None or data.empty:
                return {}
            
            stats = {
                'total_records': len(data),
                'date_range': {
                    'start': data['timestamp'].min(),
                    'end': data['timestamp'].max(),
                    'duration_days': (data['timestamp'].max() - data['timestamp'].min()).days
                },
                'price_statistics': {
                    'close_mean': float(data['close'].mean()),
                    'close_std': float(data['close'].std()),
                    'close_min': float(data['close'].min()),
                    'close_max': float(data['close'].max())
                },
                'volume_statistics': {
                    'volume_mean': float(data['volume'].mean()),
                    'volume_std': float(data['volume'].std()),
                    'volume_min': float(data['volume'].min()),
                    'volume_max': float(data['volume'].max())
                },
                'data_quality': {
                    'missing_values': data.isnull().sum().sum(),
                    'duplicate_timestamps': data['timestamp'].duplicated().sum()
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating preprocessing stats: {e}")
            return {}
