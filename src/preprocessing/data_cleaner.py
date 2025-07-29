"""
Data Cleaning and Preprocessing Module
Handles data quality issues, gaps, and outliers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Optional

class DataCleaner:
    """Data cleaner for forex market data"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data")
        
    def process_all_sources(self) -> Optional[pd.DataFrame]:
        """Process and combine data from all sources"""
        
        try:
            # Load data from all sources
            data_sources = []
            
            # MT5 data
            mt5_file = self.data_dir / "raw/mt5/BTCUSD_5min_raw.csv"
            if mt5_file.exists():
                mt5_data = pd.read_csv(mt5_file)
                mt5_data['timestamp'] = pd.to_datetime(mt5_data['timestamp'])
                data_sources.append(('mt5', mt5_data))
                self.logger.info(f"Loaded MT5 data: {len(mt5_data)} records")
            
            # Yahoo Finance data
            yf_file = self.data_dir / "raw/yfinance/BTCUSD_5min_raw.csv"
            if yf_file.exists():
                yf_data = pd.read_csv(yf_file)
                yf_data['timestamp'] = pd.to_datetime(yf_data['timestamp'])
                data_sources.append(('yfinance', yf_data))
                self.logger.info(f"Loaded Yahoo Finance data: {len(yf_data)} records")
            
            # Alpha Vantage data
            av_file = self.data_dir / "raw/alpha_vantage/BTCUSD_5min_raw.csv"
            if av_file.exists():
                av_data = pd.read_csv(av_file)
                av_data['timestamp'] = pd.to_datetime(av_data['timestamp'])
                data_sources.append(('alpha_vantage', av_data))
                self.logger.info(f"Loaded Alpha Vantage data: {len(av_data)} records")
            
            # Binance data
            binance_file = self.data_dir / "raw/binance/BTCUSD_5min_raw.csv"
            if binance_file.exists():
                binance_data = pd.read_csv(binance_file)
                binance_data['timestamp'] = pd.to_datetime(binance_data['timestamp'])
                data_sources.append(('binance', binance_data))
                self.logger.info(f"Loaded Binance data: {len(binance_data)} records")
            
            if not data_sources:
                self.logger.error("No data sources found")
                return None
            
            # Clean each source
            cleaned_sources = []
            for source_name, data in data_sources:
                self.logger.info(f"Cleaning {source_name} data...")
                cleaned_data = self.clean_single_source(data, source_name)
                if cleaned_data is not None:
                    cleaned_sources.append((source_name, cleaned_data))
            
            if not cleaned_sources:
                self.logger.error("No data sources could be cleaned")
                return None
            
            # Combine sources
            combined_data = self.combine_sources(cleaned_sources)
            
            if combined_data is not None:
                # Final processing
                final_data = self.final_processing(combined_data)
                
                # Save processed data
                processed_dir = self.data_dir / "processed"
                processed_dir.mkdir(parents=True, exist_ok=True)
                output_file = processed_dir / "BTCUSD_5min_processed.csv"
                final_data.to_csv(output_file, index=False)
                
                self.logger.info(f"Final processed data saved: {len(final_data)} records")
                self.logger.info(f"Date range: {final_data['timestamp'].min()} to {final_data['timestamp'].max()}")
                
                return final_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing all sources: {e}")
            return None
    
    def clean_single_source(self, data: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
        """Clean data from a single source"""
        
        try:
            original_count = len(data)
            self.logger.info(f"Cleaning {source_name}: {original_count} records")
            
            # Remove duplicates
            data = data.drop_duplicates(subset=['timestamp'])
            if len(data) < original_count:
                self.logger.info(f"Removed {original_count - len(data)} duplicate records")
            
            # Sort by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Remove weekends and holidays
            data = self.remove_weekends_holidays(data)
            
            # Handle missing data
            data = self.handle_missing_data(data)
            
            # Detect and handle outliers
            data = self.handle_outliers(data)
            
            # Ensure data consistency
            data = self.ensure_consistency(data)
            
            # Validate data quality
            if not self.validate_data_quality(data):
                self.logger.warning(f"Data quality issues in {source_name}")
            
            self.logger.info(f"Cleaned {source_name}: {len(data)} records remaining")
            return data
            
        except Exception as e:
            self.logger.error(f"Error cleaning {source_name} data: {e}")
            return None
    
    def remove_weekends_holidays(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove weekend and holiday data gaps"""
        
        original_count = len(data)
        
        # Remove weekends (Saturday=5, Sunday=6)
        data = data[data['timestamp'].dt.weekday < 5]
        
        if len(data) < original_count:
            self.logger.info(f"Removed {original_count - len(data)} weekend records")
        
        # Note: Holiday removal would require a holiday calendar
        # For now, we'll just remove weekends
        
        return data
    
    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data with forward fill for small gaps"""
        
        # Check for missing values
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            self.logger.info(f"Found {missing_count} missing values")
        
        # Forward fill for small gaps (configurable)
        max_gap_minutes = self.config.get('data', {}).get('max_gap_minutes', 30)
        max_gap_periods = max_gap_minutes // 5  # 5-minute intervals
        
        # Forward fill OHLCV columns
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill', limit=max_gap_periods)
        
        # Remove rows that still have missing values
        data = data.dropna()
        
        return data
    
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and remove outliers beyond specified standard deviations"""
        
        threshold = self.config.get('data', {}).get('outlier_threshold', 4)
        
        # Calculate price changes
        data['price_change'] = data['close'].pct_change()
        
        # Identify outliers based on price change
        outliers = abs(data['price_change']) > (threshold * data['price_change'].std())
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            self.logger.info(f"Found {outlier_count} outlier records (> {threshold} std dev)")
            data = data[~outliers]
        
        # Remove the temporary column
        if 'price_change' in data.columns:
            data = data.drop(columns=['price_change'])
        
        return data
    
    def ensure_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data consistency and timezone alignment"""
        
        # Ensure timestamp is timezone-naive or UTC
        if data['timestamp'].dt.tz is not None:
            data['timestamp'] = data['timestamp'].dt.tz_localize(None)
        
        # Ensure OHLCV columns are numeric
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Ensure high >= low
        invalid_hl = data['high'] < data['low']
        if invalid_hl.any():
            self.logger.warning(f"Found {invalid_hl.sum()} records with high < low")
            # Fix by swapping
            data.loc[invalid_hl, ['high', 'low']] = data.loc[invalid_hl, ['low', 'high']].values
        
        # Ensure high >= close >= low
        invalid_hc = data['high'] < data['close']
        invalid_cl = data['close'] < data['low']
        
        if invalid_hc.any():
            data.loc[invalid_hc, 'high'] = data.loc[invalid_hc, 'close']
        
        if invalid_cl.any():
            data.loc[invalid_cl, 'low'] = data.loc[invalid_cl, 'close']
        
        return data
    
    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """Validate overall data quality"""
        
        if data.empty:
            self.logger.error("Data is empty")
            return False
        
        # Check for required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for missing values
        if data[required_columns].isnull().any().any():
            self.logger.warning("Data contains missing values")
        
        # Check for negative prices or volume
        if (data['open'] < 0).any() or (data['high'] < 0).any() or \
           (data['low'] < 0).any() or (data['close'] < 0).any():
            self.logger.error("Data contains negative prices")
            return False
        
        if (data['volume'] < 0).any():
            self.logger.error("Data contains negative volume")
            return False
        
        return True
    
    def combine_sources(self, sources: List[tuple]) -> Optional[pd.DataFrame]:
        """Combine data from multiple sources, prioritizing primary source"""
        
        if not sources:
            return None
        
        # Determine primary source
        primary_source = self.config.get('data', {}).get('sources', {}).get('primary', 'mt5')
        
        # Start with primary source
        primary_data = None
        other_sources = []
        
        for source_name, data in sources:
            if source_name == primary_source:
                primary_data = data
            else:
                other_sources.append((source_name, data))
        
        if primary_data is None:
            self.logger.warning(f"Primary source {primary_source} not found, using first available source")
            primary_data = sources[0][1]
            other_sources = sources[1:]
        
        # For now, we'll just return the primary source
        # In a more advanced implementation, we could combine sources
        self.logger.info(f"Using {primary_source} as primary data source")
        return primary_data
    
    def final_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final processing steps before saving"""
        
        # Ensure data is sorted
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Add any final columns or transformations
        # For example, we might add a "complete" flag for data validation
        data['data_complete'] = True
        
        return data
