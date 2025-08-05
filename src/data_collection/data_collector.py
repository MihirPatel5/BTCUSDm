"""
Main Data Collector Module
Unified interface for collecting BTCUSD data from multiple sources
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Import individual collectors
from .mt5_collector import MT5Collector
from .yfinance_collector import YFinanceCollector
from .alpha_vantage_collector import AlphaVantageCollector
from .binance_collector import BinanceCollector


class DataCollector:
    """Main data collector that manages multiple data sources"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize collectors
        self.collectors = {
            'mt5': MT5Collector(config),
            'yfinance': YFinanceCollector(config),
            'alpha_vantage': AlphaVantageCollector(config),
            'binance': BinanceCollector(config)
        }
        
        # Get data source preferences from config
        self.primary_source = config.get('data', {}).get('sources', {}).get('primary', 'mt5')
        self.alternative_sources = config.get('data', {}).get('sources', {}).get('alternatives', ['yfinance'])
        
    def collect_historical_data(self, years: int = 3) -> Optional[pd.DataFrame]:
        """Collect historical BTCUSD data"""
        
        try:
            self.logger.info(f"Starting historical data collection for {years} years")
            
            # Try primary source first
            data = self._try_collector(self.primary_source, years)
            
            # If primary fails, try alternatives
            if data is None:
                for source in self.alternative_sources:
                    self.logger.info(f"Trying alternative source: {source}")
                    data = self._try_collector(source, years)
                    if data is not None:
                        break
            
            if data is None:
                self.logger.error("All data sources failed")
                return None
            
            # Validate and clean data
            data = self._validate_data(data)
            
            # Save raw data
            self._save_raw_data(data)
            
            self.logger.info(f"Historical data collection completed: {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error in historical data collection: {e}")
            return None
    
    def collect_latest_data(self, hours: int = 24) -> Optional[pd.DataFrame]:
        """Collect latest BTCUSD data for specified hours"""
        
        try:
            self.logger.info(f"Collecting latest {hours} hours of data")
            
            # Try primary source first
            data = self._try_latest_collector(self.primary_source, hours)
            
            # If primary fails, try alternatives
            if data is None:
                for source in self.alternative_sources:
                    self.logger.info(f"Trying alternative source: {source}")
                    data = self._try_latest_collector(source, hours)
                    if data is not None:
                        break
            
            if data is None:
                self.logger.error("All data sources failed for latest data")
                return None
            
            # Validate data
            data = self._validate_data(data)
            
            self.logger.info(f"Latest data collection completed: {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error in latest data collection: {e}")
            return None
    
    def _try_collector(self, source: str, years: int) -> Optional[pd.DataFrame]:
        """Try to collect data from a specific source"""
        
        try:
            if source not in self.collectors:
                self.logger.warning(f"Unknown data source: {source}")
                return None
            
            collector = self.collectors[source]
            
            if source == 'mt5':
                return collector.collect_historical_data(years=years)
            elif source == 'yfinance':
                return collector.collect_historical_data(period=f"{years}y")
            elif source == 'alpha_vantage':
                return collector.collect_historical_data()
            elif source == 'binance':
                return collector.collect_historical_data(years=years)
            
        except Exception as e:
            self.logger.error(f"Error with {source} collector: {e}")
            return None
    
    def _try_latest_collector(self, source: str, hours: int) -> Optional[pd.DataFrame]:
        """Try to collect latest data from a specific source"""
        
        try:
            if source not in self.collectors:
                self.logger.warning(f"Unknown data source: {source}")
                return None
            
            collector = self.collectors[source]
            
            if source == 'mt5':
                return collector.collect_latest_data(hours=hours)
            elif source == 'yfinance':
                return collector.collect_latest_data(period="1d", interval="5m")
            elif source == 'alpha_vantage':
                return collector.collect_latest_data()
            elif source == 'binance':
                return collector.collect_latest_data(hours=hours)
            
        except Exception as e:
            self.logger.error(f"Error with {source} collector: {e}")
            return None
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the collected data"""
        
        try:
            if data is None or data.empty:
                return data
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Sort by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            initial_count = len(data)
            data = data.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            if len(data) < initial_count:
                self.logger.info(f"Removed {initial_count - len(data)} duplicate records")
            
            # Basic data validation
            data = data.dropna(subset=['open', 'high', 'low', 'close'])
            
            # Ensure OHLC relationships are valid
            data = data[
                (data['high'] >= data['open']) &
                (data['high'] >= data['close']) &
                (data['low'] <= data['open']) &
                (data['low'] <= data['close']) &
                (data['volume'] >= 0)
            ]
            
            self.logger.info(f"Data validation completed: {len(data)} valid records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {e}")
            return data
    
    def _save_raw_data(self, data: pd.DataFrame):
        """Save raw data to file"""
        
        try:
            if data is None or data.empty:
                return
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"BTCUSD_5min_raw_{timestamp}.csv"
            filepath = self.data_dir / filename
            
            # Save data
            data.to_csv(filepath, index=False)
            self.logger.info(f"Raw data saved to: {filepath}")
            
            # Also save as latest
            latest_filepath = self.data_dir / "BTCUSD_5min_latest.csv"
            data.to_csv(latest_filepath, index=False)
            self.logger.info(f"Latest data saved to: {latest_filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving raw data: {e}")
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about available data sources"""
        
        info = {
            'primary_source': self.primary_source,
            'alternative_sources': self.alternative_sources,
            'available_collectors': list(self.collectors.keys()),
            'data_directory': str(self.data_dir)
        }
        
        return info
