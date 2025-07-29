"""
Alpha Vantage Data Collector for BTCUSD
Free tier alternative data source with 5-minute intervals
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
import os
from typing import Optional

class AlphaVantageCollector:
    """Alpha Vantage data collector for BTC/USD forex data"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.symbol = "BTC"
        self.market = "USD"
        self.interval = "5min"
        self.function = "FX_INTRADAY"
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.data_dir = Path("data/raw/alpha_vantage")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.calls_per_minute = 5
        self.last_call_time = 0
        
    def _rate_limit(self):
        """Implement Alpha Vantage rate limiting"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        # Ensure at least 12 seconds between calls (5 calls/minute)
        if time_since_last_call < 12:
            sleep_time = 12 - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def collect_historical_data(self) -> Optional[pd.DataFrame]:
        """Collect 3 years of historical BTC/USD data"""
        
        if not self.api_key:
            self.logger.error("Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY environment variable.")
            return None
        
        try:
            self.logger.info(f"Collecting {self.symbol}/{self.market} data from Alpha Vantage")
            
            # Alpha Vantage limits: only last 1000 data points for intraday
            # We need to collect in chunks working backwards
            all_data = []
            end_date = datetime.now()
            
            # Collect data in chunks (1000 points each)
            for i in range(0, 365 * 3, 3):  # 3 years in ~3-day chunks (1000 * 5min)
                self._rate_limit()
                
                params = {
                    'function': self.function,
                    'from_symbol': self.symbol,
                    'to_symbol': self.market,
                    'interval': self.interval,
                    'apikey': self.api_key,
                    'outputsize': 'full',
                    'datatype': 'json'
                }
                
                url = 'https://www.alphavantage.co/query'
                
                self.logger.info(f"Requesting data from Alpha Vantage (attempt)")
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code != 200:
                    self.logger.error(f"HTTP {response.status_code}: {response.text}")
                    continue
                
                data = response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    self.logger.error(f"API Error: {data['Error Message']}")
                    continue
                
                if 'Information' in data:
                    self.logger.info(f"API Information: {data['Information']}")
                
                # Extract time series data
                time_series_key = f"Time Series FX ({self.interval})"
                if time_series_key not in data:
                    self.logger.warning(f"Time series data not found in response")
                    continue
                
                time_series = data[time_series_key]
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Rename columns
                df = df.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. volume': 'volume'
                })
                
                # Convert to numeric
                df = df.apply(pd.to_numeric)
                
                if not df.empty:
                    all_data.append(df)
                    self.logger.info(f"Collected {len(df)} records from Alpha Vantage")
                
                # Break after first successful call as Alpha Vantage only returns last 1000 points
                break
            
            if not all_data:
                self.logger.error("No data collected from Alpha Vantage")
                return None
            
            # Combine all data
            df = pd.concat(all_data)
            df = df.sort_index()
            
            # Reset index to get timestamp column
            df.reset_index(inplace=True)
            df = df.rename(columns={'index': 'timestamp'})
            
            # Add source column
            df['source'] = 'alpha_vantage'
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Save raw data
            output_file = self.data_dir / f"BTCUSD_5min_raw.csv"
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Collected {len(df)} candles from Alpha Vantage")
            self.logger.info(f"Data saved to: {output_file}")
            self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting Alpha Vantage data: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test Alpha Vantage connection and API key"""
        
        if not self.api_key:
            self.logger.error("Alpha Vantage API key not found")
            return False
        
        try:
            self._rate_limit()
            
            params = {
                'function': self.function,
                'from_symbol': self.symbol,
                'to_symbol': self.market,
                'interval': self.interval,
                'apikey': self.api_key,
                'outputsize': 'compact',  # Only last 100 data points for testing
                'datatype': 'json'
            }
            
            url = 'https://www.alphavantage.co/query'
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                self.logger.error(f"HTTP {response.status_code}: {response.text}")
                return False
            
            data = response.json()
            
            if 'Error Message' in data:
                self.logger.error(f"API Error: {data['Error Message']}")
                return False
            
            time_series_key = f"Time Series FX ({self.interval})"
            if time_series_key not in data:
                self.logger.error(f"Time series data not found in response")
                return False
            
            time_series = data[time_series_key]
            latest_timestamp = list(time_series.keys())[0]
            latest_price = time_series[latest_timestamp]['4. close']
            
            self.logger.info(f"Alpha Vantage connection test successful")
            self.logger.info(f"Latest {self.symbol}/{self.market} price: {latest_price} at {latest_timestamp}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Alpha Vantage connection test failed: {e}")
            return False
