"""
Binance Data Collector for BTCUSDT
Free alternative data source with 5-minute intervals
Note: BTCUSDT (not BTCUSD) - closest available pair
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
from typing import Optional

class BinanceCollector:
    """Binance data collector for BTC/USDT spot data"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.symbol = "BTCUSDT"  # Closest to BTCUSD available on Binance
        self.interval = "5m"
        self.data_dir = Path("data/raw/binance")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Binance API endpoints
        self.base_url = "https://api.binance.com"
        self.klines_endpoint = "/api/v3/klines"
        
        # Rate limiting
        self.calls_per_minute = 1200  # Binance limit
        self.last_call_time = 0
        
    def _rate_limit(self):
        """Implement Binance rate limiting"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        # Ensure at least 0.05 seconds between calls (1200 calls/minute)
        if time_since_last_call < 0.05:
            sleep_time = 0.05 - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def collect_historical_data(self) -> Optional[pd.DataFrame]:
        """Collect 3 years of historical BTC/USDT data"""
        
        try:
            self.logger.info(f"Collecting {self.symbol} data from Binance")
            
            # Binance limits: 1000 candles per request
            # We need to collect in chunks working backwards
            all_data = []
            end_time = int(datetime.now().timestamp() * 1000)  # Current time in milliseconds
            
            # Collect 3 years of data (approx. 315,000 5-minute candles)
            # Need about 315 requests (1000 candles each)
            for i in range(315):
                self._rate_limit()
                
                # Calculate start time for this chunk
                start_time = end_time - (1000 * 5 * 60 * 1000)  # 1000 candles * 5 min * 60 sec * 1000 ms
                
                params = {
                    'symbol': self.symbol,
                    'interval': self.interval,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1000
                }
                
                url = self.base_url + self.klines_endpoint
                
                self.logger.info(f"Requesting data from Binance: {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code != 200:
                    self.logger.error(f"HTTP {response.status_code}: {response.text}")
                    continue
                
                data = response.json()
                
                # Check for API errors
                if isinstance(data, dict) and 'code' in data:
                    self.logger.error(f"API Error: {data['msg']}")
                    continue
                
                # Convert klines to DataFrame
                if data:
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 
                        'volume', 'close_time', 'quote_asset_volume',
                        'number_of_trades', 'taker_buy_base_asset_volume',
                        'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Keep only needed columns
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    # Convert to numeric
                    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
                    
                    if not df.empty:
                        all_data.append(df)
                        self.logger.info(f"Collected {len(df)} records from Binance")
                
                # Update end_time for next iteration
                end_time = start_time
                
                # Stop if we've gone back 3 years
                three_years_ago = int((datetime.now() - timedelta(days=365 * 3)).timestamp() * 1000)
                if start_time < three_years_ago:
                    break
            
            if not all_data:
                self.logger.error("No data collected from Binance")
                return None
            
            # Combine all data
            df = pd.concat(all_data)
            df = df.sort_values('timestamp')
            
            # Add source column
            df['source'] = 'binance'
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Save raw data
            output_file = self.data_dir / f"BTCUSD_5min_raw.csv"
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Collected {len(df)} candles from Binance")
            self.logger.info(f"Data saved to: {output_file}")
            self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting Binance data: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test Binance connection and data availability"""
        
        try:
            self._rate_limit()
            
            params = {
                'symbol': self.symbol,
                'interval': self.interval,
                'limit': 10
            }
            
            url = self.base_url + self.klines_endpoint
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                self.logger.error(f"HTTP {response.status_code}: {response.text}")
                return False
            
            data = response.json()
            
            if isinstance(data, dict) and 'code' in data:
                self.logger.error(f"API Error: {data['msg']}")
                return False
            
            if data:
                latest_candle = data[-1]
                timestamp = pd.to_datetime(latest_candle[0], unit='ms')
                close_price = float(latest_candle[4])
                
                self.logger.info(f"Binance connection test successful")
                self.logger.info(f"Latest {self.symbol} price: {close_price} at {timestamp}")
                
                return True
            
            self.logger.error("No data returned from Binance")
            return False
            
        except Exception as e:
            self.logger.error(f"Binance connection test failed: {e}")
            return False
