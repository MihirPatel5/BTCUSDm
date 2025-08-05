"""
Yahoo Finance Data Collector for BTCUSD
Free alternative data source with 5-minute intervals
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional

class YFinanceCollector:
    """Yahoo Finance data collector for BTC-USD"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.symbol = "BTC-USD"  # Yahoo Finance symbol for Bitcoin
        self.interval = "5m"
        self.data_dir = Path("data/raw/yfinance")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_historical_data(self, period: str = "3y") -> Optional[pd.DataFrame]:
        """Collect historical BTC-USD data for specified period"""
        
        try:
            self.logger.info(f"Collecting {self.symbol} data from Yahoo Finance")
            
            # Create ticker object
            ticker = yf.Ticker(self.symbol)
            
            # Yahoo Finance limits: max 60 days for 5m interval
            # So we need to collect in chunks
            all_data = []
            end_date = datetime.now()
            
            # Collect data in 60-day chunks
            for i in range(0, 365 * 3, 60):  # 3 years in 60-day chunks
                chunk_end = end_date - timedelta(days=i)
                chunk_start = chunk_end - timedelta(days=60)
                
                if chunk_start < end_date - timedelta(days=365 * 3):
                    chunk_start = end_date - timedelta(days=365 * 3)
                
                self.logger.info(f"Collecting chunk: {chunk_start} to {chunk_end}")
                
                try:
                    data = ticker.history(
                        start=chunk_start,
                        end=chunk_end,
                        interval=self.interval
                    )
                    
                    if not data.empty:
                        all_data.append(data)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to collect chunk {chunk_start}-{chunk_end}: {e}")
                    continue
                
                if chunk_start <= end_date - timedelta(days=365 * 3):
                    break
            
            if not all_data:
                self.logger.error("No data collected from Yahoo Finance")
                return None
            
            # Combine all chunks
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_index()
            
            # Reset index to get timestamp column
            df.reset_index(inplace=True)
            
            # Rename columns to standard format
            df = df.rename(columns={
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low', 
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add source column
            df['source'] = 'yfinance'
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Save raw data
            output_file = self.data_dir / f"BTCUSD_5min_raw.csv"
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Collected {len(df)} candles from Yahoo Finance")
            self.logger.info(f"Data saved to: {output_file}")
            self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting Yahoo Finance data: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test Yahoo Finance connection and data availability"""
        
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Test with last 10 candles
            data = ticker.history(period="1d", interval=self.interval)
            
            if data.empty:
                self.logger.error(f"No data available for {self.symbol}")
                return False
            
            self.logger.info(f"Yahoo Finance connection test successful for {self.symbol}")
            self.logger.info(f"Latest price: {data['Close'].iloc[-1]:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance connection test failed: {e}")
            return False
