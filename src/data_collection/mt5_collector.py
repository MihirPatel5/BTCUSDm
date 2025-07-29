"""
MetaTrader 5 Data Collector for BTCUSD
Primary data source with 5-minute intervals
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional

class MT5Collector:
    """MetaTrader 5 data collector for BTCUSD forex data"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.symbol = config['data']['symbol']
        self.timeframe = mt5.TIMEFRAME_M5  # 5-minute candles
        self.data_dir = Path("data/raw/mt5")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            self.logger.info(f"MT5 initialized successfully")
            self.logger.info(f"MT5 version: {mt5.version()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MT5: {e}")
            return False
    
    def collect_historical_data(self) -> Optional[pd.DataFrame]:
        """Collect 3 years of historical BTCUSD data"""
        
        if not self.initialize_mt5():
            return None
        
        try:
            # Calculate date range (3 years)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 3)
            
            self.logger.info(f"Collecting {self.symbol} data from {start_date} to {end_date}")
            
            # Get historical data
            rates = mt5.copy_rates_range(
                self.symbol,
                self.timeframe,
                start_date,
                end_date
            )
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"No data received for {self.symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to standard format
            df = df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })
            
            # Add source column
            df['source'] = 'mt5'
            
            # Save raw data
            output_file = self.data_dir / f"{self.symbol}_5min_raw.csv"
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Collected {len(df)} candles from MT5")
            self.logger.info(f"Data saved to: {output_file}")
            self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting MT5 data: {e}")
            return None
        
        finally:
            mt5.shutdown()
    
    def get_symbol_info(self) -> dict:
        """Get symbol information and specifications"""
        
        if not self.initialize_mt5():
            return {}
        
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {self.symbol} not found")
                return {}
            
            info = {
                'name': symbol_info.name,
                'description': symbol_info.description,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'trade_mode': symbol_info.trade_mode,
                'min_lot': symbol_info.volume_min,
                'max_lot': symbol_info.volume_max,
                'lot_step': symbol_info.volume_step
            }
            
            self.logger.info(f"Symbol info for {self.symbol}: {info}")
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            return {}
        
        finally:
            mt5.shutdown()
    
    def test_connection(self) -> bool:
        """Test MT5 connection and symbol availability"""
        
        if not self.initialize_mt5():
            return False
        
        try:
            # Test symbol availability
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {self.symbol} not available")
                return False
            
            # Test data retrieval (last 10 candles)
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 10)
            if rates is None or len(rates) == 0:
                self.logger.error(f"Cannot retrieve data for {self.symbol}")
                return False
            
            self.logger.info(f"MT5 connection test successful for {self.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection test failed: {e}")
            return False
        
        finally:
            mt5.shutdown()
