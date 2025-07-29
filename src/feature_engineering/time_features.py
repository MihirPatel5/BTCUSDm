"""
Time Features Module
Generates time-based features for BTCUSD prediction model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

class TimeFeatures:
    """Generator for time-based features"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/processed/with_features")
        
    def generate_all_features(self) -> pd.DataFrame:
        """Generate all configured time-based features"""
        
        try:
            # Load data with existing features
            data_file = self.data_dir / "BTCUSD_5min_with_statistical_features.csv"
            if not data_file.exists():
                # Fallback to technical features
                data_file = self.data_dir / "BTCUSD_5min_with_technical_features.csv"
                if not data_file.exists():
                    # Fallback to processed data
                    data_file = Path("data/processed/BTCUSD_5min_processed.csv")
                    if not data_file.exists():
                        self.logger.error(f"Data file not found: {data_file}")
                        return None
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Generating time-based features for {len(data)} records")
            
            # Generate basic time features
            data = self._generate_basic_time_features(data)
            
            # Generate trading session features
            data = self._generate_trading_session_features(data)
            
            # Generate cyclical time features
            data = self._generate_cyclical_features(data)
            
            # Generate time since features
            data = self._generate_time_since_features(data)
            
            # Save feature-engineered data
            features_dir = Path("data/processed/with_features")
            features_dir.mkdir(parents=True, exist_ok=True)
            output_file = features_dir / "BTCUSD_5min_with_time_features.csv"
            data.to_csv(output_file, index=False)
            
            self.logger.info(f"Time-based features generated and saved to: {output_file}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error generating time-based features: {e}")
            return None
    
    def _generate_basic_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic time-based features"""
        
        # Extract time components
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
        data['day_of_month'] = data['timestamp'].dt.day
        data['month'] = data['timestamp'].dt.month
        data['quarter'] = data['timestamp'].dt.quarter
        data['year'] = data['timestamp'].dt.year
        
        # Weekend indicator
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Time of day categories
        data['time_of_day'] = pd.cut(
            data['hour'], 
            bins=[-1, 6, 12, 18, 24], 
            labels=['night', 'morning', 'afternoon', 'evening']
        ).astype(str)
        
        return data
    
    def _generate_trading_session_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading session features based on UTC time"""
        
        # Trading sessions in UTC (approximate)
        # Asian session: 00:00-09:00 UTC (09:00-18:00 Asia/Tokyo)
        # European session: 07:00-16:00 UTC (08:00-17:00 Europe/London)
        # US session: 13:00-22:00 UTC (08:00-17:00 US/Eastern)
        
        hour = data['timestamp'].dt.hour
        
        # Session indicators
        data['asian_session'] = ((hour >= 0) & (hour < 9)).astype(int)
        data['european_session'] = ((hour >= 7) & (hour < 16)).astype(int)
        data['us_session'] = ((hour >= 13) & (hour < 22)).astype(int)
        
        # Overlap periods
        data['asia_europe_overlap'] = ((hour >= 7) & (hour < 9)).astype(int)
        data['europe_us_overlap'] = ((hour >= 13) & (hour < 16)).astype(int)
        
        # Session transitions
        data['session_start'] = (hour.isin([0, 7, 13])).astype(int)
        data['session_end'] = (hour.isin([9, 16, 22])).astype(int)
        
        return data
    
    def _generate_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cyclical time features using sine/cosine transformations"""
        
        # Hour cyclical features (24-hour cycle)
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # Day of week cyclical features (7-day cycle)
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Month cyclical features (12-month cycle)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        return data
    
    def _generate_time_since_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate time since specific events or periods"""
        
        # Time since start of dataset (in hours)
        data['hours_since_start'] = (data['timestamp'] - data['timestamp'].iloc[0]).dt.total_seconds() / 3600
        
        # Time since start of week (in hours)
        data['hours_since_week_start'] = data['timestamp'].dt.dayofweek * 24 + data['timestamp'].dt.hour
        
        # Time since start of month (in hours)
        data['hours_since_month_start'] = (data['timestamp'].dt.day - 1) * 24 + data['timestamp'].dt.hour
        
        return data
