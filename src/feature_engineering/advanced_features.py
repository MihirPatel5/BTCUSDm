import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats


class AdvancedFeatures:
    """Generator for advanced features to improve model performance"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/processed/with_features")
    
    def generate_all_features(self) -> pd.DataFrame:
        """Generate all advanced features"""
        
        try:
            # Load data with existing features
            data_file = self.data_dir / "BTCUSD_5min_with_time_features.csv"
            if not data_file.exists():
                self.logger.error(f"Feature data file not found: {data_file}")
                return None
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Generating advanced features for {len(data)} records")
            
            # Generate volatility regime features
            data = self._generate_volatility_regimes(data)
            
            # Generate momentum divergence features
            data = self._generate_momentum_divergence(data)
            
            # Generate support/resistance features
            data = self._generate_support_resistance(data)
            
            # Generate order flow approximation features
            data = self._generate_order_flow_features(data)
            
            # Generate fractal features
            data = self._generate_fractal_features(data)
            
            # Save feature-engineered data
            features_dir = Path("data/processed/with_features")
            features_dir.mkdir(parents=True, exist_ok=True)
            output_file = features_dir / "BTCUSD_5min_with_advanced_features.csv"
            data.to_csv(output_file, index=False)
            
            self.logger.info(f"Advanced features generated and saved to: {output_file}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error generating advanced features: {e}")
            return None
    
    def _generate_volatility_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility regime features"""
        
        # Calculate volatility using different methods
        returns = data['close'].pct_change()
        
        # Short-term volatility (5 periods)
        data['vol_short'] = returns.rolling(window=5).std()
        
        # Medium-term volatility (20 periods)
        data['vol_medium'] = returns.rolling(window=20).std()
        
        # Long-term volatility (50 periods)
        data['vol_long'] = returns.rolling(window=50).std()
        
        # Volatility ratios
        data['vol_ratio_short_medium'] = data['vol_short'] / data['vol_medium']
        data['vol_ratio_short_long'] = data['vol_short'] / data['vol_long']
        
        # Volatility regime indicators
        data['low_volatility_regime'] = (data['vol_short'] < data['vol_medium']).astype(int)
        data['high_volatility_regime'] = (data['vol_short'] > data['vol_medium']).astype(int)
        
        return data
    
    def _generate_momentum_divergence(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum divergence features"""
        
        # Price momentum (20 periods)
        data['price_momentum'] = data['close'] / data['close'].shift(20) - 1
        
        # RSI momentum (if RSI exists)
        if 'rsi' in data.columns:
            data['rsi_momentum'] = data['rsi'] - data['rsi'].shift(5)
            
            # RSI divergence
            data['rsi_divergence'] = data['price_momentum'] * data['rsi_momentum']
            
            # Hidden divergence (price makes new low but RSI doesn't)
            data['hidden_bullish_divergence'] = (
                (data['close'] < data['close'].shift(5)) & 
                (data['rsi'] > data['rsi'].shift(5))
            ).astype(int)
            
            # Regular divergence (price makes new high but RSI doesn't)
            data['regular_bearish_divergence'] = (
                (data['close'] > data['close'].shift(5)) & 
                (data['rsi'] < data['rsi'].shift(5))
            ).astype(int)
        
        return data
    
    def _generate_support_resistance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate support and resistance level features"""
        
        window = 20
        
        # Rolling high and low
        data['rolling_high'] = data['high'].rolling(window=window).max()
        data['rolling_low'] = data['low'].rolling(window=window).min()
        
        # Distance to recent high/low
        data['dist_to_high'] = (data['rolling_high'] - data['close']) / data['close']
        data['dist_to_low'] = (data['close'] - data['rolling_low']) / data['close']
        
        # Breakout signals
        data['high_breakout'] = (data['close'] > data['rolling_high'].shift(1)).astype(int)
        data['low_breakout'] = (data['close'] < data['rolling_low'].shift(1)).astype(int)
        
        return data
    
    def _generate_order_flow_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate order flow approximation features"""
        
        # Price-weighted volume
        data['price_volume'] = data['close'] * data['volume']
        
        # Volume moving averages
        data['volume_sma_5'] = data['volume'].rolling(window=5).mean()
        data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
        
        # Volume ratio
        data['volume_ratio'] = data['volume'] / data['volume_sma_20']
        
        # Volume trend
        data['volume_trend'] = data['volume'].diff().rolling(window=5).mean()
        
        # Volume-price trend (similar to OBV but simplified)
        data['volume_price_trend'] = (data['close'].diff() * data['volume']).cumsum()
        
        return data
    
    def _generate_fractal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate fractal-based features"""
        
        # Simple fractal detection
        def is_bullish_fractal(i, df):
            if i < 2 or i > len(df) - 3:
                return False
            return (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                   df['low'].iloc[i] < df['low'].iloc[i-2] and
                   df['low'].iloc[i] < df['low'].iloc[i+1] and
                   df['low'].iloc[i] < df['low'].iloc[i+2])
        
        def is_bearish_fractal(i, df):
            if i < 2 or i > len(df) - 3:
                return False
            return (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                   df['high'].iloc[i] > df['high'].iloc[i-2] and
                   df['high'].iloc[i] > df['high'].iloc[i+1] and
                   df['high'].iloc[i] > df['high'].iloc[i+2])
        
        # Apply fractal detection
        bullish_fractals = [is_bullish_fractal(i, data) for i in range(len(data))]
        bearish_fractals = [is_bearish_fractal(i, data) for i in range(len(data))]
        
        data['bullish_fractal'] = pd.Series(bullish_fractals).astype(int)
        data['bearish_fractal'] = pd.Series(bearish_fractals).astype(int)
        
        # Fractal-based signals
        data['fractal_signal'] = data['bullish_fractal'] - data['bearish_fractal']
        
        return data
