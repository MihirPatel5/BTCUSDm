"""
Technical Indicators Module
Generates technical analysis features for BTCUSD prediction model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Try to import TA-Lib, use fallback if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using fallback implementations.")

# Try to import alternative TA library
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

class TechnicalIndicators:
    """Generator for technical analysis indicators"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/processed")
        
    def generate_all_indicators(self) -> pd.DataFrame:
        """Generate all configured technical indicators"""
        
        try:
            # Load processed data
            data_file = self.data_dir / "BTCUSD_5min_processed.csv"
            if not data_file.exists():
                self.logger.error(f"Processed data file not found: {data_file}")
                return None
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Generating technical indicators for {len(data)} records")
            
            # Convert to numpy arrays for TA-Lib
            open_prices = data['open'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values
            volume = data['volume'].values if 'volume' in data.columns else np.zeros(len(data))
            
            # Generate moving averages
            data = self._generate_moving_averages(data, close_prices)
            
            # Generate oscillators
            data = self._generate_oscillators(data, open_prices, high_prices, low_prices, close_prices)
            
            # Generate volatility indicators
            data = self._generate_volatility_indicators(data, high_prices, low_prices, close_prices)
            
            # Generate momentum indicators
            data = self._generate_momentum_indicators(data, open_prices, high_prices, low_prices, close_prices)
            
            # Generate volume indicators
            data = self._generate_volume_indicators(data, high_prices, low_prices, close_prices, volume)
            
            # Save feature-engineered data
            features_dir = Path("data/processed/with_features")
            features_dir.mkdir(parents=True, exist_ok=True)
            output_file = features_dir / "BTCUSD_5min_with_technical_features.csv"
            data.to_csv(output_file, index=False)
            
            self.logger.info(f"Technical indicators generated and saved to: {output_file}")
            self.logger.info(f"Total features: {len(data.columns) - 7}")  # Exclude timestamp, OHLCV, data_complete
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error generating technical indicators: {e}")
            return None
    
    def _generate_moving_averages(self, data: pd.DataFrame, close_prices: np.array) -> pd.DataFrame:
        """Generate simple and exponential moving averages"""
        
        # Simple Moving Averages
        sma_periods = self.config.get('features', {}).get('technical_indicators', {}).get('sma_periods', [5, 10, 20, 50, 100, 200])
        
        for period in sma_periods:
            try:
                if TALIB_AVAILABLE:
                    data[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
                else:
                    # Fallback: pandas rolling mean
                    data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            except Exception as e:
                self.logger.warning(f"Error calculating SMA {period}: {e}")
        
        # Exponential Moving Averages
        ema_periods = self.config.get('features', {}).get('technical_indicators', {}).get('ema_periods', [5, 10, 20, 50])
        
        for period in ema_periods:
            try:
                if TALIB_AVAILABLE:
                    data[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
                else:
                    # Fallback: pandas exponential weighted mean
                    data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            except Exception as e:
                self.logger.warning(f"Error calculating EMA {period}: {e}")
        
        # Moving Average Convergence Divergence (MACD)
        macd_params = self.config.get('features', {}).get('technical_indicators', {}).get('macd', [12, 26, 9])
        
        try:
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices, 
                fastperiod=macd_params[0], 
                slowperiod=macd_params[1], 
                signalperiod=macd_params[2]
            )
            data['macd'] = macd
            data['macd_signal'] = macd_signal
            data['macd_histogram'] = macd_hist
        except Exception as e:
            self.logger.warning(f"Error calculating MACD: {e}")
        
        return data
    
    def _generate_oscillators(self, data: pd.DataFrame, open_prices: np.array, high_prices: np.array, 
                             low_prices: np.array, close_prices: np.array) -> pd.DataFrame:
        """Generate oscillator-based indicators"""
        
        # Relative Strength Index (RSI)
        rsi_period = self.config.get('features', {}).get('technical_indicators', {}).get('rsi_period', 14)
        
        try:
            data['rsi'] = talib.RSI(close_prices, timeperiod=rsi_period)
        except Exception as e:
            self.logger.warning(f"Error calculating RSI: {e}")
        
        # Stochastic Oscillator
        stoch_params = self.config.get('features', {}).get('technical_indicators', {}).get('stochastic', [14, 3, 3])
        
        try:
            slowk, slowd = talib.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=stoch_params[0],
                slowk_period=stoch_params[1],
                slowk_matype=0,
                slowd_period=stoch_params[2],
                slowd_matype=0
            )
            data['stoch_k'] = slowk
            data['stoch_d'] = slowd
        except Exception as e:
            self.logger.warning(f"Error calculating Stochastic: {e}")
        
        # Williams %R
        williams_r_period = self.config.get('features', {}).get('technical_indicators', {}).get('williams_r_period', 14)
        
        try:
            data['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=williams_r_period)
        except Exception as e:
            self.logger.warning(f"Error calculating Williams %R: {e}")
        
        return data
    
    def _generate_volatility_indicators(self, data: pd.DataFrame, high_prices: np.array, 
                                       low_prices: np.array, close_prices: np.array) -> pd.DataFrame:
        """Generate volatility-based indicators"""
        
        # Bollinger Bands
        bb_params = self.config.get('features', {}).get('technical_indicators', {}).get('bollinger_bands', [20, 2])
        
        try:
            upper, middle, lower = talib.BBANDS(
                close_prices,
                timeperiod=bb_params[0],
                nbdevup=bb_params[1],
                nbdevdn=bb_params[1],
                matype=0
            )
            data['bb_upper'] = upper
            data['bb_middle'] = middle
            data['bb_lower'] = lower
            
            # Bollinger Band features
            data['bb_width'] = (upper - lower) / middle
            data['bb_position'] = (close_prices - lower) / (upper - lower)
        except Exception as e:
            self.logger.warning(f"Error calculating Bollinger Bands: {e}")
        
        # Average True Range (ATR)
        atr_period = 14  # Standard period
        
        try:
            data['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)
        except Exception as e:
            self.logger.warning(f"Error calculating ATR: {e}")
        
        return data
    
    def _generate_momentum_indicators(self, data: pd.DataFrame, open_prices: np.array, high_prices: np.array, 
                                     low_prices: np.array, close_prices: np.array) -> pd.DataFrame:
        """Generate momentum-based indicators"""
        
        # Rate of Change (ROC)
        roc_periods = [1, 2, 3, 5, 10]
        
        for period in roc_periods:
            try:
                data[f'roc_{period}'] = talib.ROC(close_prices, timeperiod=period)
            except Exception as e:
                self.logger.warning(f"Error calculating ROC {period}: {e}")
        
        # Price Rate of Change
        try:
            data['price_roc'] = talib.ROCP(close_prices, timeperiod=1)
        except Exception as e:
            self.logger.warning(f"Error calculating Price ROC: {e}")
        
        # Average Directional Index (ADX)
        adx_period = self.config.get('features', {}).get('technical_indicators', {}).get('adx_period', 14)
        
        try:
            data['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=adx_period)
            data['adx_pos'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=adx_period)
            data['adx_neg'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=adx_period)
        except Exception as e:
            self.logger.warning(f"Error calculating ADX: {e}")
        
        return data
    
    def _generate_volume_indicators(self, data: pd.DataFrame, high_prices: np.array, low_prices: np.array, 
                                   close_prices: np.array, volume: np.array) -> pd.DataFrame:
        """Generate volume-based indicators"""
        
        # On-Balance Volume (OBV)
        try:
            data['obv'] = talib.OBV(close_prices, volume)
        except Exception as e:
            self.logger.warning(f"Error calculating OBV: {e}")
        
        # Chaikin A/D Oscillator
        try:
            data['chaikin_ad'] = talib.AD(high_prices, low_prices, close_prices, volume)
        except Exception as e:
            self.logger.warning(f"Error calculating Chaikin A/D: {e}")
        
        # Volume Rate of Change
        try:
            data['volume_roc'] = talib.ROC(volume, timeperiod=5)
        except Exception as e:
            self.logger.warning(f"Error calculating Volume ROC: {e}")
        
        return data
