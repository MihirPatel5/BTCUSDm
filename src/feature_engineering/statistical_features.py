"""
Statistical Features Module
Generates statistical analysis features for BTCUSD prediction model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats

class StatisticalFeatures:
    """Generator for statistical analysis features"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/processed/with_features")
        
    def generate_all_features(self) -> pd.DataFrame:
        """Generate all configured statistical features"""
        
        try:
            # Load data with technical features
            data_file = self.data_dir / "BTCUSD_5min_with_technical_features.csv"
            if not data_file.exists():
                # Fallback to processed data
                data_file = Path("data/processed/BTCUSD_5min_processed.csv")
                if not data_file.exists():
                    self.logger.error(f"Data file not found: {data_file}")
                    return None
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Generating statistical features for {len(data)} records")
            
            # Generate volatility features
            data = self._generate_volatility_features(data)
            
            # Generate distribution features
            data = self._generate_distribution_features(data)
            
            # Generate autocorrelation features
            data = self._generate_autocorrelation_features(data)
            
            # Generate rolling statistics
            data = self._generate_rolling_statistics(data)
            
            # Save feature-engineered data
            features_dir = Path("data/processed/with_features")
            features_dir.mkdir(parents=True, exist_ok=True)
            output_file = features_dir / "BTCUSD_5min_with_statistical_features.csv"
            data.to_csv(output_file, index=False)
            
            self.logger.info(f"Statistical features generated and saved to: {output_file}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error generating statistical features: {e}")
            return None
    
    def _generate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based statistical features"""
        
        # Volatility periods from config
        vol_periods = self.config.get('features', {}).get('statistical_features', {}).get('volatility_periods', [5, 10, 20])
        
        # Price returns
        data['returns'] = data['close'].pct_change()
        
        for period in vol_periods:
            # Rolling volatility (standard deviation of returns)
            data[f'volatility_{period}'] = data['returns'].rolling(window=period).std()
            
            # Realized volatility (sum of squared returns)
            data[f'realized_vol_{period}'] = (
                data['returns'].rolling(window=period).apply(lambda x: np.sqrt(np.sum(x**2)))
            )
            
            # Parkinson volatility (using high/low prices)
            if 'high' in data.columns and 'low' in data.columns:
                data[f'parkinson_vol_{period}'] = (
                    np.sqrt(1 / (4 * np.log(2))) * 
                    np.sqrt((np.log(data['high'] / data['low']) ** 2).rolling(window=period).mean())
                )
        
        # Volatility clustering (autocorrelation of volatility)
        data['volatility_cluster'] = data['returns'].rolling(window=10).std().pct_change().rolling(window=10).mean()
        
        return data
    
    def _generate_distribution_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate distribution-based statistical features"""
        
        window = 20  # Standard window for distribution analysis
        
        # Rolling skewness
        data['skewness'] = data['returns'].rolling(window=window).apply(stats.skew, raw=True)
        
        # Rolling kurtosis
        data['kurtosis'] = data['returns'].rolling(window=window).apply(stats.kurtosis, raw=True)
        
        # Z-score normalization
        data['z_score'] = (data['close'] - data['close'].rolling(window=window).mean()) / data['close'].rolling(window=window).std()
        
        # Percentile ranks
        data['percentile_rank'] = data['close'].rolling(window=window).rank(pct=True)
        
        # Value at Risk (VaR) approximations
        data['var_95'] = data['returns'].rolling(window=window).mean() - 1.645 * data['returns'].rolling(window=window).std()
        data['var_99'] = data['returns'].rolling(window=window).mean() - 2.33 * data['returns'].rolling(window=window).std()
        
        return data
    
    def _generate_autocorrelation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate autocorrelation-based features"""
        
        # Autocorrelation lags from config
        lags = self.config.get('features', {}).get('statistical_features', {}).get('autocorr_lags', [1, 2, 3, 5, 10])
        
        for lag in lags:
            data[f'autocorr_lag_{lag}'] = data['returns'].rolling(window=lag*2).apply(
                lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        # Hurst exponent approximation (using rescaled range)
        def hurst_exponent(ts, max_lag=20):
            """Calculate Hurst exponent""" 
            if len(ts) < max_lag * 2:
                return np.nan
            
            lags = range(2, min(max_lag, len(ts) // 4))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            # Avoid NaNs in log transformation
            tau = [x for x in tau if x > 0 and not np.isnan(x)]
            lags = [lags[i] for i in range(len(tau))]
            
            if len(tau) < 2:
                return np.nan
            
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        data['hurst_exponent'] = data['close'].rolling(window=100).apply(hurst_exponent, raw=False)
        
        return data
    
    def _generate_rolling_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling statistical features"""
        
        # Rolling mean ratios
        data['price_to_sma_ratio'] = data['close'] / data['sma_20'] if 'sma_20' in data.columns else np.nan
        
        # Rolling standard deviation ratios
        data['std_ratio'] = data['returns'].rolling(window=10).std() / data['returns'].rolling(window=50).std()
        
        # Bollinger Band position (if available)
        if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
            data['bb_position_normalized'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # RSI momentum
        if 'rsi' in data.columns:
            data['rsi_momentum'] = data['rsi'].diff()
            
            # RSI overbought/oversold signals
            data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
            data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
        
        # MACD histogram trend
        if 'macd_histogram' in data.columns:
            data['macd_hist_trend'] = data['macd_histogram'].diff()
            data['macd_signal_cross'] = ((data['macd'] > data['macd_signal']) & 
                                        (data['macd'].shift(1) <= data['macd_signal'].shift(1))).astype(int)
        
        return data
