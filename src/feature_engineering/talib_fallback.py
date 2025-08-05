"""
TA-Lib Fallback Implementations
Provides fallback implementations for technical indicators when TA-Lib is not available
"""

import pandas as pd
import numpy as np


def sma(data, period):
    """Simple Moving Average fallback"""
    return data.rolling(window=period).mean()


def ema(data, period):
    """Exponential Moving Average fallback"""
    return data.ewm(span=period).mean()


def rsi(data, period=14):
    """Relative Strength Index fallback"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(data, fast=12, slow=26, signal=9):
    """MACD fallback"""
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(data, period=20, std_dev=2):
    """Bollinger Bands fallback"""
    sma_line = sma(data, period)
    std = data.rolling(window=period).std()
    upper_band = sma_line + (std * std_dev)
    lower_band = sma_line - (std * std_dev)
    return upper_band, sma_line, lower_band


def stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator fallback"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent


def atr(high, low, close, period=14):
    """Average True Range fallback"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def adx(high, low, close, period=14):
    """Average Directional Index fallback (simplified)"""
    # Simplified ADX calculation
    tr = atr(high, low, close, 1)
    dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
    dm_minus = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
    
    dm_plus_series = pd.Series(dm_plus, index=high.index)
    dm_minus_series = pd.Series(dm_minus, index=low.index)
    
    di_plus = 100 * (dm_plus_series.rolling(window=period).mean() / tr.rolling(window=period).mean())
    di_minus = 100 * (dm_minus_series.rolling(window=period).mean() / tr.rolling(window=period).mean())
    
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_line = dx.rolling(window=period).mean()
    
    return adx_line


def williams_r(high, low, close, period=14):
    """Williams %R fallback"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low))


def roc(data, period=10):
    """Rate of Change fallback"""
    return ((data - data.shift(period)) / data.shift(period)) * 100


def momentum(data, period=10):
    """Momentum fallback"""
    return data - data.shift(period)


def obv(close, volume):
    """On Balance Volume fallback"""
    obv_values = []
    obv_val = 0
    
    for i in range(len(close)):
        if i == 0:
            obv_val = volume.iloc[i]
        else:
            if close.iloc[i] > close.iloc[i-1]:
                obv_val += volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv_val -= volume.iloc[i]
            # If close is unchanged, OBV remains the same
        
        obv_values.append(obv_val)
    
    return pd.Series(obv_values, index=close.index)
