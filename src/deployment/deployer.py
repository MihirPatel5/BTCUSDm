"""
Production Deployment Module
Implements real-time deployment pipeline for BTCUSD prediction models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
import time
from datetime import datetime
import threading
from typing import Optional, Dict, Any

# For MT5 integration
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 not available. Using simulated data.")

# For real-time data fetching
import yfinance as yf


class Deployer:
    """Production deployment pipeline for real-time BTCUSD prediction"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models_dir = Path("models")
        self.deployment_dir = Path("deployment")
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load trained models
        self.xgboost_model = None
        self.lstm_model = None
        self.lstm_scaler = None
        self.ensemble_model = None
        
        # Load models
        self._load_models()
        
        # Initialize MT5 connection
        self.mt5_connected = False
        if MT5_AVAILABLE:
            self._initialize_mt5()
        
        # Trading state
        self.current_position = 0  # 0 = no position, 1 = long, -1 = short
        self.last_signal = None
        self.last_signal_time = None
        
        # Performance tracking
        self.performance_log = []
        
    def _load_models(self):
        """Load trained models"""
        
        try:
            # Load XGBoost model
            xgb_model_path = self.models_dir / "xgboost_model.pkl"
            if xgb_model_path.exists():
                self.xgboost_model = joblib.load(xgb_model_path)
                self.logger.info("XGBoost model loaded successfully")
            else:
                self.logger.warning(f"XGBoost model not found: {xgb_model_path}")
            
            # Load LSTM model and scaler
            lstm_model_path = self.models_dir / "lstm_model.h5"
            lstm_scaler_path = self.models_dir / "lstm_scaler.pkl"
            
            if lstm_model_path.exists() and lstm_scaler_path.exists():
                from tensorflow.keras.models import load_model
                self.lstm_model = load_model(lstm_model_path)
                self.lstm_scaler = joblib.load(lstm_scaler_path)
                self.logger.info("LSTM model and scaler loaded successfully")
            else:
                self.logger.warning(f"LSTM model or scaler not found")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def _initialize_mt5(self):
        """Initialize MetaTrader 5 connection"""
        
        try:
            if not mt5.initialize():
                self.logger.error("Failed to initialize MetaTrader 5")
                return
            
            # Check if we're connected
            if mt5.terminal_info().connected:
                self.mt5_connected = True
                self.logger.info("MetaTrader 5 connected successfully")
                
                # Log account info
                account_info = mt5.account_info()
                if account_info:
                    self.logger.info(f"Account: {account_info.login}, Balance: {account_info.balance}")
            else:
                self.logger.warning("MetaTrader 5 not connected")
                
        except Exception as e:
            self.logger.error(f"Error initializing MetaTrader 5: {e}")
    
    def get_latest_data(self) -> Optional[pd.DataFrame]:
        """Get latest market data for prediction"""
        
        try:
            # If MT5 is available, get data from MT5
            if self.mt5_connected:
                return self._get_mt5_data()
            else:
                # Fallback to Yahoo Finance
                return self._get_yahoo_data()
                
        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            return None
    
    def _get_mt5_data(self) -> Optional[pd.DataFrame]:
        """Get latest data from MetaTrader 5"""
        
        try:
            if not self.mt5_connected:
                return None
            
            # Get last 100 5-minute candles
            rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M5, 0, 100)
            
            if rates is None or len(rates) == 0:
                self.logger.warning("No data received from MT5")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })
            
            # Select required columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting MT5 data: {e}")
            return None
    
    def _get_yahoo_data(self) -> Optional[pd.DataFrame]:
        """Get latest data from Yahoo Finance (simulated)"""
        
        try:
            # Get last 7 days of 5-minute data
            ticker = yf.Ticker("BTC-USD")
            df = ticker.history(period="7d", interval="5m")
            
            if df.empty:
                self.logger.warning("No data received from Yahoo Finance")
                return None
            
            # Reset index to get timestamp as column
            df = df.reset_index()
            df = df.rename(columns={'Datetime': 'timestamp'})
            
            # Select required columns
            df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting Yahoo Finance data: {e}")
            return None
    
    def preprocess_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Preprocess data for prediction (same as training preprocessing)"""
        
        try:
            # Apply same preprocessing steps as used in training
            # This would include:
            # 1. Feature engineering (technical indicators, statistical features, time features)
            # 2. Feature selection
            # 3. Handling missing values
            
            # For now, we'll just ensure the data is properly formatted
            # In a real implementation, this would call the same preprocessing functions
            # used during training
            
            # Ensure data is sorted by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Handle missing values
            data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            self.logger.info(f"Preprocessed data with {len(data)} rows")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return None
    
    def make_prediction(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Make prediction using ensemble model"""
        
        try:
            if self.xgboost_model is None and self.lstm_model is None:
                self.logger.error("No models loaded for prediction")
                return None
            
            # Prepare features (same as in training)
            feature_columns = [col for col in data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            if not feature_columns:
                self.logger.error("No features found in data")
                return None
            
            X = data[feature_columns].iloc[-1:].copy()  # Use only the latest row
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            predictions = {}
            probabilities = {}
            
            # XGBoost prediction
            if self.xgboost_model:
                xgb_pred = self.xgboost_model.predict(X)[0]
                xgb_proba = self.xgboost_model.predict_proba(X)[0][1] if hasattr(self.xgboost_model, 'predict_proba') else xgb_pred
                predictions['xgboost'] = int(xgb_pred)
                probabilities['xgboost'] = float(xgb_proba)
            
            # LSTM prediction
            if self.lstm_model and self.lstm_scaler:
                X_scaled = self.lstm_scaler.transform(X)
                X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                lstm_pred_proba = self.lstm_model.predict(X_lstm)[0][0]
                lstm_pred = 1 if lstm_pred_proba > 0.5 else 0
                predictions['lstm'] = int(lstm_pred)
                probabilities['lstm'] = float(lstm_pred_proba)
            
            # Ensemble prediction (simple average)
            if predictions:
                ensemble_pred = round(np.mean(list(predictions.values())))
                ensemble_proba = np.mean(list(probabilities.values())) if probabilities else ensemble_pred
                predictions['ensemble'] = int(ensemble_pred)
                probabilities['ensemble'] = float(ensemble_proba)
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions,
                'probabilities': probabilities,
                'close_price': float(data['close'].iloc[-1])
            }
            
            self.logger.info(f"Prediction made: {predictions.get('ensemble', 'N/A')} with probability {probabilities.get('ensemble', 'N/A'):.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return None
    
    def apply_risk_management(self, prediction: int, probability: float) -> bool:
        """Apply risk management rules before executing trade"""
        
        try:
            # Get risk management parameters
            min_probability = self.config.get('risk_management', {}).get('min_probability', 0.55)
            max_positions = self.config.get('risk_management', {}).get('max_positions', 1)
            cooldown_minutes = self.config.get('risk_management', {}).get('signal_cooldown_minutes', 5)
            
            # Check probability threshold
            if probability < min_probability:
                self.logger.info(f"Signal rejected: Probability {probability:.4f} below threshold {min_probability}")
                return False
            
            # Check position limits
            if self.current_position != 0 and self.current_position == (1 if prediction == 1 else -1):
                self.logger.info("Signal rejected: Already in same position")
                return False
            
            # Check cooldown period
            if self.last_signal_time:
                minutes_since_last = (datetime.now() - self.last_signal_time).total_seconds() / 60
                if minutes_since_last < cooldown_minutes:
                    self.logger.info(f"Signal rejected: Cooldown period not expired ({minutes_since_last:.1f} minutes since last signal)")
                    return False
            
            # All checks passed
            return True
            
        except Exception as e:
            self.logger.error(f"Error in risk management: {e}")
            return False
    
    def execute_trade(self, prediction: int, probability: float) -> bool:
        """Execute trade based on prediction (simulated or real)"""
        
        try:
            # Apply risk management
            if not self.apply_risk_management(prediction, probability):
                return False
            
            # Determine trade action
            if prediction == 1:
                action = "BUY"
                new_position = 1
            else:
                action = "SELL"
                new_position = -1
            
            # Execute trade
            if self.mt5_connected:
                success = self._execute_mt5_trade(action, probability)
            else:
                success = self._execute_simulated_trade(action, probability)
            
            if success:
                # Update position tracking
                self.current_position = new_position
                self.last_signal = action
                self.last_signal_time = datetime.now()
                
                self.logger.info(f"Trade executed: {action} BTCUSD at probability {probability:.4f}")
                
                # Log performance
                self.performance_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': action,
                    'probability': probability,
                    'position': new_position
                })
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False
    
    def _execute_mt5_trade(self, action: str, probability: float) -> bool:
        """Execute real trade through MetaTrader 5"""
        
        try:
            if not self.mt5_connected:
                return False
            
            # Get symbol info
            symbol = "BTCUSD"
            symbol_info = mt5.symbol_info(symbol)
            
            if symbol_info is None:
                self.logger.error(f"Symbol {symbol} not found")
                return False
            
            # Check if symbol is available for trading
            if not symbol_info.visible:
                mt5.symbol_select(symbol, True)
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.error(f"Failed to get tick data for {symbol}")
                return False
            
            # Prepare trade request
            lot_size = self.config.get('risk_management', {}).get('lot_size', 0.1)
            
            if action == "BUY":
                trade_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:
                trade_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": trade_type,
                "price": price,
                "sl": price * 0.98 if action == "BUY" else price * 1.02,  # 2% stop loss
                "tp": price * 1.04 if action == "BUY" else price * 0.96,  # 4% take profit
                "deviation": 20,
                "magic": 234000,
                "comment": f"BTCUSD ML Signal (prob: {probability:.4f})",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return False
            
            self.logger.info(f"MT5 trade executed: {action} {lot_size} lots at {price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing MT5 trade: {e}")
            return False
    
    def _execute_simulated_trade(self, action: str, probability: float) -> bool:
        """Execute simulated trade for testing"""
        
        try:
            self.logger.info(f"SIMULATED TRADE: {action} BTCUSD (probability: {probability:.4f})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in simulated trade: {e}")
            return False
    
    def run_real_time_pipeline(self, interval_seconds: int = 300):
        """Run real-time prediction pipeline"""
        
        try:
            self.logger.info(f"Starting real-time prediction pipeline (interval: {interval_seconds}s)")
            
            while True:
                try:
                    # Get latest data
                    data = self.get_latest_data()
                    if data is None or len(data) < 10:
                        self.logger.warning("Insufficient data for prediction")
                        time.sleep(interval_seconds)
                        continue
                    
                    # Preprocess data
                    processed_data = self.preprocess_data(data)
                    if processed_data is None:
                        time.sleep(interval_seconds)
                        continue
                    
                    # Make prediction
                    prediction_result = self.make_prediction(processed_data)
                    if prediction_result is None:
                        time.sleep(interval_seconds)
                        continue
                    
                    # Log prediction
                    self.logger.info(f"Prediction: {prediction_result}")
                    
                    # Save prediction
                    pred_file = self.deployment_dir / f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(pred_file, 'w') as f:
                        json.dump(prediction_result, f, indent=2)
                    
                    # Execute trade if ensemble model predicts a signal
                    ensemble_pred = prediction_result['predictions'].get('ensemble')
                    ensemble_proba = prediction_result['probabilities'].get('ensemble', 0.5)
                    
                    if ensemble_pred is not None:
                        self.execute_trade(ensemble_pred, ensemble_proba)
                    
                    # Wait for next interval
                    time.sleep(interval_seconds)
                    
                except KeyboardInterrupt:
                    self.logger.info("Real-time pipeline stopped by user")
                    break
                
                except Exception as e:
                    self.logger.error(f"Error in real-time pipeline iteration: {e}")
                    time.sleep(interval_seconds)
            
        except Exception as e:
            self.logger.error(f"Error in real-time pipeline: {e}")
    
    def start_real_time_pipeline(self, interval_seconds: int = 300):
        """Start real-time pipeline in a separate thread"""
        
        try:
            thread = threading.Thread(
                target=self.run_real_time_pipeline,
                args=(interval_seconds,),
                daemon=True
            )
            thread.start()
            self.logger.info("Real-time pipeline started in background thread")
            return thread
            
        except Exception as e:
            self.logger.error(f"Error starting real-time pipeline: {e}")
            return None
    
    def get_performance_summary(self) -> dict:
        """Get performance summary of deployed model"""
        
        try:
            summary = {
                'current_position': self.current_position,
                'last_signal': self.last_signal,
                'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
                'total_signals': len(self.performance_log),
                'performance_log': self.performance_log[-10:]  # Last 10 signals
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown deployment pipeline"""
        
        try:
            if self.mt5_connected and MT5_AVAILABLE:
                mt5.shutdown()
                self.logger.info("MetaTrader 5 connection closed")
            
            # Save performance log
            log_file = self.deployment_dir / "performance_log.json"
            with open(log_file, 'w') as f:
                json.dump(self.performance_log, f, indent=2, default=str)
            
            self.logger.info("Deployment pipeline shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    # This allows the file to be run directly for testing
    pass
