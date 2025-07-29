"""
Continuous Improvement Module
Implements automated model retraining and feature updates for BTCUSD prediction models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
from datetime import datetime, timedelta
import threading
import time

# Import feature engineering modules
from src.feature_engineering.technical_indicators import TechnicalIndicators
from src.feature_engineering.statistical_features import StatisticalFeatures
from src.feature_engineering.time_features import TimeFeatures
from src.feature_engineering.feature_selector import FeatureSelector

# Import target definition module
from src.target_definition.target_creator import TargetCreator

# Import models
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel


class ContinuousImprover:
    """Continuous improvement pipeline for BTCUSD prediction models"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data")
        self.models_dir = Path("models")
        self.improvement_dir = Path("improvement")
        self.improvement_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_selector = FeatureSelector(config)
        self.target_creator = TargetCreator(config)
        
        # Retraining schedule
        self.last_retraining = None
        self.retraining_frequency = config.get('improvement', {}).get('retraining_frequency_days', 7)
        
        # Performance thresholds for retraining
        self.accuracy_threshold = config.get('improvement', {}).get('accuracy_threshold', 0.55)
        self.sharpe_threshold = config.get('improvement', {}).get('sharpe_threshold', 1.0)
        
    def check_retraining_trigger(self, current_performance: dict) -> bool:
        """Check if model retraining should be triggered"""
        
        try:
            # Check performance thresholds
            should_retrain = False
            reasons = []
            
            # Check accuracy
            current_accuracy = current_performance.get('accuracy', 0)
            if current_accuracy < self.accuracy_threshold:
                should_retrain = True
                reasons.append(f"Accuracy below threshold: {current_accuracy:.4f} < {self.accuracy_threshold}")
            
            # Check Sharpe ratio
            current_sharpe = current_performance.get('sharpe_ratio', 0)
            if current_sharpe < self.sharpe_threshold:
                should_retrain = True
                reasons.append(f"Sharpe ratio below threshold: {current_sharpe:.4f} < {self.sharpe_threshold}")
            
            # Check time since last retraining
            if self.last_retraining:
                days_since_retraining = (datetime.now() - self.last_retraining).days
                if days_since_retraining >= self.retraining_frequency:
                    should_retrain = True
                    reasons.append(f"Time since last retraining: {days_since_retraining} days >= {self.retraining_frequency} days")
            else:
                should_retrain = True
                reasons.append("Initial retraining required")
            
            if should_retrain:
                self.logger.info(f"Retraining triggered: {', '.join(reasons)}")
            
            return should_retrain
            
        except Exception as e:
            self.logger.error(f"Error checking retraining trigger: {e}")
            return False
    
    def update_features(self, new_data_file: str) -> bool:
        """Update features with new data"""
        
        try:
            self.logger.info("Updating features with new data")
            
            # Load new data
            new_data_path = Path(new_data_file)
            if not new_data_path.exists():
                self.logger.error(f"New data file not found: {new_data_path}")
                return False
            
            new_data = pd.read_csv(new_data_path)
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
            
            # Apply feature engineering
            # Technical indicators
            ti = TechnicalIndicators(self.config)
            data_with_ti = ti.calculate_indicators(new_data.copy())
            
            # Statistical features
            sf = StatisticalFeatures(self.config)
            data_with_stats = sf.calculate_features(data_with_ti.copy())
            
            # Time features
            tf = TimeFeatures(self.config)
            data_with_time = tf.calculate_features(data_with_stats.copy())
            
            # Save updated features
            features_dir = self.data_dir / "processed" / "with_features"
            features_dir.mkdir(parents=True, exist_ok=True)
            
            updated_features_file = features_dir / f"BTCUSD_5min_with_features_{datetime.now().strftime('%Y%m%d')}.csv"
            data_with_time.to_csv(updated_features_file, index=False)
            
            self.logger.info(f"Updated features saved to: {updated_features_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating features: {e}")
            return False
    
    def update_targets(self, data_file: str) -> bool:
        """Update targets with new data"""
        
        try:
            self.logger.info("Updating targets with new data")
            
            # Use target creator to generate targets
            success = self.target_creator.create_targets(data_file)
            
            if success:
                self.logger.info("Targets updated successfully")
            else:
                self.logger.error("Failed to update targets")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating targets: {e}")
            return False
    
    def retrain_models(self, data_file: str) -> dict:
        """Retrain all models with updated data"""
        
        try:
            self.logger.info("Retraining models with updated data")
            
            # Load data with targets
            data_path = Path(data_file)
            if not data_path.exists():
                self.logger.error(f"Data file not found: {data_path}")
                return {}
            
            data = pd.read_csv(data_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                               'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
            
            X = data[feature_columns]
            y = data['target_binary']
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Split data
            split_ratio = self.config.get('model', {}).get('train_test_split', 0.8)
            split_index = int(len(data) * split_ratio)
            
            X_train = X.iloc[:split_index]
            y_train = y.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_test = y.iloc[split_index:]
            
            # Retrain XGBoost model
            self.logger.info("Retraining XGBoost model")
            xgb_model = XGBoostModel(self.config)
            xgb_results = xgb_model.train(X_train, y_train, X_test, y_test)
            
            # Retrain LSTM model
            self.logger.info("Retraining LSTM model")
            lstm_model = LSTMModel(self.config)
            lstm_results = lstm_model.train(X_train, y_train, X_test, y_test)
            
            # Save retrained models
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save XGBoost model
            xgb_model_path = self.models_dir / f"xgboost_model_{timestamp}.pkl"
            xgb_model.save_model(xgb_model_path)
            
            # Save LSTM model and scaler
            lstm_model_path = self.models_dir / f"lstm_model_{timestamp}.h5"
            lstm_scaler_path = self.models_dir / f"lstm_scaler_{timestamp}.pkl"
            lstm_model.save_model(lstm_model_path, lstm_scaler_path)
            
            results = {
                'timestamp': timestamp,
                'xgboost_results': xgb_results,
                'lstm_results': lstm_results,
                'models_saved': {
                    'xgboost': str(xgb_model_path),
                    'lstm_model': str(lstm_model_path),
                    'lstm_scaler': str(lstm_scaler_path)
                }
            }
            
            # Save retraining results
            results_file = self.improvement_dir / f"retraining_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Retraining completed. Results saved to: {results_file}")
            
            # Update last retraining timestamp
            self.last_retraining = datetime.now()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")
            return {}
    
    def evaluate_model_improvement(self, previous_results: dict, new_results: dict) -> dict:
        """Evaluate improvement between model versions"""
        
        try:
            self.logger.info("Evaluating model improvement")
            
            # Compare XGBoost results
            prev_xgb_f1 = previous_results.get('xgboost_results', {}).get('f1_score', 0)
            new_xgb_f1 = new_results.get('xgboost_results', {}).get('f1_score', 0)
            
            xgb_improvement = new_xgb_f1 - prev_xgb_f1
            
            # Compare LSTM results
            prev_lstm_f1 = previous_results.get('lstm_results', {}).get('f1_score', 0)
            new_lstm_f1 = new_results.get('lstm_results', {}).get('f1_score', 0)
            
            lstm_improvement = new_lstm_f1 - prev_lstm_f1
            
            # Overall improvement
            overall_improvement = (xgb_improvement + lstm_improvement) / 2
            
            evaluation = {
                'timestamp': datetime.now().isoformat(),
                'xgboost_improvement': xgb_improvement,
                'lstm_improvement': lstm_improvement,
                'overall_improvement': overall_improvement,
                'improved': overall_improvement > 0,
                'significant_improvement': abs(overall_improvement) > 0.01
            }
            
            self.logger.info(f"Model improvement evaluation: {evaluation}")
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating model improvement: {e}")
            return {}
    
    def deploy_improved_models(self, retraining_results: dict) -> bool:
        """Deploy improved models to production"""
        
        try:
            self.logger.info("Deploying improved models to production")
            
            # Get model paths
            models_saved = retraining_results.get('models_saved', {})
            
            if not models_saved:
                self.logger.error("No models found in retraining results")
                return False
            
            # Deploy XGBoost model
            xgb_model_path = models_saved.get('xgboost')
            if xgb_model_path:
                # Copy to production model location
                prod_xgb_path = self.models_dir / "xgboost_model.pkl"
                import shutil
                shutil.copy2(xgb_model_path, prod_xgb_path)
                self.logger.info(f"XGBoost model deployed to: {prod_xgb_path}")
            
            # Deploy LSTM model and scaler
            lstm_model_path = models_saved.get('lstm_model')
            lstm_scaler_path = models_saved.get('lstm_scaler')
            
            if lstm_model_path and lstm_scaler_path:
                # Copy to production model location
                prod_lstm_path = self.models_dir / "lstm_model.h5"
                prod_scaler_path = self.models_dir / "lstm_scaler.pkl"
                
                import shutil
                shutil.copy2(lstm_model_path, prod_lstm_path)
                shutil.copy2(lstm_scaler_path, prod_scaler_path)
                
                self.logger.info(f"LSTM model deployed to: {prod_lstm_path}")
                self.logger.info(f"LSTM scaler deployed to: {prod_scaler_path}")
            
            self.logger.info("Improved models deployed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying improved models: {e}")
            return False
    
    def run_improvement_cycle(self, current_performance: dict = None) -> dict:
        """Run a complete improvement cycle"""
        
        try:
            self.logger.info("Starting improvement cycle")
            
            # Check if retraining is needed
            if current_performance and not self.check_retraining_trigger(current_performance):
                self.logger.info("Retraining not triggered. Skipping improvement cycle.")
                return {'status': 'skipped', 'reason': 'Retraining not triggered'}
            
            # Get latest data
            latest_data_file = self.data_dir / "processed" / "BTCUSD_5min_processed.csv"
            if not latest_data_file.exists():
                self.logger.error(f"Latest data file not found: {latest_data_file}")
                return {'status': 'failed', 'reason': 'Latest data file not found'}
            
            # Update features
            if not self.update_features(str(latest_data_file)):
                self.logger.error("Failed to update features")
                return {'status': 'failed', 'reason': 'Failed to update features'}
            
            # Update targets
            features_file = self.data_dir / "processed" / "with_features" / f"BTCUSD_5min_with_features_{datetime.now().strftime('%Y%m%d')}.csv"
            if not self.update_targets(str(features_file)):
                self.logger.error("Failed to update targets")
                return {'status': 'failed', 'reason': 'Failed to update targets'}
            
            # Retrain models
            targets_file = self.data_dir / "processed" / "with_targets" / "BTCUSD_5min_with_targets.csv"
            retraining_results = self.retrain_models(str(targets_file))
            
            if not retraining_results:
                self.logger.error("Failed to retrain models")
                return {'status': 'failed', 'reason': 'Failed to retrain models'}
            
            # Deploy improved models
            if not self.deploy_improved_models(retraining_results):
                self.logger.error("Failed to deploy improved models")
                return {'status': 'failed', 'reason': 'Failed to deploy improved models'}
            
            # Save improvement cycle results
            cycle_results = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'retraining_results': retraining_results
            }
            
            results_file = self.improvement_dir / f"improvement_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(cycle_results, f, indent=2, default=str)
            
            self.logger.info(f"Improvement cycle completed. Results saved to: {results_file}")
            
            return cycle_results
            
        except Exception as e:
            self.logger.error(f"Error in improvement cycle: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def start_continuous_improvement(self, check_interval_hours: int = 24):
        """Start continuous improvement in a separate thread"""
        
        def improvement_loop():
            while True:
                try:
                    # Run improvement cycle
                    results = self.run_improvement_cycle()
                    
                    # Wait for next check
                    time.sleep(check_interval_hours * 3600)
                    
                except KeyboardInterrupt:
                    self.logger.info("Continuous improvement stopped by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error in continuous improvement: {e}")
                    time.sleep(check_interval_hours * 3600)
        
        try:
            thread = threading.Thread(target=improvement_loop, daemon=True)
            thread.start()
            self.logger.info(f"Continuous improvement started (check interval: {check_interval_hours} hours)")
            return thread
            
        except Exception as e:
            self.logger.error(f"Error starting continuous improvement: {e}")
            return None
    
    def get_improvement_summary(self) -> dict:
        """Get summary of continuous improvement status"""
        
        try:
            summary = {
                'last_retraining': self.last_retraining.isoformat() if self.last_retraining else None,
                'retraining_frequency_days': self.retraining_frequency,
                'accuracy_threshold': self.accuracy_threshold,
                'sharpe_threshold': self.sharpe_threshold,
                'next_scheduled_retraining': None
            }
            
            # Calculate next scheduled retraining
            if self.last_retraining:
                next_retraining = self.last_retraining + timedelta(days=self.retraining_frequency)
                summary['next_scheduled_retraining'] = next_retraining.isoformat()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting improvement summary: {e}")
            return {}
