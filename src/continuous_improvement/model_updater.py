"""
Model Updater Module
Implements continuous improvement and automated retraining for BTCUSD prediction models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import schedule
import time


class ModelUpdater:
    """Continuous improvement and automated retraining system"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models_dir = Path("models")
        self.data_dir = Path("data")
        self.improvement_dir = Path("continuous_improvement")
        self.improvement_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_history = []
        self.retrain_threshold = config.get('continuous_improvement', {}).get('retrain_threshold', 0.05)
        self.min_retrain_interval_days = config.get('continuous_improvement', {}).get('min_retrain_interval_days', 7)
        
    def monitor_model_performance(self) -> Dict[str, Any]:
        """Monitor current model performance and decide if retraining is needed"""
        
        try:
            self.logger.info("Monitoring model performance...")
            
            # Load recent predictions and actual results
            performance_data = self._load_recent_performance()
            
            if performance_data is None:
                self.logger.warning("No recent performance data available")
                return {'retrain_needed': False, 'reason': 'No performance data'}
            
            # Calculate current performance metrics
            current_metrics = self._calculate_performance_metrics(performance_data)
            
            # Compare with historical performance
            retrain_decision = self._should_retrain(current_metrics)
            
            # Log performance
            self._log_performance(current_metrics, retrain_decision)
            
            return {
                'current_metrics': current_metrics,
                'retrain_needed': retrain_decision['needed'],
                'reason': retrain_decision['reason'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring model performance: {e}")
            return {'retrain_needed': False, 'reason': f'Error: {e}'}
    
    def automated_retrain(self) -> bool:
        """Perform automated model retraining"""
        
        try:
            self.logger.info("Starting automated model retraining...")
            
            # Check if retraining is actually needed
            performance_check = self.monitor_model_performance()
            if not performance_check['retrain_needed']:
                self.logger.info("Retraining not needed at this time")
                return False
            
            # Collect latest data
            from data_collection.data_collector import DataCollector
            data_collector = DataCollector(self.config)
            latest_data = data_collector.collect_latest_data(hours=168)  # 1 week
            
            if latest_data is None:
                self.logger.error("Failed to collect latest data for retraining")
                return False
            
            # Retrain models
            retrain_success = self._retrain_models(latest_data)
            
            if retrain_success:
                self.logger.info("Automated retraining completed successfully")
                self._update_model_metadata()
                return True
            else:
                self.logger.error("Automated retraining failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in automated retraining: {e}")
            return False
    
    def schedule_retraining(self):
        """Schedule automated retraining"""
        
        try:
            retrain_frequency = self.config.get('training', {}).get('retrain_frequency', 'weekly')
            
            if retrain_frequency == 'daily':
                schedule.every().day.at("02:00").do(self.automated_retrain)
            elif retrain_frequency == 'weekly':
                schedule.every().sunday.at("02:00").do(self.automated_retrain)
            elif retrain_frequency == 'monthly':
                schedule.every().month.do(self.automated_retrain)
            
            self.logger.info(f"Retraining scheduled: {retrain_frequency}")
            
            # Run scheduler
            while True:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour
                
        except Exception as e:
            self.logger.error(f"Error in scheduling retraining: {e}")
    
    def update_features(self) -> bool:
        """Update and improve feature engineering"""
        
        try:
            self.logger.info("Updating feature engineering...")
            
            # Analyze feature importance from recent models
            feature_importance = self._analyze_feature_importance()
            
            # Identify underperforming features
            underperforming_features = self._identify_underperforming_features(feature_importance)
            
            # Generate new feature candidates
            new_features = self._generate_new_features()
            
            # Test new features
            if new_features:
                feature_performance = self._test_new_features(new_features)
                
                # Update feature selection if improvements found
                if feature_performance['improvement_found']:
                    self._update_feature_selection(feature_performance['best_features'])
                    self.logger.info("Feature engineering updated successfully")
                    return True
            
            self.logger.info("No feature improvements found")
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating features: {e}")
            return False
    
    def _load_recent_performance(self) -> Optional[pd.DataFrame]:
        """Load recent model performance data"""
        
        try:
            # Look for recent prediction files
            predictions_dir = Path("data/predictions")
            if not predictions_dir.exists():
                return None
            
            # Get most recent prediction file
            prediction_files = list(predictions_dir.glob("*.csv"))
            if not prediction_files:
                return None
            
            latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
            
            # Load predictions
            predictions = pd.read_csv(latest_file)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error loading recent performance: {e}")
            return None
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from prediction data"""
        
        try:
            if 'actual' not in data.columns or 'predicted' not in data.columns:
                return {}
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(data['actual'], data['predicted']),
                'precision': precision_score(data['actual'], data['predicted'], zero_division=0),
                'recall': recall_score(data['actual'], data['predicted'], zero_division=0),
                'f1_score': f1_score(data['actual'], data['predicted'], zero_division=0)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _should_retrain(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Determine if model retraining is needed"""
        
        try:
            if not current_metrics:
                return {'needed': False, 'reason': 'No metrics available'}
            
            # Load historical performance
            if not self.performance_history:
                return {'needed': False, 'reason': 'No historical data for comparison'}
            
            # Get recent average performance
            recent_performance = np.mean([p['accuracy'] for p in self.performance_history[-10:]])
            current_accuracy = current_metrics.get('accuracy', 0)
            
            # Check if performance has degraded
            performance_drop = recent_performance - current_accuracy
            
            if performance_drop > self.retrain_threshold:
                return {
                    'needed': True, 
                    'reason': f'Performance dropped by {performance_drop:.3f} (threshold: {self.retrain_threshold})'
                }
            
            return {'needed': False, 'reason': 'Performance within acceptable range'}
            
        except Exception as e:
            self.logger.error(f"Error determining retrain need: {e}")
            return {'needed': False, 'reason': f'Error: {e}'}
    
    def _retrain_models(self, data: pd.DataFrame) -> bool:
        """Retrain models with new data"""
        
        try:
            # Import training modules
            from training.training_strategy import TrainingStrategy
            from models.xgboost_model import XGBoostModel
            
            # Prepare data for training
            trainer = TrainingStrategy(self.config)
            splits = trainer.prepare_time_series_splits()
            
            if not splits:
                self.logger.error("Failed to prepare training splits")
                return False
            
            # Retrain XGBoost model
            xgb_model = XGBoostModel(self.config)
            training_results = xgb_model.train_model()
            
            if training_results and training_results.get('accuracy', 0) > 0.5:
                self.logger.info("Model retraining successful")
                return True
            else:
                self.logger.warning("Retrained model performance is poor")
                return False
                
        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")
            return False
    
    def _log_performance(self, metrics: Dict[str, float], retrain_decision: Dict[str, Any]):
        """Log performance metrics and decisions"""
        
        try:
            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'retrain_decision': retrain_decision
            }
            
            self.performance_history.append(performance_entry)
            
            # Save to file
            log_file = self.improvement_dir / "performance_history.json"
            with open(log_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging performance: {e}")
    
    def _analyze_feature_importance(self) -> Dict[str, float]:
        """Analyze feature importance from recent models"""
        
        try:
            # Load latest model
            model_file = self.models_dir / "xgboost_model.pkl"
            if not model_file.exists():
                return {}
            
            model = joblib.load(model_file)
            
            if hasattr(model, 'feature_importances_'):
                # Get feature names (this would need to be stored with the model)
                feature_names = getattr(model, 'feature_names_', [f'feature_{i}' for i in range(len(model.feature_importances_))])
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                return importance_dict
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {e}")
            return {}
    
    def _identify_underperforming_features(self, feature_importance: Dict[str, float]) -> list:
        """Identify features with low importance"""
        
        if not feature_importance:
            return []
        
        # Features with importance below 1% of max importance
        max_importance = max(feature_importance.values())
        threshold = max_importance * 0.01
        
        underperforming = [
            feature for feature, importance in feature_importance.items()
            if importance < threshold
        ]
        
        return underperforming
    
    def _generate_new_features(self) -> list:
        """Generate new feature candidates"""
        
        # This would implement logic to generate new features
        # For now, return empty list
        return []
    
    def _test_new_features(self, new_features: list) -> Dict[str, Any]:
        """Test new features for performance improvement"""
        
        # This would implement feature testing logic
        return {'improvement_found': False, 'best_features': []}
    
    def _update_feature_selection(self, best_features: list):
        """Update feature selection with best performing features"""
        
        # This would update the feature selection configuration
        pass
    
    def _update_model_metadata(self):
        """Update model metadata after retraining"""
        
        try:
            metadata = {
                'last_retrain': datetime.now().isoformat(),
                'retrain_reason': 'Automated performance monitoring',
                'version': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            metadata_file = self.models_dir / "model_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error updating model metadata: {e}")
