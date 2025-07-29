"""
Ensemble Model Module
Combines multiple models for improved BTCUSD prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import model classes
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel

class EnsembleModel:
    """Ensemble classifier combining multiple models for BTCUSD price prediction"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.data_dir = Path("data/processed/with_targets")
        self.models_dir = Path("models/ensemble")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def train_models(self) -> dict:
        """Train all individual models and create ensemble"""
        
        try:
            results = {}
            
            # Train XGBoost model
            self.logger.info("Training XGBoost model...")
            xgb_model = XGBoostModel(self.config)
            xgb_results = xgb_model.train_model()
            self.models['xgboost'] = xgb_model
            results['xgboost'] = xgb_results
            
            # Train LSTM model
            self.logger.info("Training LSTM model...")
            lstm_model = LSTMModel(self.config)
            lstm_results = lstm_model.train_model()
            self.models['lstm'] = lstm_model
            results['lstm'] = lstm_results
            
            # Create ensemble predictions
            ensemble_results = self._create_ensemble_predictions()
            results['ensemble'] = ensemble_results
            
            self.logger.info("All models trained and ensemble created")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training ensemble models: {e}")
            return {}
    
    def _create_ensemble_predictions(self) -> dict:
        """Create ensemble predictions by combining individual model predictions"""
        
        try:
            # Load test data
            data_file = self.data_dir / "BTCUSD_5min_with_targets.csv"
            if not data_file.exists():
                self.logger.error(f"Training data file not found: {data_file}")
                return {}
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                               'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
            
            X = data[feature_columns]
            y = data['target_binary']
            
            # Split data using time-based split
            split_ratio = self.config.get('model', {}).get('train_test_split', 0.8)
            split_index = int(len(data) * split_ratio)
            
            X_test = X.iloc[split_index:]
            y_test = y.iloc[split_index:]
            timestamps_test = data['timestamp'].iloc[split_index:].reset_index(drop=True)
            
            # Get predictions from individual models
            predictions = {}
            probabilities = {}
            
            # XGBoost predictions
            if 'xgboost' in self.models:
                xgb_pred, xgb_proba = self.models['xgboost'].predict(X_test)
                predictions['xgboost'] = xgb_pred
                probabilities['xgboost'] = xgb_proba
            
            # LSTM predictions
            if 'lstm' in self.models:
                lstm_pred, lstm_proba = self.models['lstm'].predict(X_test)
                predictions['lstm'] = lstm_pred
                probabilities['lstm'] = lstm_proba
            
            # Ensemble methods
            ensemble_results = {}
            
            # Simple average of probabilities
            if probabilities:
                prob_arrays = list(probabilities.values())
                ensemble_proba = np.mean(prob_arrays, axis=0)
                ensemble_pred = (ensemble_proba > 0.5).astype(int)
                
                # Calculate metrics
                ensemble_metrics = self._calculate_metrics(y_test, ensemble_pred, ensemble_proba)
                
                # Save ensemble predictions
                predictions_df = pd.DataFrame({
                    'timestamp': timestamps_test,
                    'actual': y_test.reset_index(drop=True),
                    'predicted': ensemble_pred,
                    'probability_up': ensemble_proba
                })
                
                # Add individual model predictions
                for model_name, preds in predictions.items():
                    predictions_df[f'{model_name}_pred'] = preds
                
                for model_name, probs in probabilities.items():
                    predictions_df[f'{model_name}_prob'] = probs
                
                predictions_path = self.models_dir / "ensemble_predictions.csv"
                predictions_df.to_csv(predictions_path, index=False)
                
                # Plot ensemble confusion matrix
                self._plot_confusion_matrix(y_test, ensemble_pred, "Ensemble")
                
                ensemble_results = {
                    'metrics': ensemble_metrics,
                    'predictions_path': str(predictions_path)
                }
                
                self.logger.info(f"Ensemble model metrics: {ensemble_metrics}")
            
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble predictions: {e}")
            return {}
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba) -> dict:
        """Calculate comprehensive model metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        return metrics
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Plot confusion matrix"""
        
        try:
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            plt.title(f'{model_name} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            # Save plot
            plots_dir = Path("data/reports/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plots_dir / f"{model_name.lower()}_confusion_matrix.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"{model_name} confusion matrix plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting {model_name} confusion matrix: {e}")
    
    def load_models(self):
        """Load pre-trained models for ensemble prediction"""
        
        try:
            # Load XGBoost model
            xgb_model = XGBoostModel(self.config)
            xgb_model.load_model()
            self.models['xgboost'] = xgb_model
            
            # Load LSTM model
            lstm_model = LSTMModel(self.config)
            lstm_model.load_model()
            self.models['lstm'] = lstm_model
            
            self.logger.info("Ensemble models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble models: {e}")
    
    def predict(self, X: pd.DataFrame) -> dict:
        """Make ensemble predictions using all loaded models"""
        
        if not self.models:
            self.logger.error("No models loaded for prediction")
            return {}
        
        try:
            predictions = {}
            probabilities = {}
            
            # Get predictions from individual models
            for model_name, model in self.models.items():
                pred, prob = model.predict(X)
                predictions[model_name] = pred
                probabilities[model_name] = prob
            
            # Ensemble prediction (simple average)
            if probabilities:
                prob_arrays = list(probabilities.values())
                ensemble_proba = np.mean(prob_arrays, axis=0)
                ensemble_pred = (ensemble_proba > 0.5).astype(int)
                
                return {
                    'individual_predictions': predictions,
                    'individual_probabilities': probabilities,
                    'ensemble_prediction': ensemble_pred,
                    'ensemble_probability': ensemble_proba
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error making ensemble predictions: {e}")
            return {}
