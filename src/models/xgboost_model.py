"""
XGBoost Model Module
Implements XGBoost baseline model for BTCUSD prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class XGBoostModel:
    """XGBoost classifier for BTCUSD price prediction"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.data_dir = Path("data/processed/with_targets")
        self.models_dir = Path("models/xgboost")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def train_model(self) -> dict:
        """Train XGBoost model with the configured parameters"""
        
        try:
            # Load data with targets
            data_file = self.data_dir / "BTCUSD_5min_with_targets.csv"
            if not data_file.exists():
                self.logger.error(f"Training data file not found: {data_file}")
                return {}
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Training XGBoost model on {len(data)} records")
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                               'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
            
            X = data[feature_columns]
            y = data['target_binary']  # Binary classification target
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Split data using time-based split (important for financial data)
            split_ratio = self.config.get('model', {}).get('train_test_split', 0.8)
            split_index = int(len(data) * split_ratio)
            
            X_train = X.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_train = y.iloc[:split_index]
            y_test = y.iloc[split_index:]
            
            timestamps_test = data['timestamp'].iloc[split_index:].reset_index(drop=True)
            
            self.logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
            
            # Get model parameters from config
            model_params = self.config.get('model', {}).get('xgboost', {})
            
            # Create and train model
            self.model = xgb.XGBClassifier(**model_params)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Save model
            model_path = self.models_dir / "xgboost_model.pkl"
            joblib.dump(self.model, model_path)
            
            # Save predictions for further analysis
            predictions_df = pd.DataFrame({
                'timestamp': timestamps_test,
                'actual': y_test.reset_index(drop=True),
                'predicted': y_pred,
                'probability_up': y_pred_proba
            })
            
            predictions_path = self.models_dir / "xgboost_predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            
            # Plot feature importance
            self._plot_feature_importance(feature_columns)
            
            # Plot confusion matrix
            self._plot_confusion_matrix(y_test, y_pred)
            
            self.logger.info(f"XGBoost model trained and saved to: {model_path}")
            self.logger.info(f"Model metrics: {metrics}")
            
            return {
                'model_path': str(model_path),
                'metrics': metrics,
                'predictions_path': str(predictions_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {e}")
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
    
    def _plot_feature_importance(self, feature_names: list):
        """Plot XGBoost feature importance"""
        
        try:
            if self.model is None:
                return
            
            # Get feature importance
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(30)
            
            # Plot
            plt.figure(figsize=(12, 10))
            sns.barplot(data=feature_importance, y='feature', x='importance')
            plt.title('XGBoost Feature Importance (Top 30)')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            # Save plot
            plots_dir = Path("data/reports/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plots_dir / "xgboost_feature_importance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Feature importance plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {e}")
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        
        try:
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            # Save plot
            plots_dir = Path("data/reports/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plots_dir / "xgboost_confusion_matrix.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confusion matrix plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {e}")
    
    def load_model(self, model_path: str = None):
        """Load a pre-trained XGBoost model"""
        
        try:
            if model_path is None:
                model_path = self.models_dir / "xgboost_model.pkl"
            
            self.model = joblib.load(model_path)
            self.logger.info(f"XGBoost model loaded from: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.array:
        """Make predictions using the trained model"""
        
        if self.model is None:
            self.logger.error("Model not trained or loaded")
            return None
        
        try:
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            
            return predictions, probabilities
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return None, None
