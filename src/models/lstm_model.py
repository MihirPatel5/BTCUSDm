"""
LSTM Model Module
Implements LSTM neural network model for BTCUSD prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class LSTMModel:
    """LSTM neural network for BTCUSD price prediction"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.data_dir = Path("data/processed/with_targets")
        self.models_dir = Path("models/lstm")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def train_model(self) -> dict:
        """Train LSTM model with the configured parameters"""
        
        try:
            # Load data with targets
            data_file = self.data_dir / "BTCUSD_5min_with_targets.csv"
            if not data_file.exists():
                self.logger.error(f"Training data file not found: {data_file}")
                return {}
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Training LSTM model on {len(data)} records")
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                               'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
            
            X = data[feature_columns]
            y = data['target_binary']  # Binary classification target
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data using time-based split
            split_ratio = self.config.get('model', {}).get('train_test_split', 0.8)
            split_index = int(len(data) * split_ratio)
            
            X_train_scaled = X_scaled[:split_index]
            X_test_scaled = X_scaled[split_index:]
            y_train = y.iloc[:split_index]
            y_test = y.iloc[split_index:]
            
            timestamps_test = data['timestamp'].iloc[split_index:].reset_index(drop=True)
            
            # Reshape data for LSTM (samples, timesteps, features)
            # For simplicity, we'll use a single timestep per sample
            # In a more advanced implementation, we could use sequences
            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            
            self.logger.info(f"Training set: {len(X_train_lstm)} samples, Test set: {len(X_test_lstm)} samples")
            
            # Create LSTM model
            self.model = self._create_lstm_model(X_train_lstm.shape[2])
            
            # Get training parameters from config
            epochs = self.config.get('model', {}).get('lstm', {}).get('epochs', 50)
            batch_size = self.config.get('model', {}).get('lstm', {}).get('batch_size', 32)
            validation_split = self.config.get('model', {}).get('lstm', {}).get('validation_split', 0.1)
            
            # Train model
            history = self.model.fit(
                X_train_lstm, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            
            # Make predictions
            y_pred_proba = self.model.predict(X_test_lstm)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_pred_proba = y_pred_proba.flatten()
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Save model and scaler
            model_path = self.models_dir / "lstm_model.h5"
            self.model.save(model_path)
            
            scaler_path = self.models_dir / "lstm_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            
            # Save predictions for further analysis
            predictions_df = pd.DataFrame({
                'timestamp': timestamps_test,
                'actual': y_test.reset_index(drop=True),
                'predicted': y_pred,
                'probability_up': y_pred_proba
            })
            
            predictions_path = self.models_dir / "lstm_predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            
            # Plot training history
            self._plot_training_history(history)
            
            # Plot confusion matrix
            self._plot_confusion_matrix(y_test, y_pred)
            
            self.logger.info(f"LSTM model trained and saved to: {model_path}")
            self.logger.info(f"Model metrics: {metrics}")
            
            return {
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'metrics': metrics,
                'predictions_path': str(predictions_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")
            return {}
    
    def _create_lstm_model(self, n_features: int) -> tf.keras.Model:
        """Create LSTM model architecture"""
        
        # Get model parameters from config
        lstm_params = self.config.get('model', {}).get('lstm', {})
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=lstm_params.get('lstm_units_1', 50),
            return_sequences=True,
            input_shape=(1, n_features)
        ))
        model.add(Dropout(lstm_params.get('dropout_1', 0.2)))
        
        # Second LSTM layer
        model.add(LSTM(
            units=lstm_params.get('lstm_units_2', 50),
            return_sequences=False
        ))
        model.add(Dropout(lstm_params.get('dropout_2', 0.2)))
        
        # Dense layers
        model.add(Dense(
            units=lstm_params.get('dense_units', 25),
            activation='relu'
        ))
        
        # Output layer
        model.add(Dense(
            units=1,
            activation='sigmoid'
        ))
        
        # Compile model
        learning_rate = lstm_params.get('learning_rate', 0.001)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
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
    
    def _plot_training_history(self, history):
        """Plot training history (loss and accuracy)"""
        
        try:
            # Plot training & validation loss
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            
            # Save plot
            plots_dir = Path("data/reports/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plots_dir / "lstm_training_history.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Training history plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting training history: {e}")
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        
        try:
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            plt.title('LSTM Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            # Save plot
            plots_dir = Path("data/reports/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plots_dir / "lstm_confusion_matrix.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confusion matrix plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {e}")
    
    def load_model(self, model_path: str = None, scaler_path: str = None):
        """Load a pre-trained LSTM model and scaler"""
        
        try:
            if model_path is None:
                model_path = self.models_dir / "lstm_model.h5"
            
            if scaler_path is None:
                scaler_path = self.models_dir / "lstm_scaler.pkl"
            
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            
            self.logger.info(f"LSTM model loaded from: {model_path}")
            self.logger.info(f"Scaler loaded from: {scaler_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.array:
        """Make predictions using the trained model"""
        
        if self.model is None or self.scaler is None:
            self.logger.error("Model or scaler not trained or loaded")
            return None, None
        
        try:
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Reshape for LSTM
            X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            
            # Make predictions
            probabilities = self.model.predict(X_lstm).flatten()
            predictions = (probabilities > 0.5).astype(int)
            
            return predictions, probabilities
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return None, None
