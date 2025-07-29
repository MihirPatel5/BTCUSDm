"""
Hyperparameter Tuning Module
Implements hyperparameter optimization for BTCUSD prediction models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Import model classes
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel

class HyperparameterTuner:
    """Tuner for optimizing model hyperparameters"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/processed/with_targets")
        self.tuning_dir = Path("models/tuning")
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        
    def tune_xgboost(self) -> dict:
        """Tune XGBoost hyperparameters using grid search and Bayesian optimization"""
        
        try:
            self.logger.info("Tuning XGBoost hyperparameters")
            
            # Load data
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
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Split data using time-based split
            split_ratio = self.config.get('model', {}).get('train_test_split', 0.8)
            split_index = int(len(data) * split_ratio)
            
            X_train = X.iloc[:split_index]
            y_train = y.iloc[:split_index]
            
            # Get tuning parameters from config
            tuning_config = self.config.get('tuning', {}).get('xgboost', {})
            
            # Method 1: Grid Search
            if tuning_config.get('method', 'optuna') == 'grid':
                results = self._grid_search_xgboost(X_train, y_train, tuning_config)
            
            # Method 2: Bayesian Optimization (Optuna)
            else:
                results = self._bayesian_optimization_xgboost(X_train, y_train, tuning_config)
            
            # Save tuning results
            results_file = self.tuning_dir / "xgboost_tuning_results.pkl"
            joblib.dump(results, results_file)
            
            self.logger.info(f"XGBoost tuning results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error tuning XGBoost: {e}")
            return {}
    
    def _grid_search_xgboost(self, X, y, tuning_config: dict) -> dict:
        """Perform grid search for XGBoost hyperparameters"""
        
        try:
            self.logger.info("Performing grid search for XGBoost")
            
            # Define parameter grid
            param_grid = tuning_config.get('param_grid', {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            })
            
            # Create time series cross-validator
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Define scoring metrics
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score, zero_division=0),
                'recall': make_scorer(recall_score, zero_division=0),
                'f1': make_scorer(f1_score, zero_division=0)
            }
            
            # Create XGBoost classifier
            xgb_model = xgb.XGBClassifier(random_state=42)
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='f1',  # Primary metric
                cv=tscv,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            self.logger.info(f"Grid search best parameters: {results['best_params']}")
            self.logger.info(f"Grid search best F1 score: {results['best_score']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in XGBoost grid search: {e}")
            return {}
    
    def _bayesian_optimization_xgboost(self, X, y, tuning_config: dict) -> dict:
        """Perform Bayesian optimization for XGBoost hyperparameters using Optuna"""
        
        try:
            self.logger.info("Performing Bayesian optimization for XGBoost")
            
            # Create time series cross-validator
            tscv = TimeSeriesSplit(n_splits=5)
            
            def objective(trial):
                # Define hyperparameter search space
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
                }
                
                # Create model
                model = xgb.XGBClassifier(**params, random_state=42)
                
                # Perform cross-validation
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train_fold, y_train_fold, verbose=False)
                    y_pred = model.predict(X_val_fold)
                    score = f1_score(y_val_fold, y_pred, zero_division=0)
                    scores.append(score)
                
                return np.mean(scores)
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            
            # Get number of trials from config or default
            n_trials = tuning_config.get('n_trials', 50)
            
            # Optimize
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            results = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'study': study
            }
            
            self.logger.info(f"Bayesian optimization best parameters: {results['best_params']}")
            self.logger.info(f"Bayesian optimization best F1 score: {results['best_value']:.4f}")
            
            # Save study
            study_file = self.tuning_dir / "xgboost_optuna_study.pkl"
            joblib.dump(study, study_file)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in XGBoost Bayesian optimization: {e}")
            return {}
    
    def tune_lstm(self) -> dict:
        """Tune LSTM hyperparameters using Bayesian optimization"""
        
        try:
            self.logger.info("Tuning LSTM hyperparameters")
            
            # Load data
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
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data using time-based split
            split_ratio = self.config.get('model', {}).get('train_test_split', 0.8)
            split_index = int(len(data) * split_ratio)
            
            X_train_scaled = X_scaled[:split_index]
            y_train = y.iloc[:split_index]
            
            # Reshape data for LSTM
            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            
            # Get tuning parameters from config
            tuning_config = self.config.get('tuning', {}).get('lstm', {})
            
            # Perform Bayesian optimization
            results = self._bayesian_optimization_lstm(X_train_lstm, y_train, tuning_config)
            
            # Save tuning results
            results_file = self.tuning_dir / "lstm_tuning_results.pkl"
            joblib.dump(results, results_file)
            
            self.logger.info(f"LSTM tuning results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error tuning LSTM: {e}")
            return {}
    
    def _bayesian_optimization_lstm(self, X, y, tuning_config: dict) -> dict:
        """Perform Bayesian optimization for LSTM hyperparameters using Optuna"""
        
        try:
            self.logger.info("Performing Bayesian optimization for LSTM")
            
            # Create time series cross-validator
            tscv = TimeSeriesSplit(n_splits=3)  # Fewer splits for computational efficiency
            
            def create_lstm_model(params):
                """Create LSTM model with given parameters"""
                model = Sequential()
                
                # First LSTM layer
                model.add(LSTM(
                    units=params['lstm_units_1'],
                    return_sequences=True,
                    input_shape=(X.shape[1], X.shape[2])
                ))
                model.add(Dropout(params['dropout_1']))
                
                # Second LSTM layer
                model.add(LSTM(
                    units=params['lstm_units_2'],
                    return_sequences=False
                ))
                model.add(Dropout(params['dropout_2']))
                
                # Dense layers
                model.add(Dense(
                    units=params['dense_units'],
                    activation='relu'
                ))
                
                # Output layer
                model.add(Dense(
                    units=1,
                    activation='sigmoid'
                ))
                
                # Compile model
                model.compile(
                    optimizer=Adam(learning_rate=params['learning_rate']),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                return model
            
            def objective(trial):
                # Define hyperparameter search space
                params = {
                    'lstm_units_1': trial.suggest_int('lstm_units_1', 32, 256),
                    'lstm_units_2': trial.suggest_int('lstm_units_2', 32, 256),
                    'dropout_1': trial.suggest_float('dropout_1', 0.1, 0.5),
                    'dropout_2': trial.suggest_float('dropout_2', 0.1, 0.5),
                    'dense_units': trial.suggest_int('dense_units', 16, 128),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                }
                
                # Perform cross-validation
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Create and train model
                    model = create_lstm_model(params)
                    model.fit(
                        X_train_fold, y_train_fold,
                        epochs=10,  # Reduced epochs for faster tuning
                        batch_size=32,
                        verbose=0
                    )
                    
                    # Evaluate
                    y_pred_proba = model.predict(X_val_fold)
                    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                    score = f1_score(y_val_fold, y_pred, zero_division=0)
                    scores.append(score)
                
                return np.mean(scores)
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            
            # Get number of trials from config or default
            n_trials = tuning_config.get('n_trials', 30)
            
            # Optimize
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            results = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'study': study
            }
            
            self.logger.info(f"LSTM Bayesian optimization best parameters: {results['best_params']}")
            self.logger.info(f"LSTM Bayesian optimization best F1 score: {results['best_value']:.4f}")
            
            # Save study
            study_file = self.tuning_dir / "lstm_optuna_study.pkl"
            joblib.dump(study, study_file)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in LSTM Bayesian optimization: {e}")
            return {}
    
    def plot_optimization_history(self, study, model_name: str):
        """Plot optimization history"""
        
        try:
            import matplotlib.pyplot as plt
            
            # Plot optimization history
            fig = optuna.visualization.plot_optimization_history(study)
            fig.update_layout(title=f"{model_name} Optimization History")
            
            # Save plot
            plots_dir = Path("data/reports/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plots_dir / f"{model_name.lower()}_optimization_history.png"
            fig.write_image(str(plot_file))
            
            self.logger.info(f"{model_name} optimization history plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting {model_name} optimization history: {e}")
    
    def plot_param_importance(self, study, model_name: str):
        """Plot parameter importance"""
        
        try:
            import matplotlib.pyplot as plt
            
            # Plot parameter importance
            fig = optuna.visualization.plot_param_importances(study)
            fig.update_layout(title=f"{model_name} Parameter Importance")
            
            # Save plot
            plots_dir = Path("data/reports/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plots_dir / f"{model_name.lower()}_param_importance.png"
            fig.write_image(str(plot_file))
            
            self.logger.info(f"{model_name} parameter importance plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting {model_name} parameter importance: {e}")
