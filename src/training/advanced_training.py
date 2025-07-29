import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class AdvancedTrainingStrategy:
    """Advanced training strategies with ensemble methods and improved validation"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/processed/with_targets")
    
    def prepare_advanced_splits(self, n_splits: int = 5) -> list:
        """Create advanced time-based train/validation splits with multiple validation approaches"""
        
        try:
            # Load data with targets
            data_file = self.data_dir / "BTCUSD_5min_with_targets.csv"
            if not data_file.exists():
                self.logger.error(f"Training data file not found: {data_file}")
                return []
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Preparing advanced time series splits for {len(data)} records")
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                               'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
            
            X = data[feature_columns]
            y = data['target_binary']
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Create multiple types of splits
            splits = []
            
            # 1. Standard time series splits
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for i, (train_index, val_index) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                timestamps_train = data['timestamp'].iloc[train_index].reset_index(drop=True)
                timestamps_val = data['timestamp'].iloc[val_index].reset_index(drop=True)
                
                splits.append({
                    'split_type': 'standard',
                    'split_id': f'standard_{i+1}',
                    'X_train': X_train,
                    'X_val': X_val,
                    'y_train': y_train,
                    'y_val': y_val,
                    'timestamps_train': timestamps_train,
                    'timestamps_val': timestamps_val
                })
            
            # 2. Walk-forward expanding window splits
            for i in range(n_splits):
                # Expanding window: use more data for training as we go forward
                train_size = int(len(X) * 0.6) + int((len(X) * 0.2) * i / (n_splits - 1)) if n_splits > 1 else int(len(X) * 0.6)
                val_start = train_size
                val_end = min(train_size + int(len(X) * 0.2), len(X))
                
                if val_end > val_start:
                    X_train = X.iloc[:val_start]
                    X_val = X.iloc[val_start:val_end]
                    y_train = y.iloc[:val_start]
                    y_val = y.iloc[val_start:val_end]
                    timestamps_train = data['timestamp'].iloc[:val_start].reset_index(drop=True)
                    timestamps_val = data['timestamp'].iloc[val_start:val_end].reset_index(drop=True)
                    
                    splits.append({
                        'split_type': 'expanding',
                        'split_id': f'expanding_{i+1}',
                        'X_train': X_train,
                        'X_val': X_val,
                        'y_train': y_train,
                        'y_val': y_val,
                        'timestamps_train': timestamps_train,
                        'timestamps_val': timestamps_val
                    })
            
            self.logger.info(f"Created {len(splits)} advanced splits")
            return splits
            
        except Exception as e:
            self.logger.error(f"Error preparing advanced time series splits: {e}")
            return []
    
    def train_advanced_ensemble(self, splits: list) -> dict:
        """Train an advanced ensemble of multiple models"""
        
        try:
            self.logger.info("Training advanced ensemble models")
            
            # Define models
            models = {
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'svm': SVC(
                    probability=True,
                    kernel='rbf',
                    C=1.0,
                    random_state=42
                )
            }
            
            # Train models on each split and collect results
            model_results = {name: [] for name in models.keys()}
            ensemble_results = []
            
            for split_data in splits:
                split_id = split_data['split_id']
                X_train = split_data['X_train']
                X_val = split_data['X_val']
                y_train = split_data['y_train']
                y_val = split_data['y_val']
                
                self.logger.info(f"Training models on split {split_id}...")
                
                # Handle class imbalance if configured
                imbalance_method = self.config.get('training', {}).get('class_imbalance_method', 'none')
                if imbalance_method != 'none':
                    X_train, y_train = self.handle_class_imbalance(X_train, y_train, imbalance_method)
                
                # Train each model and collect predictions
                split_predictions = {}
                
                for model_name, model in models.items():
                    try:
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_val)
                        y_pred_proba = model.predict_proba(X_val)[:, 1]
                        
                        # Calculate metrics
                        metrics = {
                            'split_id': split_id,
                            'model': model_name,
                            'accuracy': accuracy_score(y_val, y_pred),
                            'precision': precision_score(y_val, y_pred, zero_division=0),
                            'recall': recall_score(y_val, y_pred, zero_division=0),
                            'f1_score': f1_score(y_val, y_pred, zero_division=0),
                            'roc_auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.0
                        }
                        
                        model_results[model_name].append(metrics)
                        split_predictions[model_name] = y_pred_proba
                        
                    except Exception as e:
                        self.logger.error(f"Error training {model_name} on split {split_id}: {e}")
                
                # Create ensemble prediction (average of probabilities)
                if split_predictions:
                    ensemble_proba = np.mean(list(split_predictions.values()), axis=0)
                    ensemble_pred = (ensemble_proba > 0.5).astype(int)
                    
                    # Calculate ensemble metrics
                    ensemble_metrics = {
                        'split_id': split_id,
                        'model': 'ensemble',
                        'accuracy': accuracy_score(y_val, ensemble_pred),
                        'precision': precision_score(y_val, ensemble_pred, zero_division=0),
                        'recall': recall_score(y_val, ensemble_pred, zero_division=0),
                        'f1_score': f1_score(y_val, ensemble_pred, zero_division=0),
                        'roc_auc': roc_auc_score(y_val, ensemble_proba) if len(np.unique(y_val)) > 1 else 0.0
                    }
                    
                    ensemble_results.append(ensemble_metrics)
            
            # Calculate overall metrics for each model
            overall_results = {}
            for model_name, results in model_results.items():
                if results:
                    df_results = pd.DataFrame(results)
                    overall_metrics = df_results.mean(numeric_only=True).to_dict()
                    overall_results[model_name] = {
                        'split_metrics': results,
                        'overall_metrics': overall_metrics
                    }
            
            # Add ensemble results
            if ensemble_results:
                df_ensemble = pd.DataFrame(ensemble_results)
                ensemble_overall = df_ensemble.mean(numeric_only=True).to_dict()
                overall_results['ensemble'] = {
                    'split_metrics': ensemble_results,
                    'overall_metrics': ensemble_overall
                }
            
            self.logger.info("Advanced ensemble training completed")
            return overall_results
            
        except Exception as e:
            self.logger.error(f"Error in advanced ensemble training: {e}")
            return {}
    
    def handle_class_imbalance(self, X, y, method: str = 'smote'):
        """Handle class imbalance in target variable"""
        
        try:
            self.logger.info(f"Handling class imbalance using {method} method")
            self.logger.info(f"Original class distribution: {y.value_counts().to_dict()}")
            
            if method == 'smote':
                # Apply SMOTE oversampling
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
            elif method == 'undersample':
                # Apply random undersampling
                undersampler = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = undersampler.fit_resample(X, y)
                
            elif method == 'oversample':
                # Apply random oversampling
                # Separate majority and minority classes
                df = pd.concat([X, y], axis=1)
                majority_class = df[y == y.mode()[0]]
                minority_class = df[y != y.mode()[0]]
                
                # Upsample minority class
                minority_upsampled = resample(minority_class, 
                                            replace=True,     # sample with replacement
                                            n_samples=len(majority_class),    # match majority class
                                            random_state=42) # reproducible results
                
                # Combine majority class with upsampled minority class
                df_resampled = pd.concat([majority_class, minority_upsampled])
                
                # Separate features and target
                X_resampled = df_resampled.drop(columns=[y.name])
                y_resampled = df_resampled[y.name]
                
            else:
                # No resampling
                X_resampled, y_resampled = X, y
                
            self.logger.info(f"Resampled class distribution: {y_resampled.value_counts().to_dict()}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.error(f"Error handling class imbalance: {e}")
            return X, y
    
    def plot_model_comparison(self, results: dict):
        """Plot comparison of different models"""
        
        try:
            if not results:
                self.logger.warning("No results to plot")
                return
            
            # Prepare data for plotting
            model_names = []
            metrics_data = []
            
            for model_name, model_results in results.items():
                if 'overall_metrics' in model_results:
                    model_names.append(model_name)
                    metrics_data.append(model_results['overall_metrics'])
            
            if not model_names:
                return
            
            df_metrics = pd.DataFrame(metrics_data, index=model_names)
            
            # Plot metrics comparison
            plt.figure(figsize=(15, 10))
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            
            for i, metric in enumerate(metrics_to_plot, 1):
                plt.subplot(2, 3, i)
                values = [df_metrics.loc[model, metric] for model in model_names]
                bars = plt.bar(model_names, values)
                plt.title(f'{metric.capitalize()} Comparison')
                plt.ylabel(metric.capitalize())
                plt.xticks(rotation=45)
                plt.grid(True, axis='y')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            training_dir = Path("data/training")
            training_dir.mkdir(parents=True, exist_ok=True)
            plot_file = training_dir / "model_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Model comparison plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting model comparison: {e}")
