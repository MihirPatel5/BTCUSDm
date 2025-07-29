"""
Training Strategy Module
Implements advanced training strategies for BTCUSD prediction models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingStrategy:
    """Advanced training strategies for financial time series models"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/processed/with_targets")
        
    def prepare_time_series_splits(self, n_splits: int = 5) -> list:
        """Create time-based train/validation splits for walk-forward analysis"""
        
        try:
            # Load data with targets
            data_file = self.data_dir / "BTCUSD_5min_with_targets.csv"
            if not data_file.exists():
                self.logger.error(f"Training data file not found: {data_file}")
                return []
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Preparing time series splits for {len(data)} records")
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                               'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
            
            X = data[feature_columns]
            y = data['target_binary']
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Create time series splits
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = []
            
            for i, (train_index, val_index) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                
                timestamps_train = data['timestamp'].iloc[train_index].reset_index(drop=True)
                timestamps_val = data['timestamp'].iloc[val_index].reset_index(drop=True)
                
                splits.append({
                    'split': i+1,
                    'X_train': X_train,
                    'X_val': X_val,
                    'y_train': y_train,
                    'y_val': y_val,
                    'timestamps_train': timestamps_train,
                    'timestamps_val': timestamps_val
                })
                
                self.logger.info(f"Split {i+1}: Train={len(train_index)}, Val={len(val_index)}")
            
            return splits
            
        except Exception as e:
            self.logger.error(f"Error preparing time series splits: {e}")
            return []
    
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
    
    def walk_forward_validation(self, model, splits: list) -> dict:
        """Perform walk-forward validation on time series splits"""
        
        try:
            self.logger.info("Performing walk-forward validation")
            
            results = []
            all_predictions = []
            
            for split_data in splits:
                split_num = split_data['split']
                X_train = split_data['X_train']
                X_val = split_data['X_val']
                y_train = split_data['y_train']
                y_val = split_data['y_val']
                timestamps_val = split_data['timestamps_val']
                
                self.logger.info(f"Training on split {split_num}...")
                
                # Handle class imbalance if configured
                imbalance_method = self.config.get('training', {}).get('class_imbalance_method', 'none')
                if imbalance_method != 'none':
                    X_train, y_train = self.handle_class_imbalance(X_train, y_train, imbalance_method)
                
                # Train model (assuming it has a fit method)
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                
                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_val)
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_val)[:, 1]
                    else:
                        y_pred_proba = y_pred
                else:
                    # For models without predict method, skip
                    continue
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                metrics = {
                    'split': split_num,
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, zero_division=0),
                    'recall': recall_score(y_val, y_pred, zero_division=0),
                    'f1_score': f1_score(y_val, y_pred, zero_division=0)
                }
                
                results.append(metrics)
                
                # Store predictions
                pred_df = pd.DataFrame({
                    'timestamp': timestamps_val,
                    'actual': y_val.reset_index(drop=True),
                    'predicted': y_pred,
                    'probability': y_pred_proba
                })
                pred_df['split'] = split_num
                all_predictions.append(pred_df)
                
                self.logger.info(f"Split {split_num} metrics: {metrics}")
            
            # Combine all predictions
            if all_predictions:
                all_preds_df = pd.concat(all_predictions, ignore_index=True)
                
                # Save predictions
                training_dir = Path("data/training")
                training_dir.mkdir(parents=True, exist_ok=True)
                predictions_file = training_dir / "walk_forward_predictions.csv"
                all_preds_df.to_csv(predictions_file, index=False)
                
                self.logger.info(f"Walk-forward predictions saved to: {predictions_file}")
            
            # Calculate overall metrics
            if results:
                df_results = pd.DataFrame(results)
                overall_metrics = df_results.mean(numeric_only=True).to_dict()
                
                self.logger.info(f"Overall walk-forward metrics: {overall_metrics}")
                
                return {
                    'split_metrics': results,
                    'overall_metrics': overall_metrics,
                    'predictions_file': str(predictions_file) if all_predictions else None
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward validation: {e}")
            return {}
    
    def plot_walk_forward_results(self, results: dict):
        """Plot walk-forward validation results"""
        
        try:
            if not results or 'split_metrics' not in results:
                self.logger.warning("No results to plot")
                return
            
            df_metrics = pd.DataFrame(results['split_metrics'])
            
            # Plot metrics across splits
            plt.figure(figsize=(12, 8))
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for i, metric in enumerate(metrics_to_plot, 1):
                plt.subplot(2, 2, i)
                plt.plot(df_metrics['split'], df_metrics[metric], marker='o')
                plt.title(f'{metric.capitalize()} Across Splits')
                plt.xlabel('Split')
                plt.ylabel(metric.capitalize())
                plt.grid(True)
                
                # Add value labels
                for x, y in zip(df_metrics['split'], df_metrics[metric]):
                    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
            
            # Save plot
            plots_dir = Path("data/reports/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plots_dir / "walk_forward_validation.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Walk-forward validation plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting walk-forward results: {e}")
    
    def analyze_temporal_stability(self, splits: list) -> dict:
        """Analyze temporal stability of features and targets"""
        
        try:
            self.logger.info("Analyzing temporal stability")
            
            stability_results = {}
            
            for split_data in splits:
                split_num = split_data['split']
                X_train = split_data['X_train']
                X_val = split_data['X_val']
                y_train = split_data['y_train']
                y_val = split_data['y_val']
                
                # Compare feature distributions
                feature_stability = {}
                for col in X_train.columns:
                    if col in X_val.columns:
                        # Calculate mean difference
                        mean_diff = abs(X_train[col].mean() - X_val[col].mean())
                        std_diff = abs(X_train[col].std() - X_val[col].std())
                        
                        feature_stability[col] = {
                            'mean_diff': mean_diff,
                            'std_diff': std_diff
                        }
                
                # Compare target distributions
                target_stability = {
                    'train_dist': y_train.value_counts().to_dict(),
                    'val_dist': y_val.value_counts().to_dict()
                }
                
                stability_results[f'split_{split_num}'] = {
                    'feature_stability': feature_stability,
                    'target_stability': target_stability
                }
            
            # Save stability analysis
            training_dir = Path("data/training")
            training_dir.mkdir(parents=True, exist_ok=True)
            stability_file = training_dir / "temporal_stability_analysis.json"
            
            import json
            with open(stability_file, 'w') as f:
                json.dump(stability_results, f, indent=2)
            
            self.logger.info(f"Temporal stability analysis saved to: {stability_file}")
            
            return stability_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal stability: {e}")
            return {}


if __name__ == "__main__":
    # This allows the file to be run directly for testing
    pass
