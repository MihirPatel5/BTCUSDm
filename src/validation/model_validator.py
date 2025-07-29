"""
Model Validation Module
Implements comprehensive validation for BTCUSD prediction models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns


class ModelValidator:
    """Validator for comprehensive model robustness and validation"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/processed")
        self.validation_dir = Path("data/validation")
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_out_of_sample(self, model, oos_data_file: str) -> dict:
        """Validate model on out-of-sample data"""
        
        try:
            self.logger.info("Validating model on out-of-sample data")
            
            # Load out-of-sample data
            oos_path = Path(oos_data_file)
            if not oos_path.exists():
                self.logger.error(f"Out-of-sample data file not found: {oos_path}")
                return {}
            
            oos_data = pd.read_csv(oos_path)
            oos_data['timestamp'] = pd.to_datetime(oos_data['timestamp'])
            
            # Prepare features and target
            feature_columns = [col for col in oos_data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                               'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
            
            X_oos = oos_data[feature_columns]
            y_oos = oos_data['target_binary']
            
            # Handle missing values
            X_oos = X_oos.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Make predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_oos)
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_oos)[:, 1]
                else:
                    y_pred_proba = y_pred
            else:
                self.logger.error("Model does not have predict method")
                return {}
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_oos, y_pred),
                'precision': precision_score(y_oos, y_pred, zero_division=0),
                'recall': recall_score(y_oos, y_pred, zero_division=0),
                'f1_score': f1_score(y_oos, y_pred, zero_division=0)
            }
            
            # Save validation results
            results = {
                'validation_type': 'out_of_sample',
                'metrics': metrics,
                'sample_size': len(y_oos)
            }
            
            results_file = self.validation_dir / "oos_validation_results.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Out-of-sample validation completed. Metrics: {metrics}")
            self.logger.info(f"Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in out-of-sample validation: {e}")
            return {}
    
    def validate_temporal_stability(self, model, data_file: str, n_periods: int = 5) -> dict:
        """Validate model performance across different time periods"""
        
        try:
            self.logger.info("Validating temporal stability")
            
            # Load data
            data_path = Path(data_file)
            if not data_path.exists():
                self.logger.error(f"Data file not found: {data_path}")
                return {}
            
            data = pd.read_csv(data_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Sort by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Divide data into periods
            period_size = len(data) // n_periods
            period_results = []
            
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = min((i + 1) * period_size, len(data))
                
                period_data = data.iloc[start_idx:end_idx]
                
                # Prepare features and target
                feature_columns = [col for col in period_data.columns if col not in 
                                  ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                   'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
                
                X_period = period_data[feature_columns]
                y_period = period_data['target_binary']
                
                # Handle missing values
                X_period = X_period.fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_period)
                else:
                    continue
                
                # Calculate metrics
                period_metrics = {
                    'period': i + 1,
                    'start_date': period_data['timestamp'].iloc[0].strftime('%Y-%m-%d'),
                    'end_date': period_data['timestamp'].iloc[-1].strftime('%Y-%m-%d'),
                    'accuracy': accuracy_score(y_period, y_pred),
                    'precision': precision_score(y_period, y_pred, zero_division=0),
                    'recall': recall_score(y_period, y_pred, zero_division=0),
                    'f1_score': f1_score(y_period, y_pred, zero_division=0),
                    'sample_size': len(y_period)
                }
                
                period_results.append(period_metrics)
                
                self.logger.info(f"Period {i+1} ({period_metrics['start_date']} to {period_metrics['end_date']}): Accuracy={period_metrics['accuracy']:.4f}")
            
            # Calculate stability metrics
            accuracies = [r['accuracy'] for r in period_results]
            precision_scores = [r['precision'] for r in period_results]
            recall_scores = [r['recall'] for r in period_results]
            f1_scores = [r['f1_score'] for r in period_results]
            
            stability_metrics = {
                'accuracy_std': np.std(accuracies),
                'precision_std': np.std(precision_scores),
                'recall_std': np.std(recall_scores),
                'f1_std': np.std(f1_scores),
                'accuracy_cv': np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 0,
                'overall_metrics': {
                    'mean_accuracy': np.mean(accuracies),
                    'mean_precision': np.mean(precision_scores),
                    'mean_recall': np.mean(recall_scores),
                    'mean_f1': np.mean(f1_scores)
                }
            }
            
            results = {
                'validation_type': 'temporal_stability',
                'period_results': period_results,
                'stability_metrics': stability_metrics
            }
            
            # Save results
            results_file = self.validation_dir / "temporal_stability_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Plot temporal stability
            self._plot_temporal_stability(period_results)
            
            self.logger.info(f"Temporal stability validation completed. Stability metrics: {stability_metrics}")
            self.logger.info(f"Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in temporal stability validation: {e}")
            return {}
    
    def _plot_temporal_stability(self, period_results: list):
        """Plot temporal stability results"""
        
        try:
            df_results = pd.DataFrame(period_results)
            
            plt.figure(figsize=(12, 8))
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for i, metric in enumerate(metrics_to_plot, 1):
                plt.subplot(2, 2, i)
                plt.plot(df_results['period'], df_results[metric], marker='o')
                plt.title(f'{metric.capitalize()} Over Time')
                plt.xlabel('Period')
                plt.ylabel(metric.capitalize())
                plt.grid(True)
                
                # Add value labels
                for x, y in zip(df_results['period'], df_results[metric]):
                    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
            
            plot_file = self.validation_dir / "plots" / "temporal_stability.png"
            plot_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Temporal stability plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting temporal stability: {e}")
    
    def validate_data_drift(self, training_data_file: str, validation_data_file: str) -> dict:
        """Validate for data drift between training and validation periods"""
        
        try:
            self.logger.info("Validating data drift")
            
            # Load training data
            train_path = Path(training_data_file)
            if not train_path.exists():
                self.logger.error(f"Training data file not found: {train_path}")
                return {}
            
            train_data = pd.read_csv(train_path)
            train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
            
            # Load validation data
            val_path = Path(validation_data_file)
            if not val_path.exists():
                self.logger.error(f"Validation data file not found: {val_path}")
                return {}
            
            val_data = pd.read_csv(val_path)
            val_data['timestamp'] = pd.to_datetime(val_data['timestamp'])
            
            # Prepare features
            feature_columns = [col for col in train_data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                               'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
            
            X_train = train_data[feature_columns]
            X_val = val_data[feature_columns]
            
            # Handle missing values
            X_train = X_train.fillna(method='ffill').fillna(method='bfill').fillna(0)
            X_val = X_val.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Calculate drift metrics for each feature
            drift_results = []
            
            for col in feature_columns:
                if col in X_train.columns and col in X_val.columns:
                    # Calculate means and stds
                    train_mean = X_train[col].mean()
                    train_std = X_train[col].std()
                    val_mean = X_val[col].mean()
                    val_std = X_val[col].std()
                    
                    # Calculate drift metrics
                    mean_diff = abs(train_mean - val_mean)
                    std_diff = abs(train_std - val_std)
                    
                    # Normalized drift (as percentage of training std)
                    normalized_drift = mean_diff / train_std if train_std > 0 else 0
                    
                    drift_results.append({
                        'feature': col,
                        'train_mean': train_mean,
                        'val_mean': val_mean,
                        'mean_diff': mean_diff,
                        'normalized_drift': normalized_drift,
                        'drift_level': self._classify_drift(normalized_drift)
                    })
            
            # Calculate overall drift metrics
            drift_df = pd.DataFrame(drift_results)
            significant_drift = drift_df[drift_df['drift_level'] == 'High']
            
            overall_metrics = {
                'total_features': len(drift_results),
                'features_with_high_drift': len(significant_drift),
                'drift_percentage': len(significant_drift) / len(drift_results) if drift_results else 0,
                'mean_normalized_drift': drift_df['normalized_drift'].mean(),
                'max_normalized_drift': drift_df['normalized_drift'].max()
            }
            
            results = {
                'validation_type': 'data_drift',
                'feature_drift': drift_results,
                'overall_metrics': overall_metrics
            }
            
            # Save results
            results_file = self.validation_dir / "data_drift_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Plot drift analysis
            self._plot_drift_analysis(drift_results)
            
            self.logger.info(f"Data drift validation completed. Drift percentage: {overall_metrics['drift_percentage']:.2%}")
            self.logger.info(f"Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in data drift validation: {e}")
            return {}
    
    def _classify_drift(self, normalized_drift: float) -> str:
        """Classify drift level based on normalized drift value"""
        
        if normalized_drift > 0.5:
            return 'High'
        elif normalized_drift > 0.2:
            return 'Medium'
        else:
            return 'Low'
    
    def _plot_drift_analysis(self, drift_results: list):
        """Plot data drift analysis"""
        
        try:
            df_drift = pd.DataFrame(drift_results)
            
            # Top features with highest drift
            top_drift = df_drift.nlargest(20, 'normalized_drift')
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(top_drift)), top_drift['normalized_drift'])
            plt.title('Top Features with Highest Data Drift')
            plt.xlabel('Features')
            plt.ylabel('Normalized Drift')
            
            # Color bars based on drift level
            for i, (bar, drift_level) in enumerate(zip(bars, top_drift['drift_level'])):
                if drift_level == 'High':
                    bar.set_color('red')
                elif drift_level == 'Medium':
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
            
            plt.xticks(range(len(top_drift)), top_drift['feature'], rotation=45, ha='right')
            plt.tight_layout()
            
            plot_file = self.validation_dir / "plots" / "data_drift.png"
            plot_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Data drift plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting data drift: {e}")
    
    def validate_stress_scenarios(self, model, data_file: str) -> dict:
        """Validate model performance under stress scenarios"""
        
        try:
            self.logger.info("Validating model under stress scenarios")
            
            # Load data
            data_path = Path(data_file)
            if not data_path.exists():
                self.logger.error(f"Data file not found: {data_path}")
                return {}
            
            data = pd.read_csv(data_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Define stress scenarios
            scenarios = {
                'high_volatility': '2020-03-01',  # COVID-19 market crash
                'bull_market': '2021-01-01',      # Bull run period
                'bear_market': '2022-05-01',      # Bear market period
                'normal_market': '2021-06-01'     # Normal period
            }
            
            scenario_results = {}
            
            for scenario_name, start_date in scenarios.items():
                # Filter data for scenario period (30 days)
                end_date = pd.to_datetime(start_date) + pd.Timedelta(days=30)
                scenario_data = data[
                    (data['timestamp'] >= start_date) & 
                    (data['timestamp'] < end_date)
                ]
                
                if len(scenario_data) < 100:  # Need minimum data
                    self.logger.warning(f"Insufficient data for {scenario_name} scenario")
                    continue
                
                # Prepare features and target
                feature_columns = [col for col in scenario_data.columns if col not in 
                                  ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                   'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
                
                X_scenario = scenario_data[feature_columns]
                y_scenario = scenario_data['target_binary']
                
                # Handle missing values
                X_scenario = X_scenario.fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_scenario)
                else:
                    continue
                
                # Calculate metrics
                scenario_metrics = {
                    'accuracy': accuracy_score(y_scenario, y_pred),
                    'precision': precision_score(y_scenario, y_pred, zero_division=0),
                    'recall': recall_score(y_scenario, y_pred, zero_division=0),
                    'f1_score': f1_score(y_scenario, y_pred, zero_division=0),
                    'sample_size': len(y_scenario)
                }
                
                scenario_results[scenario_name] = scenario_metrics
                
                self.logger.info(f"{scenario_name} scenario: Accuracy={scenario_metrics['accuracy']:.4f}")
            
            results = {
                'validation_type': 'stress_scenarios',
                'scenario_results': scenario_results
            }
            
            # Save results
            results_file = self.validation_dir / "stress_scenario_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Plot stress scenario results
            self._plot_stress_scenarios(scenario_results)
            
            self.logger.info(f"Stress scenario validation completed")
            self.logger.info(f"Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in stress scenario validation: {e}")
            return {}
    
    def _plot_stress_scenarios(self, scenario_results: dict):
        """Plot stress scenario results"""
        
        try:
            df_results = pd.DataFrame(scenario_results).T
            
            plt.figure(figsize=(12, 8))
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for i, metric in enumerate(metrics_to_plot, 1):
                plt.subplot(2, 2, i)
                bars = plt.bar(df_results.index, df_results[metric])
                plt.title(f'{metric.capitalize()} by Scenario')
                plt.xlabel('Scenario')
                plt.ylabel(metric.capitalize())
                plt.xticks(rotation=45)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.annotate(f'{height:.3f}', 
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
            
            plt.tight_layout()
            
            plot_file = self.validation_dir / "plots" / "stress_scenarios.png"
            plot_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Stress scenario plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting stress scenarios: {e}")
    
    def generate_validation_report(self, validation_results: dict) -> str:
        """Generate comprehensive validation report"""
        
        try:
            report = "=" * 80 + "\n"
            report += "BTCUSD MODEL VALIDATION REPORT\n"
            report += "=" * 80 + "\n\n"
            
            for validation_type, results in validation_results.items():
                report += f"{validation_type.upper()} VALIDATION\n"
                report += "-" * 40 + "\n"
                
                if validation_type == 'out_of_sample':
                    metrics = results.get('metrics', {})
                    report += f"Sample Size: {results.get('sample_size', 0)}\n"
                    report += f"Accuracy: {metrics.get('accuracy', 0):.4f}\n"
                    report += f"Precision: {metrics.get('precision', 0):.4f}\n"
                    report += f"Recall: {metrics.get('recall', 0):.4f}\n"
                    report += f"F1-Score: {metrics.get('f1_score', 0):.4f}\n\n"
                
                elif validation_type == 'temporal_stability':
                    stability = results.get('stability_metrics', {})
                    overall = stability.get('overall_metrics', {})
                    report += f"Mean Accuracy: {overall.get('mean_accuracy', 0):.4f}\n"
                    report += f"Accuracy Std Dev: {stability.get('accuracy_std', 0):.4f}\n"
                    report += f"Accuracy Coefficient of Variation: {stability.get('accuracy_cv', 0):.4f}\n\n"
                
                elif validation_type == 'data_drift':
                    overall = results.get('overall_metrics', {})
                    report += f"Total Features: {overall.get('total_features', 0)}\n"
                    report += f"Features with High Drift: {overall.get('features_with_high_drift', 0)}\n"
                    report += f"Drift Percentage: {overall.get('drift_percentage', 0):.2%}\n"
                    report += f"Mean Normalized Drift: {overall.get('mean_normalized_drift', 0):.4f}\n\n"
                
                elif validation_type == 'stress_scenarios':
                    scenarios = results.get('scenario_results', {})
                    for scenario, metrics in scenarios.items():
                        report += f"{scenario}:\n"
                        report += f"  Accuracy: {metrics.get('accuracy', 0):.4f}\n"
                        report += f"  F1-Score: {metrics.get('f1_score', 0):.4f}\n"
                    report += "\n"
            
            # Save report
            report_file = self.validation_dir / "validation_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Validation report saved to: {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating validation report: {e}")
            return ""
