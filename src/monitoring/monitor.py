"""
Monitoring Module
Implements real-time monitoring for BTCUSD prediction models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import time
from datetime import datetime
import threading
import matplotlib.pyplot as plt
import seaborn as sns


class ModelMonitor:
    """Monitor for deployed BTCUSD prediction models"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.deployment_dir = Path("deployment")
        self.monitoring_dir = Path("monitoring")
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_history = []
        self.prediction_history = []
        
        # Load initial performance data
        self._load_performance_data()
        
    def _load_performance_data(self):
        """Load existing performance data"""
        
        try:
            # Load performance log if exists
            log_file = self.deployment_dir / "performance_log.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    self.performance_history = json.load(f)
                self.logger.info(f"Loaded {len(self.performance_history)} performance records")
            
            # Load recent predictions if exists
            pred_files = list(self.deployment_dir.glob("prediction_*.json"))
            if pred_files:
                # Sort by timestamp and get most recent
                pred_files.sort(key=lambda x: x.name, reverse=True)
                recent_files = pred_files[:50]  # Last 50 predictions
                
                for pred_file in recent_files:
                    try:
                        with open(pred_file, 'r') as f:
                            pred_data = json.load(f)
                            self.prediction_history.append(pred_data)
                    except Exception as e:
                        self.logger.warning(f"Error loading prediction file {pred_file}: {e}")
                
                self.logger.info(f"Loaded {len(self.prediction_history)} recent predictions")
            
        except Exception as e:
            self.logger.error(f"Error loading performance data: {e}")
    
    def monitor_model_performance(self) -> dict:
        """Monitor current model performance"""
        
        try:
            if not self.performance_history:
                self.logger.warning("No performance data available for monitoring")
                return {}
            
            # Convert to DataFrame for analysis
            df_perf = pd.DataFrame(self.performance_history)
            
            if df_perf.empty:
                return {}
            
            # Calculate performance metrics
            total_signals = len(df_perf)
            buy_signals = len(df_perf[df_perf['action'] == 'BUY'])
            sell_signals = len(df_perf[df_perf['action'] == 'SELL'])
            
            # Calculate recent performance (last 24 hours if available)
            if 'timestamp' in df_perf.columns:
                df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'])
                recent_cutoff = datetime.now() - pd.Timedelta(hours=24)
                df_recent = df_perf[df_perf['timestamp'] >= recent_cutoff]
                
                recent_signals = len(df_recent)
                recent_buy = len(df_recent[df_recent['action'] == 'BUY'])
                recent_sell = len(df_recent[df_recent['action'] == 'SELL'])
            else:
                recent_signals = total_signals
                recent_buy = buy_signals
                recent_sell = sell_signals
            
            metrics = {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'recent_signals_24h': recent_signals,
                'recent_buy_signals_24h': recent_buy,
                'recent_sell_signals_24h': recent_sell,
                'monitoring_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Model performance metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring model performance: {e}")
            return {}
    
    def monitor_data_quality(self, latest_data: pd.DataFrame) -> dict:
        """Monitor data quality of incoming market data"""
        
        try:
            if latest_data is None or latest_data.empty:
                self.logger.warning("No data provided for quality monitoring")
                return {}
            
            # Check for missing values
            missing_values = latest_data.isnull().sum().to_dict()
            
            # Check for data freshness
            if 'timestamp' in latest_data.columns:
                latest_timestamp = latest_data['timestamp'].max()
                data_freshness_minutes = (datetime.now() - latest_timestamp).total_seconds() / 60
            else:
                data_freshness_minutes = 0
            
            # Check for outliers (using Z-score)
            numeric_cols = latest_data.select_dtypes(include=[np.number]).columns
            outliers = {}
            
            for col in numeric_cols:
                if col != 'timestamp':
                    z_scores = np.abs((latest_data[col] - latest_data[col].mean()) / latest_data[col].std())
                    outliers[col] = int((z_scores > 3).sum())
            
            quality_metrics = {
                'missing_values': missing_values,
                'data_freshness_minutes': data_freshness_minutes,
                'outliers': outliers,
                'total_records': len(latest_data),
                'monitoring_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Data quality metrics: {quality_metrics}")
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring data quality: {e}")
            return {}
    
    def detect_model_drift(self, latest_predictions: dict) -> dict:
        """Detect model drift in predictions"""
        
        try:
            if not latest_predictions or not self.prediction_history:
                self.logger.warning("Insufficient data for drift detection")
                return {}
            
            # Add latest prediction to history
            self.prediction_history.append(latest_predictions)
            
            # Keep only recent predictions (last 100)
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            # Calculate prediction statistics
            if len(self.prediction_history) >= 10:
                # Get ensemble predictions
                ensemble_preds = [
                    pred['predictions'].get('ensemble', 0.5) 
                    for pred in self.prediction_history[-50:] 
                    if 'predictions' in pred and 'ensemble' in pred['predictions']
                ]
                
                if ensemble_preds:
                    mean_pred = np.mean(ensemble_preds)
                    std_pred = np.std(ensemble_preds)
                    
                    # Check for drift (significant change in prediction distribution)
                    drift_detected = abs(mean_pred - 0.5) > 0.1 or std_pred < 0.1
                    
                    drift_metrics = {
                        'mean_prediction': float(mean_pred),
                        'std_prediction': float(std_pred),
                        'drift_detected': drift_detected,
                        'total_predictions_monitored': len(ensemble_preds),
                        'monitoring_timestamp': datetime.now().isoformat()
                    }
                    
                    if drift_detected:
                        self.logger.warning(f"Model drift detected: {drift_metrics}")
                    else:
                        self.logger.info(f"No significant model drift detected")
                    
                    return drift_metrics
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error detecting model drift: {e}")
            return {}
    
    def generate_monitoring_report(self, model_metrics: dict, data_metrics: dict, drift_metrics: dict) -> str:
        """Generate comprehensive monitoring report"""
        
        try:
            report = "=" * 80 + "\n"
            report += "BTCUSD MODEL MONITORING REPORT\n"
            report += "=" * 80 + "\n"
            report += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Model Performance Section
            report += "MODEL PERFORMANCE\n"
            report += "-" * 40 + "\n"
            if model_metrics:
                report += f"Total Signals: {model_metrics.get('total_signals', 0)}\n"
                report += f"Buy Signals: {model_metrics.get('buy_signals', 0)}\n"
                report += f"Sell Signals: {model_metrics.get('sell_signals', 0)}\n"
                report += f"Recent Signals (24h): {model_metrics.get('recent_signals_24h', 0)}\n"
            else:
                report += "No performance metrics available\n"
            report += "\n"
            
            # Data Quality Section
            report += "DATA QUALITY\n"
            report += "-" * 40 + "\n"
            if data_metrics:
                report += f"Total Records: {data_metrics.get('total_records', 0)}\n"
                report += f"Data Freshness: {data_metrics.get('data_freshness_minutes', 0):.2f} minutes\n"
                report += f"Missing Values: {sum(data_metrics.get('missing_values', {}).values())}\n"
                
                # Outliers summary
                outliers = data_metrics.get('outliers', {})
                if outliers:
                    total_outliers = sum(outliers.values())
                    report += f"Total Outliers: {total_outliers}\n"
            else:
                report += "No data quality metrics available\n"
            report += "\n"
            
            # Model Drift Section
            report += "MODEL DRIFT\n"
            report += "-" * 40 + "\n"
            if drift_metrics:
                report += f"Mean Prediction: {drift_metrics.get('mean_prediction', 0):.4f}\n"
                report += f"Prediction Std Dev: {drift_metrics.get('std_prediction', 0):.4f}\n"
                report += f"Drift Detected: {'YES' if drift_metrics.get('drift_detected', False) else 'NO'}\n"
                report += f"Predictions Monitored: {drift_metrics.get('total_predictions_monitored', 0)}\n"
            else:
                report += "No drift metrics available\n"
            report += "\n"
            
            # Recommendations
            report += "RECOMMENDATIONS\n"
            report += "-" * 40 + "\n"
            
            # Performance recommendations
            if model_metrics:
                if model_metrics.get('total_signals', 0) < 10:
                    report += "* Insufficient signals for performance evaluation\n"
                
                recent_signals = model_metrics.get('recent_signals_24h', 0)
                if recent_signals == 0:
                    report += "* No recent signals - check data pipeline\n"
                elif recent_signals < 5:
                    report += "* Low signal frequency - review model sensitivity\n"
            
            # Data quality recommendations
            if data_metrics:
                freshness = data_metrics.get('data_freshness_minutes', 0)
                if freshness > 10:
                    report += "* Data freshness issue - check data source connection\n"
                
                missing_vals = sum(data_metrics.get('missing_values', {}).values())
                if missing_vals > 0:
                    report += "* Missing values detected - review data preprocessing\n"
                
                outliers = sum(data_metrics.get('outliers', {}).values())
                if outliers > 5:
                    report += "* High number of outliers - review data quality\n"
            
            # Drift recommendations
            if drift_metrics:
                if drift_metrics.get('drift_detected', False):
                    report += "* Model drift detected - consider retraining\n"
                
                pred_std = drift_metrics.get('std_prediction', 0)
                if pred_std < 0.05:
                    report += "* Low prediction variance - model may be underperforming\n"
            
            if not any([model_metrics, data_metrics, drift_metrics]):
                report += "* No metrics available - check monitoring pipeline\n"
            
            report += "\n"
            
            # Save report
            report_file = self.monitoring_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Monitoring report saved to: {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating monitoring report: {e}")
            return ""
    
    def plot_performance_dashboard(self, model_metrics: dict, data_metrics: dict, drift_metrics: dict):
        """Plot performance dashboard with key metrics"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('BTCUSD Model Monitoring Dashboard', fontsize=16)
            
            # 1. Signal distribution
            if model_metrics:
                signals = [model_metrics.get('buy_signals', 0), model_metrics.get('sell_signals', 0)]
                labels = ['Buy', 'Sell']
                axes[0, 0].pie(signals, labels=labels, autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title('Signal Distribution')
            else:
                axes[0, 0].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[0, 0].set_title('Signal Distribution')
            
            # 2. Data quality metrics
            if data_metrics:
                metrics = [
                    data_metrics.get('total_records', 0),
                    data_metrics.get('data_freshness_minutes', 0),
                    sum(data_metrics.get('missing_values', {}).values()),
                    sum(data_metrics.get('outliers', {}).values())
                ]
                labels = ['Records', 'Freshness (min)', 'Missing', 'Outliers']
                axes[0, 1].bar(range(len(metrics)), metrics)
                axes[0, 1].set_xticks(range(len(metrics)))
                axes[0, 1].set_xticklabels(labels, rotation=45)
                axes[0, 1].set_title('Data Quality Metrics')
            else:
                axes[0, 1].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[0, 1].set_title('Data Quality Metrics')
            
            # 3. Prediction distribution (if available)
            if self.prediction_history and len(self.prediction_history) > 10:
                ensemble_preds = [
                    pred['predictions'].get('ensemble', 0.5) 
                    for pred in self.prediction_history[-50:] 
                    if 'predictions' in pred and 'ensemble' in pred['predictions']
                ]
                
                if ensemble_preds:
                    axes[1, 0].hist(ensemble_preds, bins=20, alpha=0.7)
                    axes[1, 0].set_xlabel('Prediction Probability')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].set_title('Prediction Distribution')
                    axes[1, 0].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
                    axes[1, 0].legend()
                else:
                    axes[1, 0].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[1, 0].set_title('Prediction Distribution')
            else:
                axes[1, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                axes[1, 0].set_title('Prediction Distribution')
            
            # 4. Recent signals timeline
            if self.performance_history:
                df_perf = pd.DataFrame(self.performance_history)
                if 'timestamp' in df_perf.columns and len(df_perf) > 5:
                    df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'])
                    df_recent = df_perf.tail(50)  # Last 50 signals
                    
                    # Plot timeline
                    axes[1, 1].scatter(df_recent['timestamp'], 
                                     [1 if action == 'BUY' else 0 for action in df_recent['action']], 
                                     c=['green' if action == 'BUY' else 'red' for action in df_recent['action']],
                                     alpha=0.7)
                    axes[1, 1].set_xlabel('Time')
                    axes[1, 1].set_ylabel('Signal (0=Sell, 1=Buy)')
                    axes[1, 1].set_title('Recent Signals Timeline')
                    axes[1, 1].set_yticks([0, 1])
                    axes[1, 1].set_yticklabels(['Sell', 'Buy'])
                else:
                    axes[1, 1].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[1, 1].set_title('Recent Signals Timeline')
            else:
                axes[1, 1].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[1, 1].set_title('Recent Signals Timeline')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.monitoring_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Monitoring dashboard saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting performance dashboard: {e}")
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        
        try:
            self.logger.info("Starting monitoring cycle")
            
            # Monitor model performance
            model_metrics = self.monitor_model_performance()
            
            # For data quality monitoring, we would need access to latest data
            # This would typically be passed from the deployment pipeline
            data_metrics = {}
            
            # Detect model drift
            # This would use the latest predictions from the deployment pipeline
            drift_metrics = {}
            
            # Generate report
            report = self.generate_monitoring_report(model_metrics, data_metrics, drift_metrics)
            
            # Plot dashboard
            self.plot_performance_dashboard(model_metrics, data_metrics, drift_metrics)
            
            self.logger.info("Monitoring cycle completed")
            
            return {
                'model_metrics': model_metrics,
                'data_metrics': data_metrics,
                'drift_metrics': drift_metrics,
                'report': report
            }
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
            return {}
    
    def start_continuous_monitoring(self, interval_seconds: int = 3600):
        """Start continuous monitoring in a separate thread"""
        
        def monitoring_loop():
            while True:
                try:
                    self.run_monitoring_cycle()
                    time.sleep(interval_seconds)
                except KeyboardInterrupt:
                    self.logger.info("Continuous monitoring stopped by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error in continuous monitoring: {e}")
                    time.sleep(interval_seconds)
        
        try:
            thread = threading.Thread(target=monitoring_loop, daemon=True)
            thread.start()
            self.logger.info(f"Continuous monitoring started (interval: {interval_seconds}s)")
            return thread
            
        except Exception as e:
            self.logger.error(f"Error starting continuous monitoring: {e}")
            return None
    
    def get_alerts(self) -> list:
        """Get current alerts based on monitoring metrics"""
        
        try:
            alerts = []
            
            # Check for performance issues
            if self.performance_history:
                df_perf = pd.DataFrame(self.performance_history)
                if len(df_perf) > 10:
                    # Check for no recent signals
                    if 'timestamp' in df_perf.columns:
                        df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'])
                        recent_cutoff = datetime.now() - pd.Timedelta(hours=1)
                        recent_signals = df_perf[df_perf['timestamp'] >= recent_cutoff]
                        
                        if len(recent_signals) == 0:
                            alerts.append({
                                'type': 'WARNING',
                                'message': 'No signals generated in the last hour',
                                'timestamp': datetime.now().isoformat()
                            })
            
            # Check for data quality issues
            # This would be implemented when we have actual data quality metrics
            
            # Check for model drift
            # This would be implemented when we have actual drift metrics
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            return []
    
    def save_monitoring_state(self):
        """Save current monitoring state"""
        
        try:
            state = {
                'performance_history': self.performance_history,
                'prediction_history': self.prediction_history,
                'last_save_timestamp': datetime.now().isoformat()
            }
            
            state_file = self.monitoring_dir / "monitoring_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"Monitoring state saved to: {state_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving monitoring state: {e}")
    
    def load_monitoring_state(self):
        """Load monitoring state"""
        
        try:
            state_file = self.monitoring_dir / "monitoring_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.performance_history = state.get('performance_history', [])
                self.prediction_history = state.get('prediction_history', [])
                
                self.logger.info(f"Monitoring state loaded from: {state_file}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error loading monitoring state: {e}")
            return False
