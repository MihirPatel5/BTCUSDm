"""
Model Evaluation Module
Comprehensive evaluation of BTCUSD prediction models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluator for comprehensive model performance assessment"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/training")
        self.reports_dir = Path("data/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_predictions(self, predictions_file: str) -> dict:
        """Evaluate model predictions with comprehensive metrics"""
        
        try:
            # Load predictions
            pred_path = Path(predictions_file)
            if not pred_path.exists():
                self.logger.error(f"Predictions file not found: {pred_path}")
                return {}
            
            predictions = pd.read_csv(pred_path)
            predictions['timestamp'] = pd.to_datetime(predictions['timestamp'])
            
            self.logger.info(f"Evaluating {len(predictions)} predictions")
            
            # Calculate classification metrics
            classification_metrics = self._calculate_classification_metrics(predictions)
            
            # Calculate financial metrics
            financial_metrics = self._calculate_financial_metrics(predictions)
            
            # Generate detailed reports
            self._generate_classification_report(predictions)
            self._generate_confusion_matrix_plot(predictions)
            self._generate_predictions_plot(predictions)
            
            # Combine all metrics
            all_metrics = {
                'classification': classification_metrics,
                'financial': financial_metrics
            }
            
            # Save evaluation results
            eval_file = self.reports_dir / "model_evaluation_results.json"
            import json
            with open(eval_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            self.logger.info(f"Model evaluation results saved to: {eval_file}")
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {e}")
            return {}
    
    def _calculate_classification_metrics(self, predictions: pd.DataFrame) -> dict:
        """Calculate comprehensive classification metrics"""
        
        y_true = predictions['actual']
        y_pred = predictions['predicted']
        
        # For models with probability outputs
        if 'probability' in predictions.columns:
            y_pred_proba = predictions['probability']
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        else:
            y_pred_proba = y_pred
            roc_auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc,
            'support': len(y_true)
        }
        
        self.logger.info(f"Classification metrics: {metrics}")
        
        return metrics
    
    def _calculate_financial_metrics(self, predictions: pd.DataFrame) -> dict:
        """Calculate financial performance metrics"""
        
        try:
            # Load original price data to calculate returns
            data_file = Path("data/processed/BTCUSD_5min_processed.csv")
            if not data_file.exists():
                self.logger.error(f"Original data file not found: {data_file}")
                return {}
            
            price_data = pd.read_csv(data_file)
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            
            # Merge with predictions
            merged_data = pd.merge(predictions, price_data[['timestamp', 'close']], on='timestamp', how='left')
            
            # Calculate prediction horizon returns
            prediction_horizon = self.config.get('data', {}).get('prediction_horizon_minutes', 15)
            periods_ahead = prediction_horizon // 5  # 5-minute intervals
            
            # Get future prices
            future_prices = price_data.set_index('timestamp')['close'].shift(-periods_ahead)
            merged_data = merged_data.merge(future_prices, left_on='timestamp', right_index=True, how='left', suffixes=('', '_future'))
            
            # Calculate actual returns
            merged_data['actual_return'] = (merged_data['close_future'] - merged_data['close']) / merged_data['close']
            
            # Calculate strategy returns based on predictions
            merged_data['strategy_return'] = merged_data['predicted'] * merged_data['actual_return']
            
            # Calculate cumulative returns
            merged_data['cumulative_actual'] = (1 + merged_data['actual_return']).cumprod()
            merged_data['cumulative_strategy'] = (1 + merged_data['strategy_return']).cumprod()
            
            # Financial metrics
            total_return = merged_data['strategy_return'].sum()
            annualized_return = ((1 + total_return) ** (365.25 / (len(merged_data) * 5 / (24 * 60)))) - 1  # 5-minute intervals
            
            # Strategy volatility
            volatility = merged_data['strategy_return'].std() * np.sqrt(24 * 60 / 5 * 365.25)  # Annualized
            
            # Sharpe ratio (assuming risk-free rate of 0 for crypto)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            merged_data['rolling_max'] = merged_data['cumulative_strategy'].expanding().max()
            merged_data['drawdown'] = (merged_data['cumulative_strategy'] - merged_data['rolling_max']) / merged_data['rolling_max']
            max_drawdown = merged_data['drawdown'].min()
            
            # Win rate
            win_rate = (merged_data['strategy_return'] > 0).mean()
            
            # Profit factor
            gross_profits = merged_data[merged_data['strategy_return'] > 0]['strategy_return'].sum()
            gross_losses = abs(merged_data[merged_data['strategy_return'] < 0]['strategy_return'].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'number_of_trades': len(merged_data)
            }
            
            self.logger.info(f"Financial metrics: {metrics}")
            
            # Plot equity curves
            self._plot_equity_curves(merged_data)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating financial metrics: {e}")
            return {}
    
    def _generate_classification_report(self, predictions: pd.DataFrame):
        """Generate detailed classification report"""
        
        try:
            y_true = predictions['actual']
            y_pred = predictions['predicted']
            
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Save report
            report_file = self.reports_dir / "classification_report.csv"
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(report_file)
            
            self.logger.info(f"Classification report saved to: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating classification report: {e}")
    
    def _generate_confusion_matrix_plot(self, predictions: pd.DataFrame):
        """Generate and save confusion matrix plot"""
        
        try:
            y_true = predictions['actual']
            y_pred = predictions['predicted']
            
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            plot_file = self.reports_dir / "plots" / "confusion_matrix.png"
            plot_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confusion matrix plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating confusion matrix plot: {e}")
    
    def _generate_predictions_plot(self, predictions: pd.DataFrame):
        """Generate predictions vs actual plot"""
        
        try:
            # Sample data for plotting (too many points to plot all)
            sample_data = predictions.sample(n=min(1000, len(predictions)), random_state=42)
            sample_data = sample_data.sort_values('timestamp')
            
            plt.figure(figsize=(12, 6))
            plt.scatter(sample_data['timestamp'], sample_data['actual'], alpha=0.6, label='Actual', s=10)
            plt.scatter(sample_data['timestamp'], sample_data['predicted'], alpha=0.6, label='Predicted', s=10)
            plt.xlabel('Time')
            plt.ylabel('Direction (0=Down, 1=Up)')
            plt.title('Predictions vs Actual Over Time')
            plt.legend()
            plt.tight_layout()
            
            plot_file = self.reports_dir / "plots" / "predictions_vs_actual.png"
            plot_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Predictions vs actual plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating predictions plot: {e}")
    
    def _plot_equity_curves(self, data: pd.DataFrame):
        """Plot actual vs strategy equity curves"""
        
        try:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(data['timestamp'], data['cumulative_actual'], label='Buy & Hold', linewidth=1)
            plt.plot(data['timestamp'], data['cumulative_strategy'], label='Strategy', linewidth=1)
            plt.title('Equity Curves')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(data['timestamp'], data['drawdown'], color='red', linewidth=1)
            plt.title('Drawdown')
            plt.xlabel('Time')
            plt.ylabel('Drawdown')
            plt.grid(True)
            
            plt.tight_layout()
            
            plot_file = self.reports_dir / "plots" / "equity_curves.png"
            plot_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Equity curves plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting equity curves: {e}")
    
    def compare_model_performance(self, model_results: dict) -> dict:
        """Compare performance across different models"""
        
        try:
            comparison_data = []
            
            for model_name, results in model_results.items():
                if 'metrics' in results:
                    metrics = results['metrics']
                    row = {'model': model_name}
                    row.update(metrics)
                    comparison_data.append(row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Save comparison
                comparison_file = self.reports_dir / "model_comparison.csv"
                comparison_df.to_csv(comparison_file, index=False)
                
                self.logger.info(f"Model comparison saved to: {comparison_file}")
                
                # Plot comparison
                self._plot_model_comparison(comparison_df)
                
                return comparison_df.to_dict('records')
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error comparing model performance: {e}")
            return []
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame):
        """Plot model comparison metrics"""
        
        try:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            
            plt.figure(figsize=(15, 10))
            
            for i, metric in enumerate(metrics_to_plot, 1):
                plt.subplot(2, 3, i)
                bars = plt.bar(comparison_df['model'], comparison_df[metric])
                plt.title(f'{metric.capitalize()} Comparison')
                plt.xlabel('Model')
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
            
            plot_file = self.reports_dir / "plots" / "model_comparison.png"
            plot_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Model comparison plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting model comparison: {e}")
