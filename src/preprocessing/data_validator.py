"""
Data Validation Module
Validates data quality and integrity after preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional

class DataValidator:
    """Data validator for forex market data quality checks"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data")
        
    def validate_processed_data(self, data: pd.DataFrame) -> Dict[str, any]:
        """Run comprehensive validation on processed data"""
        
        validation_results = {
            'passed': True,
            'issues': [],
            'statistics': {},
            'recommendations': []
        }
        
        try:
            self.logger.info("Starting comprehensive data validation...")
            
            # Basic data checks
            basic_checks = self._run_basic_checks(data)
            validation_results['statistics'].update(basic_checks)
            
            # Price integrity checks
            price_checks = self._run_price_integrity_checks(data)
            if not price_checks['passed']:
                validation_results['passed'] = False
                validation_results['issues'].extend(price_checks['issues'])
            
            # Volume checks
            volume_checks = self._run_volume_checks(data)
            if not volume_checks['passed']:
                validation_results['passed'] = False
                validation_results['issues'].extend(volume_checks['issues'])
            
            # Temporal consistency checks
            temporal_checks = self._run_temporal_checks(data)
            if not temporal_checks['passed']:
                validation_results['passed'] = False
                validation_results['issues'].extend(temporal_checks['issues'])
            
            # Statistical anomaly checks
            anomaly_checks = self._run_anomaly_checks(data)
            if not anomaly_checks['passed']:
                validation_results['passed'] = False
                validation_results['issues'].extend(anomaly_checks['issues'])
            
            # Generate recommendations
            validation_results['recommendations'] = self._generate_recommendations(
                validation_results['issues']
            )
            
            # Save validation report
            self._save_validation_report(validation_results)
            
            status = "PASSED" if validation_results['passed'] else "FAILED"
            self.logger.info(f"Data validation {status}: {len(validation_results['issues'])} issues found")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error during data validation: {e}")
            validation_results['passed'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
            return validation_results
    
    def _run_basic_checks(self, data: pd.DataFrame) -> Dict[str, any]:
        """Run basic data quality checks"""
        
        stats = {}
        
        # Data shape
        stats['total_records'] = len(data)
        stats['date_range'] = f"{data['timestamp'].min()} to {data['timestamp'].max()}"
        
        # Missing values
        missing_counts = data.isnull().sum()
        stats['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        
        # Data types
        stats['data_types'] = data.dtypes.to_dict()
        
        # Duplicate timestamps
        duplicate_count = data.duplicated(subset=['timestamp']).sum()
        stats['duplicate_timestamps'] = duplicate_count
        
        if duplicate_count > 0:
            self.logger.warning(f"Found {duplicate_count} duplicate timestamps")
        
        return stats
    
    def _run_price_integrity_checks(self, data: pd.DataFrame) -> Dict[str, any]:
        """Run price integrity validation checks"""
        
        result = {'passed': True, 'issues': []}
        
        # Check for negative prices
        negative_prices = (data[['open', 'high', 'low', 'close']] < 0).any(axis=1)
        if negative_prices.any():
            count = negative_prices.sum()
            result['issues'].append(f"Found {count} records with negative prices")
            result['passed'] = False
            self.logger.error(f"Data validation failed: {count} records with negative prices")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            count = invalid_ohlc.sum()
            result['issues'].append(f"Found {count} records with invalid OHLC relationships")
            self.logger.warning(f"Found {count} records with invalid OHLC relationships")
        
        # Check for extreme price changes
        data = data.copy()
        data['price_change'] = data['close'].pct_change()
        
        # 20%+ price changes in 5 minutes are suspicious
        extreme_changes = abs(data['price_change']) > 0.20
        if extreme_changes.any():
            count = extreme_changes.sum()
            result['issues'].append(f"Found {count} records with extreme price changes (>20% in 5min)")
            self.logger.warning(f"Found {count} records with extreme price changes (>20% in 5min)")
        
        return result
    
    def _run_volume_checks(self, data: pd.DataFrame) -> Dict[str, any]:
        """Run volume data validation checks"""
        
        result = {'passed': True, 'issues': []}
        
        # Check for negative volume
        if 'volume' in data.columns:
            negative_volume = data['volume'] < 0
            if negative_volume.any():
                count = negative_volume.sum()
                result['issues'].append(f"Found {count} records with negative volume")
                result['passed'] = False
                self.logger.error(f"Data validation failed: {count} records with negative volume")
        
        # Check for zero volume (may be valid but worth noting)
        if 'volume' in data.columns:
            zero_volume = data['volume'] == 0
            if zero_volume.any():
                count = zero_volume.sum()
                result['issues'].append(f"Found {count} records with zero volume")
                self.logger.info(f"Found {count} records with zero volume")
        
        return result
    
    def _run_temporal_checks(self, data: pd.DataFrame) -> Dict[str, any]:
        """Run temporal consistency checks"""
        
        result = {'passed': True, 'issues': []}
        
        # Check for gaps in timestamp sequence
        data = data.sort_values('timestamp').copy()
        data['time_diff'] = data['timestamp'].diff()
        
        # Expected time difference for 5-minute data
        expected_diff = pd.Timedelta(minutes=5)
        
        # Large gaps (more than 30 minutes)
        large_gaps = data['time_diff'] > pd.Timedelta(minutes=30)
        if large_gaps.any():
            count = large_gaps.sum()
            result['issues'].append(f"Found {count} large time gaps (>30 minutes)")
            self.logger.info(f"Found {count} large time gaps (>30 minutes)")
        
        # Check for non-standard intervals
        non_standard = data['time_diff'] != expected_diff
        # Exclude first row (NaN diff) and large gaps
        non_standard = non_standard & ~large_gaps & (data['time_diff'].notna())
        
        if non_standard.any():
            count = non_standard.sum()
            result['issues'].append(f"Found {count} non-standard time intervals")
            self.logger.info(f"Found {count} non-standard time intervals")
        
        return result
    
    def _run_anomaly_checks(self, data: pd.DataFrame) -> Dict[str, any]:
        """Run statistical anomaly detection"""
        
        result = {'passed': True, 'issues': []}
        
        # Price outliers (beyond 4 standard deviations)
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                z_scores = abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > 4
                if outliers.any():
                    count = outliers.sum()
                    result['issues'].append(f"Found {count} {col} price outliers (>4 std dev)")
                    self.logger.info(f"Found {count} {col} price outliers (>4 std dev)")
        
        # Volume outliers
        if 'volume' in data.columns:
            z_scores = abs((data['volume'] - data['volume'].mean()) / data['volume'].std())
            outliers = z_scores > 4
            if outliers.any():
                count = outliers.sum()
                result['issues'].append(f"Found {count} volume outliers (>4 std dev)")
                self.logger.info(f"Found {count} volume outliers (>4 std dev)")
        
        return result
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on validation issues"""
        
        recommendations = []
        
        for issue in issues:
            if "negative prices" in issue:
                recommendations.append("Investigate data source for negative price errors")
            elif "invalid OHLC relationships" in issue:
                recommendations.append("Review data cleaning process for OHLC validation")
            elif "extreme price changes" in issue:
                recommendations.append("Verify data accuracy for large price movements")
            elif "negative volume" in issue:
                recommendations.append("Check data source for volume calculation errors")
            elif "large time gaps" in issue:
                recommendations.append("Consider data interpolation for large gaps or document market closures")
            elif "outliers" in issue:
                recommendations.append("Review outliers to determine if they represent real market events or data errors")
        
        if not recommendations:
            recommendations.append("Data quality appears good for model development")
        
        return recommendations
    
    def _save_validation_report(self, results: Dict[str, any]):
        """Save validation report to file"""
        
        try:
            report_dir = self.data_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / "data_validation_report.txt"
            
            with open(report_file, 'w') as f:
                f.write("BTCUSD Data Validation Report\n")
                f.write("=" * 40 + "\n\n")
                
                f.write(f"Validation Status: {'PASSED' if results['passed'] else 'FAILED'}\n\n")
                
                f.write("Statistics:\n")
                for key, value in results['statistics'].items():
                    f.write(f"  {key}: {value}\n")
                
                f.write("\nIssues Found:\n")
                if results['issues']:
                    for issue in results['issues']:
                        f.write(f"  - {issue}\n")
                else:
                    f.write("  No issues found\n")
                
                f.write("\nRecommendations:\n")
                for rec in results['recommendations']:
                    f.write(f"  - {rec}\n")
            
            self.logger.info(f"Validation report saved to: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving validation report: {e}")
    
    def plot_data_overview(self, data: pd.DataFrame, save_plot: bool = True):
        """Create overview plots of the data"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('BTCUSD Data Overview', fontsize=16)
            
            # Price chart
            axes[0, 0].plot(data['timestamp'], data['close'])
            axes[0, 0].set_title('BTCUSD Closing Prices')
            axes[0, 0].set_ylabel('Price (USD)')
            
            # Volume chart
            if 'volume' in data.columns:
                axes[0, 1].bar(data['timestamp'], data['volume'], width=0.01)
                axes[0, 1].set_title('Trading Volume')
                axes[0, 1].set_ylabel('Volume')
            
            # Price distribution
            axes[1, 0].hist(data['close'], bins=50, alpha=0.7)
            axes[1, 0].set_title('Price Distribution')
            axes[1, 0].set_xlabel('Price (USD)')
            axes[1, 0].set_ylabel('Frequency')
            
            # Returns distribution
            returns = data['close'].pct_change().dropna()
            axes[1, 1].hist(returns, bins=50, alpha=0.7)
            axes[1, 1].set_title('Returns Distribution')
            axes[1, 1].set_xlabel('Returns')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            if save_plot:
                plot_dir = self.data_dir / "reports" / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_file = plot_dir / "data_overview.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Data overview plot saved to: {plot_file}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creating data overview plots: {e}")
