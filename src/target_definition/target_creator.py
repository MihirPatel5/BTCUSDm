"""
Target Variable Definition Module
Creates target variables for BTCUSD prediction model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

class TargetCreator:
    """Creator of target variables for different prediction approaches"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/processed/selected_features")
        
    def create_targets(self) -> pd.DataFrame:
        """Create all configured target variables"""
        
        try:
            # Load data with selected features
            data_file = self.data_dir / "BTCUSD_5min_selected_features.csv"
            if not data_file.exists():
                self.logger.error(f"Selected features data file not found: {data_file}")
                return None
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Creating target variables for {len(data)} records")
            
            # Get prediction horizon from config
            prediction_horizon = self.config.get('data', {}).get('prediction_horizon_minutes', 15)
            periods_ahead = prediction_horizon // 5  # 5-minute intervals
            
            # Create binary classification target (recommended approach)
            data = self._create_binary_target(data, periods_ahead)
            
            # Create regression target (price movement magnitude)
            data = self._create_regression_target(data, periods_ahead)
            
            # Create multi-class target
            data = self._create_multiclass_target(data, periods_ahead)
            
            # Save dataset with all targets
            targets_dir = Path("data/processed/with_targets")
            targets_dir.mkdir(parents=True, exist_ok=True)
            output_file = targets_dir / "BTCUSD_5min_with_targets.csv"
            data.to_csv(output_file, index=False)
            
            self.logger.info(f"Target variables created and saved to: {output_file}")
            
            # Log target distributions
            self._log_target_distributions(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating target variables: {e}")
            return None
    
    def _create_binary_target(self, data: pd.DataFrame, periods_ahead: int) -> pd.DataFrame:
        """Create binary classification target (up/down prediction)"""
        
        # Binary classification: 1 if price goes up, 0 if down
        future_price = data['close'].shift(-periods_ahead)
        data['target_binary'] = (future_price > data['close']).astype(int)
        
        # Remove last periods_ahead rows as they have NaN targets
        data = data.iloc[:-periods_ahead].copy()
        
        self.logger.info(f"Binary target created: {data['target_binary'].sum()} up moves, "
                        f"{len(data) - data['target_binary'].sum()} down moves")
        
        return data
    
    def _create_regression_target(self, data: pd.DataFrame, periods_ahead: int) -> pd.DataFrame:
        """Create regression target (price movement percentage)"""
        
        # Percentage change over n periods
        future_price = data['close'].shift(-periods_ahead)
        data['target_regression'] = (future_price - data['close']) / data['close'] * 100
        
        # Remove last periods_ahead rows as they have NaN targets
        data = data.iloc[:-periods_ahead].copy()
        
        self.logger.info(f"Regression target created: mean={data['target_regression'].mean():.4f}%, "
                        f"std={data['target_regression'].std():.4f}%")
        
        return data
    
    def _create_multiclass_target(self, data: pd.DataFrame, periods_ahead: int) -> pd.DataFrame:
        """Create multi-class classification target"""
        
        # Percentage change over n periods
        future_price = data['close'].shift(-periods_ahead)
        pct_change = (future_price - data['close']) / data['close'] * 100
        
        # Multi-class: Strong Down, Down, Neutral, Up, Strong Up
        # Thresholds from config or defaults
        thresholds = self.config.get('target', {}).get('multiclass_thresholds', [-2, -0.5, 0.5, 2])
        
        # Create categorical target
        conditions = [
            pct_change <= thresholds[0],
            (pct_change > thresholds[0]) & (pct_change <= thresholds[1]),
            (pct_change > thresholds[1]) & (pct_change <= thresholds[2]),
            (pct_change > thresholds[2]) & (pct_change <= thresholds[3]),
            pct_change > thresholds[3]
        ]
        
        choices = ['Strong_Down', 'Down', 'Neutral', 'Up', 'Strong_Up']
        data['target_multiclass'] = np.select(conditions, choices, default='Neutral')
        
        # Convert to numeric for easier processing
        class_mapping = {'Strong_Down': 0, 'Down': 1, 'Neutral': 2, 'Up': 3, 'Strong_Up': 4}
        data['target_multiclass_numeric'] = data['target_multiclass'].map(class_mapping)
        
        # Remove last periods_ahead rows as they have NaN targets
        data = data.iloc[:-periods_ahead].copy()
        
        self.logger.info(f"Multi-class target created: {data['target_multiclass'].value_counts().to_dict()}")
        
        return data
    
    def _log_target_distributions(self, data: pd.DataFrame):
        """Log distributions of created target variables"""
        
        if 'target_binary' in data.columns:
            up_pct = data['target_binary'].mean() * 100
            self.logger.info(f"Binary target distribution: {up_pct:.2f}% up, {100-up_pct:.2f}% down")
        
        if 'target_regression' in data.columns:
            self.logger.info(f"Regression target stats: mean={data['target_regression'].mean():.4f}%, "
                            f"std={data['target_regression'].std():.4f}%, "
                            f"min={data['target_regression'].min():.4f}%, "
                            f"max={data['target_regression'].max():.4f}%")
        
        if 'target_multiclass' in data.columns:
            self.logger.info(f"Multi-class target distribution: {data['target_multiclass'].value_counts().to_dict()}")
    
    def analyze_prediction_horizon_impact(self) -> dict:
        """Analyze how different prediction horizons affect target distributions"""
        
        try:
            # Load base data
            data_file = Path("data/processed/BTCUSD_5min_processed.csv")
            if not data_file.exists():
                self.logger.error(f"Base data file not found: {data_file}")
                return {}
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Test different horizons
            horizons = [5, 10, 15, 20, 25, 30]  # Minutes
            results = {}
            
            for horizon in horizons:
                periods = horizon // 5
                future_price = data['close'].shift(-periods)
                target = (future_price > data['close']).astype(int)
                target = target.iloc[:-periods]  # Remove NaN rows
                
                up_pct = target.mean() * 100
                results[horizon] = {
                    'up_percentage': up_pct,
                    'down_percentage': 100 - up_pct,
                    'total_samples': len(target)
                }
                
                self.logger.info(f"Horizon {horizon}min: {up_pct:.2f}% up, {100-up_pct:.2f}% down ({len(target)} samples)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction horizon impact: {e}")
            return {}
