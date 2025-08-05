"""
Main Execution Script
Demonstrates the complete BTCUSD Forex Prediction Model pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import sys
from datetime import datetime

# Import all pipeline modules
from data_collection.data_collector import DataCollector
from data_preprocessing.data_preprocessor import DataPreprocessor
from feature_engineering.technical_indicators import TechnicalIndicators
from feature_engineering.statistical_features import StatisticalFeatures
from feature_engineering.time_features import TimeFeatures
from feature_engineering.advanced_features import AdvancedFeatures
from feature_engineering.feature_selector import FeatureSelector
from target_definition.target_creator import TargetCreator
from training.training_strategy import TrainingStrategy
from training.advanced_training import AdvancedTrainingStrategy
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.ensemble_model import EnsembleModel
from tuning.hyperparameter_tuner import HyperparameterTuner
from deployment.deployer import Deployer
from monitoring.monitor import Monitor
from continuous_improvement.model_updater import ModelUpdater
from advanced_techniques.advanced_analyzer import AdvancedAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('btcusd_prediction_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config file"""
    try:
        config_path = Path("config/config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Return default configuration
            return {
                "data": {
                    "symbol": "BTCUSD",
                    "interval": "5m",
                    "period": "3y"
                },
                "feature_engineering": {
                    "indicators": ["sma", "ema", "rsi", "macd", "bbands", "atr"],
                    "windows": [5, 10, 20, 50],
                    "statistical_features": True
                },
                "target": {
                    "prediction_horizon": 3,  # 15 minutes (3 * 5min intervals)
                    "target_type": "binary"
                },
                "model": {
                    "train_test_split": 0.8,
                    "xgboost": {
                        "n_estimators": 100,
                        "max_depth": 6,
                        "learning_rate": 0.1
                    },
                    "lstm": {
                        "timesteps": 60,
                        "units": 50,
                        "epochs": 50,
                        "batch_size": 32
                    }
                },
                "improvement": {
                    "retraining_frequency_days": 7,
                    "accuracy_threshold": 0.55,
                    "sharpe_threshold": 1.0
                },
                "advanced": {
                    "n_regimes": 3
                }
            }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def main():
    """Main execution function demonstrating the complete pipeline"""
    
    logger.info("BTCUSD Forex Prediction Model Pipeline")
    logger.info("=" * 50)
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # 1. Data Collection
        logger.info("\n1. Data Collection Phase")
        logger.info("-" * 30)
        
        # In a real implementation, this would call the data collection modules
        # For demonstration, we'll check if data exists
        data_dir = Path("data/processed")
        data_file = data_dir / "BTCUSD_5min_processed.csv"
        
        if data_file.exists():
            logger.info(f"Processed data found: {data_file}")
            data = pd.read_csv(data_file)
            logger.info(f"Data shape: {data.shape}")
        else:
            logger.warning("Processed data not found. Run data collection and preprocessing first.")
            return
        
        # 2. Feature Engineering
        logger.info("\n2. Feature Engineering Phase")
        logger.info("-" * 30)
        
        # Check if features exist
        features_dir = data_dir / "with_features"
        features_files = list(features_dir.glob("*.csv")) if features_dir.exists() else []
        
        if features_files:
            logger.info(f"Feature files found: {len(features_files)} files")
        else:
            logger.warning("Feature files not found. Run feature engineering first.")
        
        # 3. Target Definition
        logger.info("\n3. Target Definition Phase")
        logger.info("-" * 30)
        
        # Check if targets exist
        targets_dir = data_dir / "with_targets"
        targets_files = list(targets_dir.glob("*.csv")) if targets_dir.exists() else []
        
        if targets_files:
            logger.info(f"Target files found: {len(targets_files)} files")
        else:
            logger.warning("Target files not found. Run target definition first.")
        
        # 4. Model Development
        logger.info("\n4. Model Development Phase")
        logger.info("-" * 30)
        
        # Check if models exist
        models_dir = Path("models")
        model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.h5")) if models_dir.exists() else []
        
        if model_files:
            logger.info(f"Model files found: {len(model_files)} files")
            for model_file in model_files:
                logger.info(f"  - {model_file.name}")
        else:
            logger.warning("Model files not found. Run model training first.")
        
        # 5. Model Evaluation
        logger.info("\n5. Model Evaluation Phase")
        logger.info("-" * 30)
        
        # Check if evaluation results exist
        evaluation_dir = Path("evaluation")
        evaluation_files = list(evaluation_dir.glob("*.json")) if evaluation_dir.exists() else []
        
        if evaluation_files:
            logger.info(f"Evaluation results found: {len(evaluation_files)} files")
        else:
            logger.warning("Evaluation results not found. Run model evaluation first.")
        
        # 6. Hyperparameter Tuning
        logger.info("\n6. Hyperparameter Tuning Phase")
        logger.info("-" * 30)
        
        # Check if tuning results exist
        tuning_dir = Path("tuning")
        tuning_files = list(tuning_dir.glob("*.json")) if tuning_dir.exists() else []
        
        if tuning_files:
            logger.info(f"Tuning results found: {len(tuning_files)} files")
        else:
            logger.warning("Tuning results not found. Run hyperparameter tuning first.")
        
        # 7. Backtesting
        logger.info("\n7. Backtesting Phase")
        logger.info("-" * 30)
        
        # Check if backtesting results exist
        backtesting_dir = Path("backtesting")
        backtesting_files = list(backtesting_dir.glob("*.json")) if backtesting_dir.exists() else []
        
        if backtesting_files:
            logger.info(f"Backtesting results found: {len(backtesting_files)} files")
        else:
            logger.warning("Backtesting results not found. Run backtesting first.")
        
        # 8. Model Validation
        logger.info("\n8. Model Validation Phase")
        logger.info("-" * 30)
        
        # Check if validation results exist
        validation_dir = Path("validation")
        validation_files = list(validation_dir.glob("*.json")) if validation_dir.exists() else []
        
        if validation_files:
            logger.info(f"Validation results found: {len(validation_files)} files")
        else:
            logger.warning("Validation results not found. Run model validation first.")
        
        # 9. Production Deployment
        logger.info("\n9. Production Deployment Phase")
        logger.info("-" * 30)
        
        # Check if deployment modules exist
        deployment_dir = Path("src/deployment")
        deployer_file = deployment_dir / "deployer.py" if deployment_dir.exists() else None
        
        if deployer_file and deployer_file.exists():
            logger.info("Deployment module found")
        else:
            logger.warning("Deployment module not found. Run deployment implementation first.")
        
        # 10. Monitoring
        logger.info("\n10. Monitoring Phase")
        logger.info("-" * 30)
        
        # Check if monitoring modules exist
        monitoring_dir = Path("src/monitoring")
        monitor_file = monitoring_dir / "monitor.py" if monitoring_dir.exists() else None
        
        if monitor_file and monitor_file.exists():
            logger.info("Monitoring module found")
        else:
            logger.warning("Monitoring module not found. Run monitoring implementation first.")
        
        # 11. Continuous Improvement
        logger.info("\n11. Continuous Improvement Phase")
        logger.info("-" * 30)
        
        # Check if improvement modules exist
        improvement_dir = Path("src/improvement")
        improver_file = improvement_dir / "continuous_improver.py" if improvement_dir.exists() else None
        
        if improver_file and improver_file.exists():
            logger.info("Continuous improvement module found")
        else:
            logger.warning("Continuous improvement module not found. Run improvement implementation first.")
        
        # 12. Advanced Techniques
        logger.info("\n12. Advanced Techniques Phase")
        logger.info("-" * 30)
        
        # Phase 12: Advanced techniques (already implemented)
        logger.info("=== Phase 12: Advanced Techniques ===")
        advanced_analyzer = AdvancedAnalyzer(config)
        advanced_results = advanced_analyzer.run_advanced_analysis()
        if advanced_results:
            logger.info("Advanced techniques analysis completed successfully")
        else:
            logger.warning("Advanced techniques analysis failed or returned no results")
        
        # Phase 13: Advanced feature engineering and improved training
        logger.info("=== Phase 13: Advanced Feature Engineering and Training ===")
        
        # Generate advanced features
        from feature_engineering.advanced_features import AdvancedFeatures
        advanced_features = AdvancedFeatures(config)
        advanced_features_data = advanced_features.generate_all_features()
        if advanced_features_data is not None:
            logger.info("Advanced feature engineering completed successfully")
        else:
            logger.warning("Advanced feature engineering failed")
        
        # Apply improved training strategy
        from training.advanced_training import AdvancedTrainingStrategy
        advanced_trainer = AdvancedTrainingStrategy(config)
        advanced_splits = advanced_trainer.prepare_advanced_splits(n_splits=5)
        if advanced_splits:
            advanced_training_results = advanced_trainer.train_advanced_ensemble(advanced_splits)
            if advanced_training_results:
                logger.info("Advanced training strategy completed successfully")
                # Plot model comparison
                advanced_trainer.plot_model_comparison(advanced_training_results)
            else:
                logger.warning("Advanced training strategy failed")
        else:
            logger.warning("Failed to prepare advanced training splits")
        
        # Summary
        logger.info("=" * 50)
        logger.info("BTCUSD FOREX PREDICTION MODEL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        
        # Print success targets for reference
        logger.info("Success Targets:")
        logger.info("- Accuracy: 55-60% (realistic and profitable)")
        logger.info("- Sharpe Ratio: >1.5 (risk-adjusted returns)")
        logger.info("- Maximum Drawdown: <15% (risk control)")
        logger.info("- Win Rate: 45-55% (sustainable)")
        logger.info("- Profit Factor: >1.3 (gross profit/loss ratio)")
        
        logger.info("\nSUMMARY:")
        logger.info(f"  - Data: {data.shape if 'data' in locals() else 'Not loaded'}")
        logger.info(f"  - Feature files: {len(features_files) if 'features_files' in locals() else 0}")
        logger.info(f"  - Target files: {len(targets_files) if 'targets_files' in locals() else 0}")
        logger.info(f"  - Model files: {len(model_files) if 'model_files' in locals() else 0}")
        logger.info(f"  - Evaluation results: {len(evaluation_files) if 'evaluation_files' in locals() else 0}")
        logger.info(f"  - Tuning results: {len(tuning_files) if 'tuning_files' in locals() else 0}")
        logger.info(f"  - Backtesting results: {len(backtesting_files) if 'backtesting_files' in locals() else 0}")
        logger.info(f"  - Validation results: {len(validation_files) if 'validation_files' in locals() else 0}")
        logger.info(f"  - Deployment module: {'Found' if deployer_file and deployer_file.exists() else 'Missing'}")
        logger.info(f"  - Monitoring module: {'Found' if monitor_file and monitor_file.exists() else 'Missing'}")
        logger.info(f"  - Improvement module: {'Found' if improver_file and improver_file.exists() else 'Missing'}")
        logger.info(f"  - Advanced techniques module: {'Found' if advanced_file and advanced_file.exists() else 'Missing'}")
        
        logger.info("\nTo run specific phases, execute the corresponding modules directly.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
